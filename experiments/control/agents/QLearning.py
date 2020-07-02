import numpy as np
import torch
import torch.nn.functional as f
import torch.optim as optim

from agents.Network import Network
from utils.ReplayBuffer import ReplayBuffer
from utils.torch import device, getBatchColumns

class QLearning:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.target_refresh = params['target_refresh']

        self.h1 = params['h1']
        self.h2 = params['h2']

        self.policy_net = Network(features, self.h1, self.h2, actions).to(device)
        self.target_net = Network(features, self.h1, self.h2, actions).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha, betas=(0.9, 0.99))

        self.buffer = ReplayBuffer(4000)
        self.steps = 0

        self.policy_net.cloneWeightsTo(self.target_net)

    def selectAction(self, x):
        p = np.random.rand()
        if p < self.epsilon:
            a = np.random.randint(self.actions)
            return torch.tensor(a, device=device)

        q_s, features = self.policy_net.forward(x)

        return q_s.argmax().detach()

    def updateNetwork(self, samples):
        batch = getBatchColumns(samples)

        q_s, _ = self.policy_net(batch.states)
        q_s_a = q_s.gather(1, batch.actions).squeeze()

        q_sp_ap = torch.zeros(len(samples), device=device, requires_grad=False)
        if batch.nterm_next_states.shape[0] > 0:
            q_sp, _ = self.target_net(batch.nterm_next_states)
            q_sp_ap[batch.is_non_terminals] = q_sp.max(1).values


        target = batch.rewards + batch.gamma * q_sp_ap
        loss = f.mse_loss(target, q_s_a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, s, a, sp, r, gamma):
        self.buffer.add((s, a, sp, r, gamma))
        self.steps += 1

        if self.steps % self.target_refresh == 0:
            self.policy_net.cloneWeightsTo(self.target_net)

        if len(self.buffer) > 32:
            samples, idcs = self.buffer.sample(32)
            self.updateNetwork(samples)
