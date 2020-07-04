import torch
import torch.nn.functional as f

from agents.BaseAgent import BaseAgent
from utils.torch import device, getBatchColumns

class QLearning(BaseAgent):
    def updateNetwork(self, samples):
        # organize the mini-batch so that we can request "columns" from the data
        # e.g. we can get all of the actions, or all of the states with a single call
        batch = getBatchColumns(samples)

        # compute Q(s, a) for each sample in mini-batch
        Qs, _ = self.policy_net(batch.states)
        Qsa = Qs.gather(1, batch.actions).squeeze()

        # by default Q(s', a') = 0 unless the next states are non-terminal
        Qspap = torch.zeros(len(samples), device=device, requires_grad=False)

        # if we don't have any non-terminal next states, then no need to bootstrap
        if batch.nterm_sp.shape[0] > 0:
            Qsp, _ = self.target_net(batch.nterm_sp)

            # bootstrapping term is the max Q value for the next-state
            # only assign to indices where the next state is non-terminal
            Qspap[batch.nterm] = Qsp.max(1).values


        # compute the empirical MSBE for this mini-batch and let torch auto-diff to optimize
        # don't worry about detaching the bootstrapping term for semi-gradient Q-learning
        # the target network handles that
        target = batch.rewards + batch.gamma * Qspap
        loss = 0.5 * f.mse_loss(target, Qsa)

        # update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
