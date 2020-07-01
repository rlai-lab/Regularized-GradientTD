import torch
import torch.nn.functional as f
from algorithms.BaseLearning import BaseLearning
from utils import getBatchColumns, device

class DQRC(BaseLearning):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.beta = parameters['beta']
        self.num_actions = parameters['num_actions']
        self.num_features = self.policy_net.h1_size
        self.h = torch.zeros(self.num_actions, self.num_features, requires_grad=False).to(device)
        self.m_t = torch.zeros(self.num_actions, self.num_features, requires_grad=False).to(device)
        self.v_t = torch.zeros(self.num_actions, self.num_features, requires_grad=False).to(device)
        self.eps = 1e-8
        self.beta_1 = 0.99
        self.beta_2 = 0.999
        self.step = 0

    def update(self, sample):
        self.step += 1
        batch = getBatchColumns(sample)

        q_s = self.policy_net(batch.states)
        q_s_a = q_s.gather(1, batch.actions)
        x_s = self.policy_net.last_features

        q_sp_ap = torch.zeros(len(sample), 1, device=device)
        if batch.nterm_next_states.shape[0] > 0:
            q_sp_ap[batch.is_non_terminals] = self.target_net(batch.nterm_next_states).max(1).values.unsqueeze(1)

        target = batch.rewards + self.gamma * q_sp_ap.detach()
        td_loss = 0.5 * f.mse_loss(target, q_s_a)

        with torch.no_grad():
            # batch_size * num_actions number of delta_hats
            delta_hats = torch.matmul(x_s, self.h.t())
            delta_hat = delta_hats.gather(1, batch.actions)

        correction_loss = torch.mean(self.gamma * delta_hat * q_sp_ap)

        self.optimizer.zero_grad()
        self.target_net.zero_grad()
        td_loss.backward()

        # there is no gradient if there are no next states
        if batch.nterm_next_states.shape[0] > 0:
            correction_loss.backward()

        # add the gradients from the target network over to the policy network
        for (policy_param, target_param) in zip(self.policy_net.parameters(), self.target_net.parameters()):
            policy_param.grad.add_(target_param.grad)

        self.optimizer.step()

        # tell pytorch to not track the state of any of these
        # frees up a little bit of memory and also lets us be certain that no gradients are happening due to h
        with torch.no_grad():
            delta = target - q_s_a
            dh = (delta - delta_hat) * x_s

            for a in range(self.num_actions):
                mask = (batch.actions == a).squeeze(1)

                # if this action was never taken in the batch
                # then just skip it
                if mask.sum() == 0:
                    continue

                # update is the derivative of mean-squared error
                # plus the regularizer
                h_update = dh[mask].mean(0) - self.beta * self.h[a]

                # ADAM optimizer with bias correction
                self.v_t[a] = self.beta_2 * self.v_t[a] + (1 - self.beta_2) * (h_update**2)
                self.m_t[a] = self.beta_1 * self.m_t[a] + (1 - self.beta_1) * h_update

                m_t = self.m_t[a] / (1 - self.beta_1**self.step)
                v_t = self.v_t[a] / (1 - self.beta_2**self.step)

                self.h[a] = self.h[a] + self.alpha_h * m_t / (torch.sqrt(v_t) + self.eps)

        tde = target - q_s_a
        return tde.detach().abs().cpu().squeeze().tolist(), self.h.norm().cpu().tolist(), q_s.detach().abs().max().cpu().tolist()
