import sys, os
sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy

import gym
from TDRC.DQRC import DQRC

TARGET_REFRESH = 1
ALPHA = 0.0009765

# build a new version of mountain car that doesn't have a 200 step episode cap
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=1000,
)
env = gym.make('MountainCarMyEasyVersion-v0')

# build the structure of our neural network
# we need to output both the last layer and the second to last layer
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.output = nn.Linear(32, 3)

    def forward(self, x):
        features = self.model(x)
        outputs = self.output(features)
        return outputs, features

# build the target and policy networks
# the target net is the same as the policy net, but with the weights occupying different memory
policy_network = Network()
target_network = copy.deepcopy(policy_network)
optimizer = optim.Adam(policy_network.parameters(), lr = ALPHA, betas=(0.9, 0.999))

# construct our DQRC agent
agent = DQRC(32, 3, policy_network, target_network, optimizer, {
    'alpha': ALPHA,
    'beta': 1.0,
    'epsilon': 0.1,
})

# much much faster than np.random.choice
def choice(arr, size=1):
    idxs = np.random.permutation(len(arr))
    return [arr[i] for i in idxs[:size]]

# a very simple circular replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(args)
        else:
            self.buffer[self.location] = args
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return choice(self.buffer, batch_size)

# ---------------------
# Start the experiments
# ---------------------
buffer = ReplayBuffer(4000)
s = env.reset()
x = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

episode_lengths = []
steps = 0
for step in range(100000):
    steps += 1
    if step % TARGET_REFRESH == 0:
        target_network.load_state_dict(policy_network.state_dict())

    action = agent.selectAction(x)
    sp, reward, done, info = env.step(action)

    # termination only occurs if we hit the terminal state
    # not if we hit the episode length cap
    terminated = done and steps < 1000

    xp = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)
    if terminated:
        xp = None

    a = torch.tensor(action, dtype=torch.int64)
    r = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)

    # use a state-based gamma where gamma = 0.99 of all states
    # and gamma = 0 for the terminal state
    gamma = torch.tensor(0 if terminated else 0.99)
    buffer.add((x, a, r, xp, gamma))

    # once the buffer has enough samples to start updating
    # then start updating
    if step > 32:
        samples = buffer.sample(32)
        agent.updateNetwork(samples)

    s = sp
    x = xp

    if done:
        print(steps)
        episode_lengths.append(steps)
        steps = 0
        s = env.reset()
        x = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)

env.close()
