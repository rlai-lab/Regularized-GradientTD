import numpy as np
import torch

from RlGlue import RlGlue
from agents.QLearning import QLearning
from environments.MountainCar import MountainCar

from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper

RUNS = 10
EPISODES = 50
LEARNERS = [QLearning]

COLOR = {
    'QLearning': 'blue',
}

collector = Collector()

for run in range(RUNS):
    for Learner in LEARNERS:
        np.random.seed(run)
        torch.manual_seed(run)

        env = MountainCar()

        learner = Learner(env.features, env.num_actions, {
            'alpha': 0.001,
            'epsilon': 0.1,
            'target_refresh': 10,
            'h1': 32,
            'h2': 32,
        })

        agent = RlGlueCompatWrapper(learner, gamma=0.99)

        glue = RlGlue(agent, env)

        glue.start()
        for episode in range(EPISODES):
            glue.num_steps = 0
            glue.total_reward = 0
            glue.runEpisode(max_steps=5000)

            print(episode, glue.num_steps)

            collector.collect(Learner.__name__, glue.total_reward)

        collector.reset()


import matplotlib.pyplot as plt

mean, stderr, runs = collector.getStats('QLearning')
plt.plot(mean)
plt.show()
