import numpy as np
import torch

from RlGlue import RlGlue
from agents.QLearning import QLearning
from agents.QRC import QRC
from agents.QC import QC
from environments.MountainCar import MountainCar

from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper

RUNS = 10
EPISODES = 100
LEARNERS = [QRC, QC, QLearning]

COLORS = {
    'QLearning': 'blue',
    'QRC': 'purple',
    'QC': 'green',
}

# use stepsizes found in parameter study
STEPSIZES = {
    'QLearning': 0.003906,
    'QRC': 0.0009765,
    'QC': 0.0009765,
}

collector = Collector()

for run in range(RUNS):
    for Learner in LEARNERS:
        np.random.seed(run)
        torch.manual_seed(run)

        env = MountainCar()

        learner = Learner(env.features, env.num_actions, {
            'alpha': STEPSIZES[Learner.__name__],
            'epsilon': 0.1,
            'beta': 1.0,
            'target_refresh': 1,
            'buffer_size': 4000,
            'h1': 32,
            'h2': 32,
        })

        agent = RlGlueCompatWrapper(learner, gamma=0.99)

        glue = RlGlue(agent, env)

        glue.start()
        for episode in range(EPISODES):
            glue.num_steps = 0
            glue.total_reward = 0
            glue.runEpisode(max_steps=1000)

            print(Learner.__name__, run, episode, glue.num_steps)

            collector.collect(Learner.__name__, glue.total_reward)

        collector.reset()


import matplotlib.pyplot as plt
from utils.plotting import plot

ax = plt.gca()

for Learner in LEARNERS:
    name = Learner.__name__
    data = collector.getStats(name)
    plot(ax, data, label=name, color=COLORS[name])

plt.legend()
plt.show()
