import numpy as np

from RlGlue import RlGlue
from utils.Collector import Collector
from utils.policies import actionArrayToPolicy
from utils.rl_glue import RlGlueCompatWrapper
from utils.errors import buildRMSPBE

from environments.RandomWalk import RandomWalk, Tabular, Dependent, Inverted
from agents.TD import TD

# --------------------------------
# Set up parameters for experiment
# --------------------------------

RUNS = 10
LEARNERS = [TD]

PROBLEMS = [
    # 5-state random walk environment with tabular features
    {
        'env': RandomWalk,
        'representation': Tabular,
        # go LEFT 40% of the time
        'target': actionArrayToPolicy([0.4, 0.6]),
        # take each action equally
        'behavior': actionArrayToPolicy([0.5, 0.5]),
        'gamma': 1.0,
        'steps': 3000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.01,
        }
    },
    # 5-state random walk environment with dependent features
    {
        'env': RandomWalk,
        'representation': Dependent,
        # go LEFT 40% of the time
        'target': actionArrayToPolicy([0.4, 0.6]),
        # take each action equally
        'behavior': actionArrayToPolicy([0.5, 0.5]),
        'gamma': 1.0,
        'steps': 3000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.01,
        }
    },
    # 5-state random walk environment with inverted features
    {
        'env': RandomWalk,
        'representation': Inverted,
        # go LEFT 40% of the time
        'target': actionArrayToPolicy([0.4, 0.6]),
        # take each action equally
        'behavior': actionArrayToPolicy([0.5, 0.5]),
        'gamma': 1.0,
        'steps': 3000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.01,
        }
    },
]

COLORS = {
    'TD': 'blue',
}

# -----------------------------------
# Collect the data for the experiment
# -----------------------------------

# a convenience object to store data collected during runs
collector = Collector()

for run in range(RUNS):
    for problem in PROBLEMS:
        for Learner in LEARNERS:
            # for reproducibility, set the random seed for each run
            # also reset the seed for each learner, so we guarantee each sees the same data
            np.random.seed(run)

            # build a new instance of the environment each time
            # just to be sure we don't bleed one learner into the next
            Env = problem['env']
            env = Env()

            target = problem['target']
            behavior = problem['behavior']

            Rep = problem['representation']
            rep = Rep()

            # build the X, P, R, and D matrices for computing RMSPBE
            X, P, R, D = env.getXPRD(target, rep)
            RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

            # build a new instance of the learning algorithm
            learner = Learner(rep.features(), {
                'gamma': problem['gamma'],
                'alpha': problem['stepsizes'][Learner.__name__]
            })

            # build an "agent" which selects actions according to the behavior
            # and tries to estimate according to the target policy
            agent = RlGlueCompatWrapper(learner, behavior, target, rep.encode)

            # build the experiment runner
            glue = RlGlue(agent, env)

            # start the episode
            glue.start()
            for step in range(problem['steps']):
                _, _, _, terminal = glue.step()

                # when we hit a terminal state, start a new episode
                if terminal:
                    glue.start()

                # evaluate the RMPSBE
                w = learner.getWeights()
                rmspbe = RMSPBE(w)

                collector.collect(f'{Env.__name__}-{Rep.__name__}-{Learner.__name__}', rmspbe)

            # tell the data collector we're done collecting data for this env/learner/rep combination
            collector.reset()

# ---------------------
# Plotting the bar plot
# ---------------------
import matplotlib.pyplot as plt

ax = plt.gca()
f = plt.gcf()

# how far from the left side of the plot to put the bar
offset = -3
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    offset += 3
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__
        env = problem['env'].__name__
        rep = problem['representation'].__name__

        x = i * len(LEARNERS) + j + offset

        mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')
        auc = mean_curve.mean()
        auc_stderr = stderr_curve.mean()

        ax.bar(x, auc, yerr=auc_stderr, color=COLORS[learner])

plt.show()
