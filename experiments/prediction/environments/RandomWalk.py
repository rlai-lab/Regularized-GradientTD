import numpy as np
from RlGlue import BaseEnvironment

# Constants
LEFT = 0
RIGHT = 1

class RandomWalk(BaseEnvironment):
    def __init__(self, size=5):
        self.states = size
        self.state = size // 2

    def start(self):
        self.state = self.states // 2
        return self.state

    def step(self, a):
        if a == LEFT:
            self.state = max(self.state - 1, -1)

        elif a == RIGHT:
            self.state = min(self.state + 1, self.states)

        reward = 0
        terminal = False

        if self.state == -1:
            reward = -1
            terminal = True

        elif self.state == self.states:
            reward = 1
            terminal = True

        return (reward, self.state, terminal)

    def getXPRD(self, target, rep):
        N = self.states
        # build the state * feature matrix
        # add an extra state at the end to encode the "terminal" state
        X = np.array([
            rep.encode(i) for i in range(N + 1)
        ])

        # build a transition dynamics matrix
        # following policy "target"
        P = np.zeros((N + 1, N + 1))
        pl, pr = target.probs(0)
        P[0, 1] = pr
        P[0, N] = pl
        P[N - 1, N - 2] = pl
        P[N - 1, N] = pr
        for i in range(1, N - 1):
            P[i, i - 1] = pl
            P[i, i + 1] = pr

        # build the average reward vector
        R = np.zeros(N + 1)
        R[0] = pl * -1
        R[N-1] = pr

        # TODO: use np.linalg.matrix_power to compute this for chains of arbitrary length.
        # right now only handles 5-state chains (which is fine for this project)
        D = np.diag([0.111111, 0.222222, 0.333333, 0.222222, 0.111111, 0])

        return X, P, R, D

# -------------------------------------------
# Build feature reps from Sutton et al., 2009
# -------------------------------------------

# Generates a representation like:
# [[ 0, 1, 1, 1, 1 ],
#  [ 1, 0, 1, 1, 1 ],
#  [ 1, 1, 0, 1, 1 ],
#  [ 1, 1, 1, 0, 1 ],
#  [ 1, 1, 1, 1, 0 ]]
# then normalizes the feature vectors for each state
class InvertedRep:
    def __init__(self, N = 5):
        m = np.ones((N, N)) - np.eye(N)

        self.map = np.zeros((N + 1, N))
        self.map[:N] = (m.T / np.linalg.norm(m, axis = 1)).T

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

# Generates a representation like:
# [[ 1, 0, 0, 0, 0 ],
#  [ 0, 1, 0, 0, 0 ],
#  [ 0, 0, 1, 0, 0 ],
#  [ 0, 0, 0, 1, 0 ],
#  [ 0, 0, 0, 0, 1 ]]
class TabularRep:
    def __init__(self, N = 5):
        m = np.eye(N)

        self.map = np.zeros((N + 1, N))
        self.map[:N] = m

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

# Generates a representation like:
# [[ 1, 0, 0 ],
#  [ 1, 1, 0 ],
#  [ 0, 1, 0 ],
#  [ 0, 1, 1 ],
#  [ 0, 0, 1 ]]
# then normalizes the feature vectors for each state
class DependentRep:
    def __init__(self, N = 5):
        nfeats = int(N // 2 + 1)
        self.map = np.zeros((N + 1, nfeats))

        idx = 0
        for i in range(nfeats):
            self.map[idx, 0:i + 1] = 1
            idx += 1

        for i in range(nfeats-1,0,-1):
            self.map[idx, -i:] = 1
            idx += 1

        self.map[:N] = (self.map[:N].T / np.linalg.norm(self.map[:N], axis = 1)).T

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]
