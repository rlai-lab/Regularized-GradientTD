import numpy as np
from RlGlue import BaseEnvironment

# Constants
DASH = 0
SOLID = 1

class Baird(BaseEnvironment):
    def __init__(self):
        self.states = 7
        self.state = 0

    def start(self):
        self.state = 6
        return self.state

    def step(self, a):
        if a == SOLID:
            self.state = 6
        elif a == DASH:
            self.state = np.random.randint(6)

        return (0, self.state, False)

    def getXPRD(self, target, rep):
        N = self.states
        # build the state * feature matrix
        # add an extra state at the end to encode the "terminal" state
        X = np.array([
            rep.encode(i) for i in range(N)
        ])

        # build a transition dynamics matrix
        # following policy "target"
        P = np.zeros((N, N))
        P[:, 6] = 1

        # build the average reward vector
        R = np.zeros(N)

        D = np.diag(np.ones(N) * (1/N))

        return X, P, R, D

class BairdRep:
    def __init__(self):
        self.map = np.array([
            [1, 2, 0, 0, 0, 0, 0, 0],
            [1, 0, 2, 0, 0, 0, 0, 0],
            [1, 0, 0, 2, 0, 0, 0, 0],
            [1, 0, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 0, 0, 2, 0, 0],
            [1, 0, 0, 0, 0, 0, 2, 0],
            [2, 0, 0, 0, 0, 0, 0, 1],
        ])

    def encode(self, s):
        return self.map[s]

    def features(self):
        return 8
