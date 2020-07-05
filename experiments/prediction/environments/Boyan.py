import numpy as np
from RlGlue import BaseEnvironment

# Constants
RIGHT = 0
SKIP = 1

class Boyan(BaseEnvironment):
    def __init__(self):
        self.states = 12
        self.state = 0

    def start(self):
        self.state = 0
        return self.state

    def step(self, a):
        reward = -3
        terminal = False

        if a == SKIP and self.state > 9:
            print("Double right action is not available in state 10 or state 11... Exiting now.")
            exit()

        if a == RIGHT:
            self.state = self.state + 1
        elif a == SKIP:
            self.state = self.state + 2

        if (self.state == 12):
            terminal = True
            reward = -2

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
        for i in range(11):
            P[i, i+1] = .5
            P[i, i+2] = .5

        P[10, 11] = 1
        P[11, 12] = 1

        # build the average reward vector
        R = np.array([-3] * 10 + [-2, -2, 0])

        D = np.diag([0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308])

        return X, P, R, D


class BoyanRep:
    def __init__(self):
        self.map = np.array([
            [1,    0,    0,    0   ],
            [0.75, 0.25, 0,    0   ],
            [0.5,  0.5,  0,    0   ],
            [0.25, 0.75, 0,    0   ],
            [0,    1,    0,    0   ],
            [0,    0.75, 0.25, 0   ],
            [0,    0.5,  0.5,  0   ],
            [0,    0.25, 0.75, 0   ],
            [0,    0,    1,    0   ],
            [0,    0,    0.75, 0.25],
            [0,    0,    0.5,  0.5 ],
            [0,    0,    0.25, 0.75],
            [0,    0,    0,    1   ],
        ])

    def encode(self, s):
        return self.map[s]

    def features(self):
        return 4
