import numpy as np
from RlGlue import BaseEnvironment

BACK = 0
STAY = 1
FORWARD = 2

class MountainCar(BaseEnvironment):
    def __init__(self):
        self.position = -0.6 + np.random.random() * 0.2
        self.velocity = 0.0

        self.features = 2
        self.num_actions = 3

    def start(self):
        self.position = -0.6 + np.random.random() * 0.2
        self.velocity = 0.0

        return (self.position, self.velocity)

    # give all actions for a given state
    def actions(self, s):
        return [BACK, STAY, FORWARD]

    # give the rewards associated with a given state, action, next state tuple
    def rewards(self, s, a, sp):
        return -1

    # get the next state and termination status
    def next_state(self, s, a):
        a = a - 1
        p, v = s

        v += 0.001 * a - 0.0025 * np.cos(3 * p)

        if v < -0.07:
            v = -0.07
        elif v >= 0.07:
            v = 0.06999999

        p += v

        if p >= 0.5:
            return (p, v), True

        if p < -1.2:
            return (-1.2, 0.0), False

        return (p, v), False

    def step(self, a):
        s = (self.position, self.velocity)
        sp, t = self.next_state(s, a)
        self.position = sp[0]
        self.velocity = sp[1]

        r = self.rewards(s, a, sp)

        return (r, sp, t)
