import numpy as np
from RlGlue import BaseAgent

# ties together a learning algorithm, a behavior policy, a target policy, and a representation
# stores the previous state and representation of that state for 1-step bootstrapping methods (all methods in this project)
# this wrapper allows easy API compatibility with RLGlue, while allowing the learning code to remain small and simple (the complexity due to bookkeeping is pushed into this class)
class RlGlueCompatWrapper(BaseAgent):
    def __init__(self, learner, behavior, target, representation):
        self.learner = learner
        self.behavior = behavior
        self.target = target
        self.representation = representation
        self.s = None
        self.a = None

        # because representations are always deterministic in this project
        # just store the past representation to reduce compute
        self.x = None

    # called on the first step of the episode
    def start(self, s):
        self.s = s
        self.x = self.representation(s)
        self.a = self.behavior.selectAction(s)
        return self.a

    # called on all subsequent steps of the episode except the terminal step
    def step(self, r, s):
        xp = self.representation(s)
        rho = self.target.ratio(self.behavior, self.s, self.a)
        self.learner.update(self.x, self.a, r, xp, rho)

        self.s = s
        self.a = self.behavior.selectAction(s)
        self.x = xp

        return self.a

    # called on the terminal step of the episode
    def end(self, r):
        rho = self.target.ratio(self.behavior, self.s, self.a)

        # there is no next-state on the terminal state, so
        # encode the "next-state" as a zero vector to avoid accidental bootstrapping
        xp = np.zeros_like(self.x)
        self.learner.update(self.x, self.a, r, xp, rho)

        self.x = xp
