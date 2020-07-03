from .random import sampleFromDist

# wraps a function which takes a state and returns a list of probabilities for each action
# helps maintain a consistent API even if policies are generated in different ways
class Policy:
    def __init__(self, probs):
        self.probs = probs

    def selectAction(self, s):
        action_probabilities = self.probs(s)
        return sampleFromDist(action_probabilities)

    def ratio(self, other, s, a):
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def matrixToPolicy(probs):
    return Policy(lambda s: probs[s])

def actionArrayToPolicy(probs):
    return Policy(lambda s: probs)
