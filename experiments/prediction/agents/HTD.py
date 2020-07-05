import numpy as np

class HTD:
    def __init__(self, features, params):
        self.features = features
        self.params = params

        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.eta = params.get('eta', 1)

        self.w = np.zeros(features)
        self.h = np.zeros(features)

    def update(self, x, a, r, xp, rho):
        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + self.gamma * vp - v
        delta_hat = self.h.dot(x)

        dh = (rho * delta * x - delta_hat * (x - self.gamma * xp))
        dw = rho * delta * x + (x - self.gamma * xp) * (rho - 1) * delta_hat

        self.w = self.w + self.alpha * dw
        self.h = self.h + self.eta * self.alpha * dh

    def getWeights(self):
        return self.w
