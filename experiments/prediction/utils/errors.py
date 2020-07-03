import numpy as np

def buildRMSPBE(X, P, R, D, gamma):
    A = X.T.dot(D).dot(np.eye(X.shape[0]) - gamma * P).dot(X)
    b = X.T.dot(D).dot(R)
    C = X.T.dot(D).dot(X)

    Cinv = np.linalg.pinv(C)

    def RMSPBE(w):
        v = np.dot(-A, w) + b
        mspbe = v.T.dot(Cinv).dot(v)
        rmspbe = np.sqrt(mspbe)

        return rmspbe

    return RMSPBE
