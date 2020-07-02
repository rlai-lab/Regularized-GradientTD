import numpy as np

# way faster than np.random.choice on small lists
def sampleFromDist(arr):
    r = np.random.random()
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    # we could also recursively call sampleFromDist again
    # but the bias due to this floating point error is so negligibly small, it simply doesn't matter
    return len(arr) - 1

# like np.random.choice
# with replacement
def sample(arr, samples = None):
    indices = np.random.randint(0, len(arr), size=samples)
    return np.array(arr)[indices]
