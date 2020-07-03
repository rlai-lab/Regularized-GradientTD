import numpy as np

# much much faster than np.random.choice
def choice(arr, size=1):
    idxs = np.random.permutation(len(arr))
    return [arr[i] for i in idxs[:size]]

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(args)
        else:
            self.buffer[self.location] = args

        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return choice(self.buffer, batch_size), []

    # match api with prioritized ER buffer
    def update_priorities(self, idxes, priorities):
        pass
