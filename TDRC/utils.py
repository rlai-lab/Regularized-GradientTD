import torch
from collections import namedtuple

batch = namedtuple(
    'batch',
    'states, nterm_sp, actions, rewards, gamma, term, nterm, size'
)

def getBatchColumns(samples, device=None):
    s, a, r, sp, gamma = list(zip(*samples))
    states = torch.cat(s)
    actions = torch.tensor(a, device=device).unsqueeze(1)
    rewards = torch.cat(r)
    gamma = torch.tensor(gamma, device=device)

    is_terminal = gamma == 0

    sps = [x for x in sp if x is not None]
    if len(sps) > 0:
        non_final_next_states = torch.cat(sps)
    else:
        non_final_next_states = torch.zeros((0, states.shape[1]))

    non_term = torch.logical_not(is_terminal).to(device)

    return batch(states, non_final_next_states, actions, rewards, gamma, is_terminal, non_term, len(samples))

def merge(d1, d2):
    ret = d2.copy()
    for key in d1:
        ret[key] = d2.get(key, d1[key])

    return ret
