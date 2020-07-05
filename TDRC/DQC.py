from TDRC.DQRC import DQRC
from TDRC.utils import merge

class DQC(DQRC):
    def __init__(self, features, actions, policy_net, target_net, optimizer, params, device=None):
        super().__init__(features, actions, policy_net, target_net, optimizer, merge(params, {'beta': 0}, device))
