from agents.TDRC import TDRC
from utils.dict import merge

class TDC(TDRC):
    def __init__(self, features, params):
        # TDC is just an instance of TDRC where beta = 0
        super().__init__(features, merge(params, { 'beta': 0 }))
