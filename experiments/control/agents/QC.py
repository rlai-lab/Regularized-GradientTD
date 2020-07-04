from agents.QRC import QRC
from utils.dict import merge

class QC(QRC):
    def __init__(self, features, actions, params):
        # QC is just QRC with beta = 0
        # so overwrite any params that might be sent in and force beta = 0
        super().__init__(features, actions, merge(params, { 'beta': 0.0 }))
