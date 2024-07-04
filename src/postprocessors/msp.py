import numpy as np

class MSP():

    def __init__(self, **kwargs):
        pass


    def get_scores(
            self, 
            preds: np.ndarray, 
            **kwargs
        ) -> np.ndarray:
        return preds.max(axis=1)