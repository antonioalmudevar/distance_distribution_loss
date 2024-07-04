from typing import Optional

import faiss
import numpy as np


class KNN():

    def __init__(self, norm_features: np.ndarray, **kwargs):
        
        self.n_train_features = norm_features.shape[0]
        self.index = faiss.IndexFlatL2(norm_features.shape[1])
        self.index.add(norm_features) # type: ignore


    def get_scores(
            self, 
            norm_features: np.ndarray, 
            K: Optional[int]=None, 
            **kwargs
        ) -> np.ndarray:
        K = int(0.1 * self.n_train_features) if K is None else K
        D, _ = self.index.search(norm_features, K) # type: ignore
        kth_dist = -D[:, -1]
        return kth_dist