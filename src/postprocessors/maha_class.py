import numpy as np


class MahalanobisClass():
    
    def __init__(self, norm_features: np.ndarray, labels: np.ndarray, **kwargs):

        self.n_classes = len(set(labels))
        classwise_idx = {} 
        classwise_mean = np.zeros((
            self.n_classes, norm_features.shape[1]
        ))
        classwise_precision = np.zeros((
            self.n_classes, norm_features.shape[1], norm_features.shape[1]
        ))
        
        for class_id in range(self.n_classes):
            classwise_idx[class_id] = np.where(labels == class_id)[0]
        
        for cls in range(self.n_classes):
            norm_features_cls = norm_features[classwise_idx[cls]]
            classwise_mean[cls] = np.mean(norm_features_cls, axis=0)
            cov = np.cov(norm_features_cls.T) 
            classwise_precision[cls] = np.linalg.pinv(cov)

        self.classwise_mean = classwise_mean
        self.classwise_precision = classwise_precision


    def get_scores(
            self, 
            norm_features: np.ndarray, 
            **kwargs
        ) -> np.ndarray:

        maha_score = []
        for i in range(self.n_classes):
            zero_f = norm_features - self.classwise_mean[i]
            maha_dist = -0.5*np.matmul(np.matmul(zero_f, self.classwise_precision[i]), zero_f.T).diagonal()
            maha_score.append(maha_dist)
                
        maha_score = np.array(maha_score)
        maha_score = np.max(maha_score, axis=0)
        return maha_score