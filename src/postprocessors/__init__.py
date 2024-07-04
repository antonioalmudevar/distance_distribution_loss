from typing import Union, Dict, Any

from .msp import MSP
from .knn import KNN
from .maha import Mahalanobis
from .maha_class import MahalanobisClass

def get_postprocessor(
        post_name: str, 
        results: Dict[str, Any]
    ) -> Union[MSP, KNN, Mahalanobis, MahalanobisClass]:
    if post_name=='MSP':
        return MSP()
    elif post_name=='KNN':
        return KNN(**results)
    elif post_name=='Mahalanobis':
        return Mahalanobis(**results)
    elif post_name=='Maha Class':
        return MahalanobisClass(**results)
    else:
        raise ValueError

