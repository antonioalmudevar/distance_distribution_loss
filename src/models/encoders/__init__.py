from .ast import PretrainedASTModel
from .beats import get_beats, BEATs
from .resnet import *
from .crnn import CRNN

__all__ = [
    "get_encoder",
    "PretrainedASTModel",
    "BEATs",
    "CRNN",
]


def get_encoder(
        arch: str, 
        input_fdim: int,
        input_tdim: int,
        **kwargs
    ):

    if arch.upper()=="AST":
        return PretrainedASTModel(
            input_fdim=input_fdim, input_tdim=input_tdim, **kwargs
        )
    
    elif arch.upper()=="BEATS":
        return get_beats(**kwargs)
    
    elif arch.upper() in RESNETS:
        return get_resnet(arch, **kwargs)
    
    elif arch.upper()=="CRNN":
        return CRNN(**kwargs)
    
    else:
        raise ValueError