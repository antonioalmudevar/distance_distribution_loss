from typing import Any

import torch
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

__all__ = ["RESNETS", "get_resnet"]


class ResNetEncoder(ResNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = 512

    def _forward_impl(self, x: Tensor) -> Tensor:
        if len(x.shape)==3:
            x = x.unsqueeze(1)
        if x.shape[1]==1:
            x = x.repeat(1,3,1,1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    
#==========Original ResNets===============
def resnet18(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)


RESNETS = {
    "RESNET18": resnet18,
    "RESNET34": resnet34,
    "RESNET50": resnet50,
}


def get_resnet(arch: str, **kwargs: Any) -> ResNetEncoder:
    return RESNETS[arch.upper()](**kwargs)