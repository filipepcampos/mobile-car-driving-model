from typing import Dict, List

import torch
import torch.nn as nn
import torchvision
from torch import Tensor


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.ModuleDict(head)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        backbone_output = self.backbone(x)
        return {name: h(backbone_output) for name, h in self.head.items()}


class HeadScores(nn.Module):
    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_inputs, 1, 1)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        x = self.conv(x)
        if not self.training:
            return torch.sigmoid(x)
        return x


class HeadClasses(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_inputs, n_classes, 1)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        x = self.conv(x)
        if not self.training:
            return nn.functional.softmax(x, 1)
        return x


def create_mobilenetv3_large(nclasses):
    mobilenet_v3 = torchvision.models.mobilenet_v3_large(
        weights="MobileNet_V3_Large_Weights.IMAGENET1K_V2",
    ).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores": HeadScores(960),
            "classes": HeadClasses(960, nclasses),
        },
    )
    return model
