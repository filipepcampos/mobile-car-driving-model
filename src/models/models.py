import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = nn.ModuleDict(head)

    def forward(self, data):
        backbone_output = self.backbone(data)
        return {name: h(backbone_output) for name, h in self.head.items()}


class HeadScores(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.conv = nn.Conv2d(n_inputs, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        if not self.training:
            return torch.sigmoid(x)
        return x


class HeadClasses(nn.Module):
    def __init__(self, n_inputs, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(n_inputs, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        if not self.training:
            return nn.functional.softmax(x, 1)
        return x


def create_resnet50(nclasses):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = Model(
        resnet_backbone,
        {
            "hasobjs": HeadScores(2048),
            "classes": HeadClasses(2048, nclasses),
        },
    )
    return model


def create_split_resnet50(nclasses_kitti, nclasses_gtsdb):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = Model(
        resnet_backbone,
        {
            "scores_kitti": HeadScores(2048),
            "classes_kitti": HeadClasses(2048, nclasses_kitti),
            "scores_gtsdb": HeadScores(2048),
            "classes_gtsdb": HeadClasses(2048, nclasses_gtsdb),
        },
    )
    return model


def create_resnet34(nclasses):
    resnet = torchvision.models.resnet34(pretrained=True)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = Model(
        resnet_backbone,
        {
            "scores": HeadScores(512),
            "classes": HeadClasses(512, nclasses),
        },
    )
    return model


def create_mobilenetv3_small(nclasses):
    mobilenet_v3 = torchvision.models.mobilenet_v3_small(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores": HeadScores(576),
            "classes": HeadClasses(576, nclasses),
        },
    )
    return model


def create_split_mobilenetv3_small(nclasses_kitti, nclasses_gtsdb):
    mobilenet_v3 = torchvision.models.mobilenet_v3_small(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores_kitti": HeadScores(576),
            "classes_kitti": HeadClasses(576, nclasses_kitti),
            "scores_gtsdb": HeadScores(576),
            "classes_gtsdb": HeadClasses(576, nclasses_gtsdb),
        },
    )
    return model


def create_mobilenetv3_large(nclasses):
    mobilenet_v3 = torchvision.models.mobilenet_v3_large(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores": HeadScores(960),
            "classes": HeadClasses(960, nclasses),
        },
    )
    return model


def create_split_mobilenetv3_large(nclasses_kitti, nclasses_gtsdb):
    mobilenet_v3 = torchvision.models.mobilenet_v3_large(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores_kitti": HeadScores(960),
            "classes_kitti": HeadClasses(960, nclasses_kitti),
            "scores_gtsdb": HeadScores(960),
            "classes_gtsdb": HeadClasses(960, nclasses_gtsdb),
        },
    )
    return model
