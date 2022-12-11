import objdetect as od
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


def create_resnet50(nclasses):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = Model(
        resnet_backbone,
        {
            "hasobjs": od.models.HeadScores(2048),
            "classes": od.models.HeadClasses(2048, nclasses),
        },
    )
    return model


def create_split_resnet50(nclasses_kitti, nclasses_gtsdb):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = Model(
        resnet_backbone,
        {
            "scores_kitti": od.models.HeadScores(2048),
            "classes_kitti": od.models.HeadClasses(2048, nclasses_kitti),
            "scores_gtsdb": od.models.HeadScores(2048),
            "classes_gtsdb": od.models.HeadClasses(2048, nclasses_gtsdb),
        },
    )
    return model


def create_resnet34(nclasses):
    resnet = torchvision.models.resnet34(pretrained=True)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-2]))
    model = od.models.Model(
        resnet_backbone,
        {
            "scores": od.models.HeadScores(512),
            "classes": od.models.HeadClasses(512, nclasses),
        },
    )
    return model


def create_mobilenetv3_small(nclasses):
    mobilenet_v3 = torchvision.models.mobilenet_v3_small(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores": od.models.HeadScores(576),
            "classes": od.models.HeadClasses(576, nclasses),
        },
    )
    return model


def create_split_mobilenetv3_small(nclasses_kitti, nclasses_gtsdb):
    mobilenet_v3 = torchvision.models.mobilenet_v3_small(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores_kitti": od.models.HeadScores(576),
            "classes_kitti": od.models.HeadClasses(576, nclasses_kitti),
            "scores_gtsdb": od.models.HeadScores(576),
            "classes_gtsdb": od.models.HeadClasses(576, nclasses_gtsdb),
        },
    )
    return model


def create_mobilenetv3_large(nclasses):
    mobilenet_v3 = torchvision.models.mobilenet_v3_large(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores": od.models.HeadScores(960),
            "classes": od.models.HeadClasses(960, nclasses),
        },
    )
    return model


def create_split_mobilenetv3_large(nclasses_kitti, nclasses_gtsdb):
    mobilenet_v3 = torchvision.models.mobilenet_v3_large(pretrained=True).children()
    model = Model(
        torch.nn.Sequential(*(list(mobilenet_v3)[:-2])),
        {
            "scores_kitti": od.models.HeadScores(960),
            "classes_kitti": od.models.HeadClasses(960, nclasses_kitti),
            "scores_gtsdb": od.models.HeadScores(960),
            "classes_gtsdb": od.models.HeadClasses(960, nclasses_gtsdb),
        },
    )
    return model


def create_default(nclasses):
    model = od.models.Model(
        od.models.Backbone(5),
        {
            "scores": od.models.HeadScores(512),
            "classes": od.models.HeadClasses(512, nclasses),
        },
    )
    return model
