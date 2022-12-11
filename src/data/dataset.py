import os

import torch
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision.io import read_image


class GTSDB(Dataset):
    labels = ["stop", "giveway", "prohibited", "noovertaking", "allowovertaking"]

    label_id_type = {
        14: "stop",
        13: "giveway",
        17: "prohibited",
        9: "noovertaking",
        41: "allowovertaking",
    }

    gt_map = {}

    def __init__(self, root, img_transform, dict_transform, exclude_labels=None):
        exclude_labels = {} if exclude_labels is None else exclude_labels
        self.labels = [label for label in self.labels if label not in exclude_labels]
        self.root = os.path.join(root, "GTSDB", "TrainIJCNN2013")
        self.files = os.listdir(self.root)
        self.files = [f for f in self.files if f.endswith(".ppm")]
        self.img_transform = img_transform
        self.dict_transform = dict_transform
        self.setup_gt_map()
        self.files = [
            f for f in self.files if f in self.gt_map
        ]  # Filter out images without labels

    def setup_gt_map(self):
        lines = [
            line.split() for line in open(os.path.join(self.root, "gt.txt")).readlines()
        ]
        for line in lines:
            split_line = line[0].split(";")
            key = split_line[0]
            label = int(split_line[-1])
            bboxes = [
                float(b) / (1360 if i % 2 == 0 else 800)
                for i, b in enumerate(split_line[1:-1])
            ]  # TODO: Generalize resolution
            if label in self.label_id_type:
                label = self.label_id_type[label]
                if key in self.gt_map:
                    self.gt_map[key].append((bboxes, label))
                else:
                    self.gt_map[key] = [(bboxes, label)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        image = torch.tensor(imread(os.path.join(self.root, filename))) / 255
        image = image.permute(2, 0, 1)
        if filename in self.gt_map:
            bboxes = torch.tensor(
                [i for i, _ in self.gt_map[filename]],
                dtype=torch.float32,
            )
            classes = torch.tensor(
                [self.labels.index(i) for _, i in self.gt_map[filename]],
                dtype=torch.int64,
            )
        else:
            bboxes, classes = torch.empty(), torch.empty()

        if self.img_transform:
            image = self.img_transform(image)
        datum = {"image": image, "bboxes": bboxes, "classes_gtsdb": classes}
        if self.dict_transform:
            datum = self.dict_transform(**datum)
        return datum


class KITTIDetection(Dataset):
    """The [KITTI](http://www.cvlibs.net/datasets/kitti/) self-driving
    dataset."""

    original_labels = [
        "Car",
        "Cyclist",
        "Pedestrian",
        "Person_sitting",
        "Tram",
        "Truck",
        "Van",
        "Misc",
        "DontCare",
    ]
    labels = ["Vehicle", "Pedestrian"]

    def __init__(
        self,
        root,
        fold,
        img_transform,
        dict_transform,
        exclude_labels=None,
    ):
        assert fold in ("train",)
        exclude_labels = {"Misc", "DontCare"} if exclude_labels is None else {}
        self.img_transform = img_transform
        self.dict_transform = dict_transform
        self.original_labels = [
            label for label in self.original_labels if label not in exclude_labels
        ]
        self.root = os.path.join(root, "kitti", "object", "training")
        self.files = os.listdir(os.path.join(self.root, "image_2"))

    def __len__(self):
        return len(self.files)

    def convert_label(self, label):
        conversion_dict = {
            "Car": "Vehicle",
            "Cyclist": "Vehicle",
            "Pedestrian": "Pedestrian",
            "Person_sitting": "Pedestrian",
            "Tram": "Vehicle",
            "Truck": "Vehicle",
            "Van": "Vehicle",
        }
        return conversion_dict[label]

    def __getitem__(self, i):
        filename = self.files[i]
        image = read_image(os.path.join(self.root, "image_2", filename)) / 255
        lines = [
            line.split()
            for line in open(
                os.path.join(self.root, "label_2", filename[:-3] + "txt"),
            ).readlines()
        ]
        lines = [line for line in lines if line[0] in self.original_labels]
        bboxes = torch.tensor(
            [
                (
                    float(line[4]) / image.shape[2],
                    float(line[5]) / image.shape[1],
                    float(line[6]) / image.shape[2],
                    float(line[7]) / image.shape[1],
                )
                for line in lines
            ],
            dtype=torch.float32,
        )
        classes = torch.tensor(
            [self.labels.index(self.convert_label(line[0])) for line in lines],
            dtype=torch.int64,
        )

        if self.img_transform:
            image = self.img_transform(image)
        datum = {"image": image, "bboxes": bboxes, "classes_kitti": classes}
        if self.dict_transform:
            datum = self.dict_transform(**datum)
        return datum


# Concat two datasets without any transformation
class ConcatDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, i):
        dataset1_size = len(self.dataset1)
        if i < dataset1_size:
            return self.dataset1[i]
        return self.dataset2[i - dataset1_size]
