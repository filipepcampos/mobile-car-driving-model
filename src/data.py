import os
from typing import List, TypedDict

import albumentations
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class KITTIDatum(TypedDict):
    image: torch.Tensor
    bboxes: List[torch.Tensor]
    classes: List[torch.Tensor]


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
        root: str,
        fold: str,
        exclude_labels: List[str] = None,
    ):
        assert fold in ("train",)
        exclude_labels = {"Misc", "DontCare"} if exclude_labels is None else {}
        self.original_labels = [
            label for label in self.original_labels if label not in exclude_labels
        ]
        self.root = os.path.join(root, "training")
        self.files = os.listdir(os.path.join(self.root, "image_2"))

    def __len__(self):
        return len(self.files)

    def convert_label(self, label: str) -> str:
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

    def __getitem__(self, i: int) -> KITTIDatum:
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

        return {"image": image, "bboxes": bboxes, "classes": classes}


class AugmentedKITTIDetection(KITTIDetection):
    def __init__(
        self,
        root: str,
        fold: str,
        exclude_labels: List[str] = None,
    ):
        super().__init__(root, fold, exclude_labels)
        img_width, img_height = 1024, 256
        self.transform = albumentations.Compose(
            [
                albumentations.RandomCrop(width=img_width, height=img_height),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(),
            ],
            bbox_params=albumentations.BboxParams(
                format="albumentations",
                label_fields=["classes"],
            ),
        )

    def __getitem__(self, i: int) -> KITTIDatum:
        item = super().__getitem__(i)
        image = np.array(
            item["image"].permute(1, 2, 0),
        )  # Albumentation uses an numpy array with [H,W,C]
        transformed_data = self.transform(
            image=image,
            bboxes=item["bboxes"],
            classes=item["classes"],
        )
        transformed_data["image"] = torch.tensor(transformed_data["image"]).permute(
            2,
            0,
            1,
        )
        return transformed_data
