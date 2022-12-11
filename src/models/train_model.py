from time import time

import albumentations
import objdetect as od
import torch
from grid import (
    new_id,
    permute,
    set_classes,
    slice_across_bbox,
    to_numpy_array,
    to_tensor,
)
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss

import data
from models import create_split_mobilenetv3_small

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 500
IMG_SIZE = (256, 1024)
GRID_SIZE = (8, 32)
N_CLASSES_KITTI = 2
N_CLASSES_GTSDB = 5


def create_transforms_kitti(transform):
    grid_transform = od.grid.Transform(
        GRID_SIZE,
        None,
        slice_across_bbox(),
        {
            "scores_kitti": od.grid.NewScore(),
            "classes_kitti": od.grid.NewClasses(),
            "scores_gtsdb": od.grid.NewScore(),
            "classes_gtsdb": od.grid.NewClasses(),
            "id": new_id(1),
        },
        {
            "scores_kitti": od.grid.SetScore(),
            "classes_kitti": set_classes("classes_kitti"),
        },
    )

    return od.aug.Compose(
        [
            permute([1, 2, 0]),  # Albumentation wants [H,W,C]
            to_numpy_array(),  # Albumentation uses np arrays and not torch tensors
            transform,
            to_tensor(),
            permute([2, 0, 1]),
            grid_transform,
            od.grid.RemoveKeys(["bboxes"]),
        ],
    )


def create_transforms_gtsdb(transform):
    grid_transform = od.grid.Transform(
        GRID_SIZE,
        None,
        slice_across_bbox(),
        {
            "scores_kitti": od.grid.NewScore(),
            "classes_kitti": od.grid.NewClasses(),
            "scores_gtsdb": od.grid.NewScore(),
            "classes_gtsdb": od.grid.NewClasses(),
            "id": new_id(0),
        },
        {
            "scores_gtsdb": od.grid.SetScore(),
            "classes_gtsdb": set_classes("classes_gtsdb"),
        },
    )

    return od.aug.Compose(
        [
            permute([1, 2, 0]),  # Albumentation wants [H,W,C]
            to_numpy_array(),  # Albumentation uses np arrays and not torch tensors
            transform,
            to_tensor(),
            permute([2, 0, 1]),
            grid_transform,
            od.grid.RemoveKeys(["bboxes"]),
        ],
    )


kitti_transforms = create_transforms_kitti(
    albumentations.Compose(
        [
            albumentations.RandomCrop(width=IMG_SIZE[1], height=IMG_SIZE[0]),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(),
        ],
        bbox_params=albumentations.BboxParams(
            format="albumentations",
            label_fields=["classes_kitti"],
        ),
    ),
)

# gtsb uses 1360x800 images while kitti uses 1242x375 (1242/375 = 3.312)
gtsdb_transforms = create_transforms_gtsdb(
    albumentations.Compose(
        [
            albumentations.RandomCrop(
                width=1360,
                height=int(1360 / 3.312),
            ),  # Match kitti aspect ratio without distortion
            albumentations.Resize(width=IMG_SIZE[1], height=IMG_SIZE[0]),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(),
        ],
        bbox_params=albumentations.BboxParams(
            format="albumentations",
            label_fields=["classes_gtsdb"],
        ),
    ),
)

# Creating dataset
kitti_dataset = data.KITTIDetection("/data", "train", None, kitti_transforms)
gtsdb_dataset = data.GTSDB("/data", None, gtsdb_transforms)

tr = data.ConcatDataset(kitti_dataset, gtsdb_dataset)
tr = DataLoader(tr, 32, True, num_workers=6, pin_memory=True)


# Model creation
model = create_split_mobilenetv3_small(N_CLASSES_KITTI, N_CLASSES_GTSDB)
model = model.to(DEVICE)

weight_loss_fns = {
    "scores_kitti": lambda data: data["id"],
    "classes_kitti": lambda data: data["scores_kitti"],
    "scores_gtsdb": lambda data: 1 - data["id"],
    "classes_gtsdb": lambda data: data["scores_gtsdb"],
}
loss_fns = {
    "scores_kitti": sigmoid_focal_loss,
    "classes_kitti": torch.nn.CrossEntropyLoss(reduction="none"),
    "scores_gtsdb": sigmoid_focal_loss,
    "classes_gtsdb": torch.nn.CrossEntropyLoss(reduction="none"),
}
opt = torch.optim.Adam(model.parameters())


def train(tr, model, opt, weight_loss_fns, loss_fns, epochs, stop_condition=None):
    """Trains the model.

    `weight_loss_fns` and `loss_fns` are dictionaries, specifying
    whether the loss should be applied to that grid location and what
    loss to apply.
    """
    device = next(model.parameters()).device

    # sanity-check: all losses must have reduction='none'
    data = next(iter(tr))

    preds = model(data["image"].to(device))
    for name, func in loss_fns.items():
        loss_value = func(preds[name], data[name].to(device))
        assert len(loss_value.shape) > 0, f"Loss {name} must have reduction='none'"

    model.train()
    for epoch in range(epochs):
        logger.info(f"* Epoch {epoch+1} / {epochs}")
        tic = time()
        avg_loss = 0
        avg_losses = {name: 0 for name in loss_fns}
        for data in tr:
            datum = data["image"].to(device)
            preds = model(datum)
            data_cuda = {name: data[name].to(device) for name in loss_fns}
            data_cuda["id"] = data["id"].to(
                device,
            )  # Add id even though it doesn't have loss

            loss = 0
            for name, func in loss_fns.items():
                if name in weight_loss_fns:
                    weight = weight_loss_fns[name](data_cuda)
                    true_value = data_cuda[name]
                    pred_value = preds[name]
                    loss_value = (weight * func(pred_value, true_value)).mean()
                    loss += loss_value
                    avg_losses[name] += float(loss_value) / len(tr)

            opt.zero_grad()
            loss.backward()
            opt.step()
            avg_loss += float(loss) / len(tr)
        toc = time()
        logger.info(
            f"- {toc-tic:.1f}s - Avg loss: {avg_loss} - "
            + " - ".join(f"{name} loss: {avg}" for name, avg in avg_losses.items()),
        )
        if stop_condition and stop_condition(avg_loss):
            logger.info("Stopping due to criteria")
            break


logger.info(f"Training split {EPOCHS} epochs.")
train(tr, model, opt, weight_loss_fns, loss_fns, EPOCHS, od.loop.StopPatience(10))
torch.save(model, "../models/split.pth")
