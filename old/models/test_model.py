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

import data

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
        ],
        bbox_params=albumentations.BboxParams(
            format="albumentations",
            label_fields=["classes_gtsdb"],
        ),
    ),
)


def precision(tp, fp, tn, fn):
    return tp / (tp + fp)


def recall(tp, fp, tn, fn):
    return tp / (tp + fn)


def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def f1(tp, fp, tn, fn):
    prec = precision(tp, fp, tn, fn)
    rec = recall(tp, fp, tn, fn)
    return 2 * (prec * rec) / (prec + rec)


def log_metrics(tp, fp, tn, fn):
    logger.info(f" accuracy: {accuracy(tp, fp, tn, fn)}")
    logger.info(f" precision: {precision(tp, fp, tn, fn)}")
    logger.info(f" recall: {recall(tp, fp, tn, fn)}")
    logger.info(f" f1: {f1(tp, fp, tn, fn)}")


def log_confusion_matrix(tp, fp, tn, fn):
    logger.info("                   | Actual   | Actual   |")
    logger.info("                   | Positive | Negative |")
    logger.info(f"Predicted Positive | {tp : >8} | {fp : >8} |")
    logger.info(f"Predicted Negative | {fn : >8} | {tn : >8} |")


def test(tr, model, output_suffix=""):
    model.eval()

    # true positive, false positive, true negative, false negative
    tp, fp, tn, fn = 0, 0, 0, 0

    threshold = 0.5
    for data_dict in tr:
        datum = data_dict["image"].to(DEVICE)
        preds = model(datum)
        data_cuda = {name: data_dict[name].to(DEVICE) for name in data_dict.keys()}

        true_scores = data_cuda[f"scores{output_suffix}"]
        pred_scores = preds[f"scores{output_suffix}"] > threshold
        tp += ((pred_scores == 1) & (true_scores == 1)).sum()
        fp += ((pred_scores == 1) & (true_scores == 0)).sum()
        tn += ((pred_scores == 0) & (true_scores == 0)).sum()
        fn += ((pred_scores == 0) & (true_scores == 1)).sum()
    return tp, fp, tn, fn


# Creating dataset
kitti_dataset = data.KITTIDetection("/data", "train", None, kitti_transforms)
gtsdb_dataset = data.GTSDB("/data", None, gtsdb_transforms)

# Model creation
model = torch.load("../models/split_mobilenetv3_large.pth", map_location=DEVICE)
model = model.to(DEVICE)

logger.info("-- GTSDB --")
tr = DataLoader(gtsdb_dataset, 32, True, num_workers=6, pin_memory=True)
gtsdb_tp, gtsdb_fp, gtsdb_tn, gtsdb_fn = test(tr, model, "_gtsdb")
log_confusion_matrix(gtsdb_tp, gtsdb_fp, gtsdb_tn, gtsdb_fn)
log_metrics(gtsdb_tp, gtsdb_fp, gtsdb_tn, gtsdb_fn)

logger.info("-- KITTI --")
tr = DataLoader(kitti_dataset, 32, True, num_workers=6, pin_memory=True)
kitti_tp, kitti_fp, kitti_tn, kitti_fn = test(tr, model, "_kitti")
log_confusion_matrix(kitti_tp, kitti_fp, kitti_tn, kitti_fn)
log_metrics(kitti_tp, kitti_fp, kitti_tn, kitti_fn)

logger.info("-- Combined --")
tp, fp, tn, fn = (
    gtsdb_tp + kitti_tp,
    gtsdb_fp + kitti_fp,
    gtsdb_tn + kitti_tn,
    gtsdb_fn + kitti_fn,
)
log_confusion_matrix(tp, fp, tn, fn)
log_metrics(tp, fp, tn, fn)
