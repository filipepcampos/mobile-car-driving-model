import mlflow
import torch
import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss

import src.models.grid as grid
import src.models.utils as utils
from models import create_mobilenetv3_large
from src.data import dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
IMG_SIZE = (256, 1024)
GRID_SIZE = (8, 32)
N_CLASSES = 2

import numpy as np
from torchvision.utils import draw_bounding_boxes
def slice_across_bbox(height, width, bbox):
    """Choose all grid locations that contain the entirety of the object
    (similar to FCOS)."""

    yy = slice(int(np.floor(bbox[1] * height)), int(np.ceil(bbox[3] * height)))
    xx = slice(int(np.floor(bbox[0] * width)), int(np.ceil(bbox[2] * width)))
    return yy, xx


def slice_across_bboxes(height, width, batch_bboxes):
    return [
        [slice_across_bbox(height, width, bbox) for bbox in bboxes]
        for bboxes in batch_bboxes
    ]


def calculate_loss(images, targets, preds, scores_loss, classes_loss):
    torchy_slices = slice_across_bboxes(
        GRID_SIZE[0],
        GRID_SIZE[1],
        [i.cpu() for i in targets["bboxes"]]
    )
    
    # Convert
    slices = grid.slice_across_bboxes(
        GRID_SIZE[0],
        GRID_SIZE[1],
        [i.cpu() for i in targets["bboxes"]], # TODO: might be unecessary to use .cpu()
    )
    scores = grid.scores(8, 32, slices).to(DEVICE)
    classes = grid.classes(8, 32, slices, targets["classes"]).to(DEVICE)

    # Convert target to CUDA (or device)
    images = images.to(DEVICE)
    targets["classes"] = [i.to(DEVICE) for i in targets["classes"]]
    targets["bboxes"] = [i.to(DEVICE) for i in targets["bboxes"]]

    loss = (
        scores_loss(preds["scores"], scores).mean()
        + (scores * classes_loss(preds["classes"], classes)).mean()
    )
    return loss


from torchvision.transforms.functional import convert_image_dtype
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def train(dataloader, model, optimizer, epochs):
    model.train()

    scores_loss = sigmoid_focal_loss
    classes_loss = torch.nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epochs):
        logger.info(f"epoch {epoch}")
        total_loss = 0

        for images, targets in tqdm.tqdm(dataloader):
            # DEBUG
            img = convert_image_dtype(images[0].to(DEVICE), torch.uint8)
            bboxes = targets["bboxes"][0]
            bboxes = [[bbox[0]*IMG_SIZE[1], bbox[1]*IMG_SIZE[0], bbox[2]*IMG_SIZE[1], bbox[3]*IMG_SIZE[0]] for bbox in bboxes]
            bboxes = torch.as_tensor(bboxes).to(DEVICE)
            
            print(bboxes)
            result = draw_bounding_boxes(img, bboxes, width=3)
            
            show(result)
            break

            # Predict
            preds = model(images.to(DEVICE))

            loss = calculate_loss(images, targets, preds, scores_loss, classes_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        break
        mlflow.log_metric("loss", total_loss / len(dataloader), step=epoch)


def main():
    # Creating dataset
    kitti_dataset = dataset.AugmentedKITTIDetection("data/raw/kitti", "train")

    # TODO: Re-use gtsdb

    train_dataloader = DataLoader(
        kitti_dataset,
        16,
        True,
        num_workers=1,
        pin_memory=True,
        collate_fn=utils.collate_fn,
    )

    # Model creation
    model = create_mobilenetv3_large(N_CLASSES)
    model = model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters())
    logger.info(f"Training split {EPOCHS} epochs.")
    train(train_dataloader, model, opt, EPOCHS)
    torch.save(model, "models/model.pth")


if __name__ == "__main__":
    with mlflow.start_run():
        main()
