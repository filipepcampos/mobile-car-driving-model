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


def train(dataloader, model, optimizer, epochs):
    """Trains the model.

    `weight_loss_fns` and `loss_fns` are dictionaries, specifying
    whether the loss should be applied to that grid location and what
    loss to apply.
    """

    scores_loss = sigmoid_focal_loss
    classes_loss = torch.nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epochs):
        logger.info(f"epoch {epoch}")
        total_loss = 0
        for images, targets in tqdm.tqdm(dataloader):
            images = images.to(DEVICE)
            preds = model(images)

            slices = grid.slice_across_bboxes(
                GRID_SIZE[0],
                GRID_SIZE[1],
                targets["bboxes"],
            )
            scores = grid.scores(8, 32, slices)
            classes = grid.classes(8, 32, slices, targets["classes"])

            loss = (
                scores_loss(preds["scores"], scores).mean()
                + (scores * classes_loss(preds["classes"], classes)).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        mlflow.log_metric("loss", total_loss / len(dataloader), step=epoch)


def main():
    # Creating dataset
    kitti_dataset = dataset.AugmentedKITTIDetection("data/raw/kitti", "train")

    # TODO: Re-use gtsdb

    train_dataloader = DataLoader(
        kitti_dataset,
        8,
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
