def collate_fn(batch):
    """The number of bounding boxes varies for each image, therefore the
    default PyTorch `collate` function (which creates the batches) must be
    replaced so that only images are turned into tensors."""
    import torch

    imgs = torch.stack([torch.as_tensor(d["image"]) for d in batch])
    targets = {key: [torch.as_tensor(d[key]) for d in batch] for key in batch[0].keys()}
    return imgs, targets
