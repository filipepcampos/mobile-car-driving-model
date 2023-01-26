import numpy as np
import torch


def set_classes(class_key, grid, yy, xx, datum, i):
    """Sets the respective class wherever the object is, according to the given
    slicing."""

    klass = datum[class_key][i]
    grid[yy, xx] = klass
    return grid


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


def new_id(id_value, height, width):
    """Grid 1xhxw."""

    return torch.full((1, height, width), id_value, dtype=torch.int8)


def scores(height, width, batch_slices):
    """Grid with 1 wherever the object is, 0 otherwise, according to the chosen
    slice strategy."""
    n = len(batch_slices)
    grid = torch.zeros((n, 1, height, width), dtype=torch.float32)
    for i, slices in enumerate(batch_slices):
        for yy, xx in slices:
            grid[i, :, yy, xx] = 1
    return grid


def classes(height, width, batch_slices, batch_classes):
    """Sets the respective class wherever the object is, according to the given
    slicing."""
    n = len(batch_slices)
    grid = torch.zeros((n, height, width), dtype=torch.int64)
    for i, (slices, classes) in enumerate(zip(batch_slices, batch_classes)):
        for (yy, xx), klass in zip(slices, classes):
            grid[i, yy, xx] = klass
    return grid
