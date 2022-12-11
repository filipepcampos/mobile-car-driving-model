import numpy as np
import torch


def set_classes(class_key):
    """Sets the respective class wherever the object is, according to the given
    slicing."""

    def func(grid, yy, xx, datum, i):
        klass = datum[class_key][i]
        grid[yy, xx] = klass

    return func


def slice_across_bbox():
    """Choose all grid locations that contain the entirety of the object
    (similar to FCOS)."""

    def func(height, width, bbox):
        yy = slice(int(np.floor(bbox[1] * height)), int(np.ceil(bbox[3] * height)))
        xx = slice(int(np.floor(bbox[0] * width)), int(np.ceil(bbox[2] * width)))
        return yy, xx

    return func


def new_id(id_value):
    """Grid 1xhxw."""

    def func(height, width):
        return torch.full((1, height, width), id_value, dtype=torch.int8)

    return func


def to_numpy_array():
    def func(image, **data):
        image = np.array(image)
        return {"image": image, **data}

    return func


def to_tensor():
    def func(image, **data):
        image = torch.tensor(image)
        return {"image": image, **data}

    return func


def permute(order):
    def func(image, **data):
        image = image.permute(*order)
        return {"image": image, **data}

    return func
