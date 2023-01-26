import random
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

from data import KITTIDetection

def show(img):
    img = img.detach()
    img = F.to_pil_image(img)
    plt.imshow(np.asarray(img))
    plt.show()

# Convert bboxes from [0,1] scale to pixel scale
def rescale_bboxes(image, bboxes):
    img_size = image.size()
    return [[bbox[0]*img_size[2], bbox[1]*img_size[1], bbox[2]*img_size[2], bbox[3]*img_size[1]] for bbox in bboxes]

def show_random_image():

    dataset = KITTIDetection("data/raw/kitti", "train")
    datum = dataset[random.randint(0, len(dataset))]
    image, bboxes = datum["image"], datum["bboxes"]
    bboxes = torch.as_tensor(rescale_bboxes(image, bboxes))
    
    image = F.convert_image_dtype(image, torch.uint8)

    result = draw_bounding_boxes(image, bboxes, width=3)
    show(result)

if __name__ == '__main__':
    show_random_image()