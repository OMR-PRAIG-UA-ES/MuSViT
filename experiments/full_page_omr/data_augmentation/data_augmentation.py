from PIL import Image
from torchvision import transforms
from .transforms_custom import ElasticDistortion, RandomTransform, ContrastAdjust
from .. import _globals

import numpy as np


def set_up_processor(model):
    # MuSViT (LSMT-MAE) consumes plain RGB tensors, so no HF image processor is needed.
    global model_name
    model_name = model


def augment(image):
    resolution = _globals.resolution
    distortion_perspective = np.random.uniform(0,0.3)

    elastic_dist_magnitude = np.random.randint(1, 20 + 1)
    elastic_dist_kernel = np.random.randint(1, 3 + 1)
    magnitude_w, magnitude_h = (elastic_dist_magnitude, 1) if np.random.randint(2) == 0 else (1, elastic_dist_magnitude)

    ctr_factor = np.random.uniform(0.7, 2)

    transforms_list = [
            transforms.ToPILImage(),
            transforms.RandomPerspective(distortion_scale=distortion_perspective, p=0.1, interpolation=Image.BILINEAR, fill=255),
            transforms.RandomApply([ElasticDistortion(grid=(elastic_dist_kernel, elastic_dist_kernel), magnitude=(magnitude_w, magnitude_h), min_sep=(1,1))], p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(3, 5))], p=0.1),
            transforms.RandomApply([ContrastAdjust(factor=ctr_factor)], p=0.1),
        ]
    if resolution > 0:
        transforms_list.append(transforms.Resize([resolution,resolution]))
    transform = transforms.Compose(transforms_list)

    image = transform(image)

    img_rgb = image.convert('RGB')
    tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)

    return tensor


def convert_img_to_tensor(image):
    resolution = _globals.resolution

    transforms_list = [
            transforms.ToPILImage(),
            ]
    if resolution > 0:
        transforms_list.append(transforms.Resize([resolution,resolution]))
    transform = transforms.Compose(transforms_list)

    image = transform(image)

    img_rgb = image.convert('RGB')
    tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)

    return tensor