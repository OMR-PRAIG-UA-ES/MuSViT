"""Image-augmentation pipeline used during training.

The augmentations here are designed for photographs / scans of musical scores.
They emulate the kinds of degradation and variation seen in real archival
documents (ink dilation/erosion, paper noise, lighting changes, slight skew,
blur) so that the recognition model generalises beyond the exact appearance of
the training images.
"""

import albumentations as A
import cv2



def get_ssl_transform():
    """Build and return the albumentations augmentation pipeline.

    The name keeps the "ssl" prefix for historical reasons (it originated in a
    self-supervised-learning setup), but here it is simply the supervised
    training-time augmentation policy. Every transform is applied on the fly
    inside the dataset's ``__getitem__``.

    Returns:
        A.Compose: a callable transform. Call it as ``transform(image=np_img)``
        and read the result from the returned dict's ``'image'`` key.
    """
    custom_transforms = A.Compose([
            # Thicken or thin the strokes to mimic different pen/print weights
            # and ink spread. Exactly one of the two is chosen when this block
            # fires (p=0.6 for the whole OneOf).
            A.OneOf([
                A.Morphological(scale=(2, 2), operation='dilation',p=1.0),
                A.Morphological(scale=(2, 2), operation='erosion',p=1.0)
            ], p=0.6),
            # Occasionally sharpen edges to vary stroke crispness.
            A.Sharpen(p=0.25),
            # Small rotation (+/-3 degrees) to simulate imperfect page alignment.
            # BORDER_REPLICATE extends edge pixels instead of adding black borders.
            A.Rotate((-3, 3), border_mode=cv2.BORDER_REPLICATE),
            # Sensor / scan noise.
            A.GaussNoise(std_range=(0.01, 0.15), per_channel=False, p=0.3),
            # Photometric variation: brightness / contrast / saturation / hue.
            # This is the strongest single augmentation (p=0.75) and covers the
            # wide range of paper tones and lighting in real documents.
            A.ColorJitter(brightness=(0.25, 1.75), contrast=(0.25, 1.75), saturation=(0.25, 1.75), hue=(-0.05, 0.05), p=0.75),
            # Mild blur: either optical (Gaussian) or camera-shake (motion).
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 4), p=1.0),
                A.MotionBlur(blur_limit=(3, 4), p=1.0),
            ], p=0.25),
            # Rarely drop colour entirely to encourage colour-invariance.
            A.ToGray(p=0.1),
        ], p=1.0)  # p=1.0: the Compose itself always runs (individual ops still gated by their own p).

    return custom_transforms

