"""PyTorch ``Dataset`` for CTC-based staff recognition.

Wraps parallel lists of image paths and (already encoded + padded) target
sequences into a map-style dataset. On access it loads an image, optionally
augments it, applies the model's pre-processing transform, and returns the
tensors expected by the CTC training loop.
"""

from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T


class CTC_ds(Dataset):
    """Map-style dataset yielding (image, target_sequence, target_length).

    Args:
        img_paths (list[str]): file paths to the score-region images.
        seqs (list[np.ndarray]): padded, integer-encoded target sequences,
            one per image (all padded to the same length).
        len_seqs (list[int]): the *true* (unpadded) length of each target
            sequence; needed by CTC to know how much of ``seqs`` is real.
        processor (callable): torchvision transform that resizes/tensorises a
            PIL image into the model's expected input tensor.
        augment (callable, optional): an albumentations pipeline applied before
            ``processor``. Pass ``None`` for validation/test (no augmentation).
    """

    def __init__(self, img_paths, seqs, len_seqs, processor, augment=None):
        self.img_paths = img_paths
        self.seqs = seqs
        self.len_seqs = len_seqs
        self.augment = augment
        self.processor = processor

    def __getitem__(self, index):
        """Load and prepare a single training example.

        Returns:
            tuple: (image_tensor, target_sequence_tensor, target_length) where
                * image_tensor is the output of ``processor``,
                * target_sequence_tensor is a LongTensor of symbol ids,
                * target_length is the unpadded length (int).
        """
        # Always decode to 3-channel RGB so grayscale scores match the ViT's
        # expected input channels.
        image = Image.open(self.img_paths[index]).convert("RGB")

        # Optional on-the-fly augmentation. albumentations works on numpy
        # arrays, so we convert PIL -> ndarray, augment, then convert back.
        if self.augment is not None:
            image = Image.fromarray(self.augment(image= np.array(image))['image'])

        # Resize + tensorise (and, for linear-probing, pad) into model input.
        image = self.processor(image)

        # seqs[index] is a numpy array; convert to a long tensor for CTC.
        return image, torch.from_numpy(self.seqs[index]).long(), self.len_seqs[index]

    def __len__(self):
        """Number of examples in the dataset."""
        return len(self.img_paths)
