"""Batch data generator for image inputs stored as paths, PIL images, or NumPy arrays."""

import numpy as np
import cv2
import torch

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class DataGenerator:
    def __init__(self, list_filenames, labels, config, encoder=None):
        # list_filenames can contain:
        #  - str paths
        #  - PIL.Image instances
        #  - np.ndarray images
        self.list_filenames = list_filenames
        self.labels = labels
        self.config = config
        self.batch_size = config.batch_size
        self.num_batches = int(max(1, np.ceil(len(self.list_filenames) / self.batch_size)))
        self.current_idx = 0
        self._cache_data = {}   # Optional cache for path-based images.
        self.encoder = encoder

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.current_idx = 0
        return self

    def getNumberImages(self):
        return len(self.list_filenames)

    def getCompleteListData(self):
        list_images = []
        list_labels = []
        for images, labels in self:
            list_images += images
            list_labels += labels
        return list_images, list_labels

    def _load_rgb(self, item):
        """
        Return an RGB uint8 np.ndarray with shape (H, W, 3).
        """
        # Case 1: image path.
        if isinstance(item, str):
            if item in self._cache_data:
                return self._cache_data[item]

            image = cv2.imread(item)
            if image is None:
                raise ValueError(f"Could not read image: {item}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Optional cache.
            # self._cache_data[item] = image
            return image

        # Case 2: PIL image.
        if PILImage is not None and isinstance(item, PILImage.Image):
            return np.array(item.convert("RGB"))

        # Case 3: pre-loaded NumPy array.
        if isinstance(item, np.ndarray):
            img = item
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            return img

        raise TypeError(f"Unsupported image type: {type(item)}")

    def __next__(self):
        if self.current_idx >= self.num_batches:
            raise StopIteration

        batch_start = self.current_idx * self.batch_size
        batch_end = min((self.current_idx + 1) * self.batch_size, len(self.list_filenames))

        batch_items = self.list_filenames[batch_start:batch_end]
        batch_labels = self.labels[batch_start:batch_end]

        batch_images = []

        for idx_data, item in enumerate(batch_items):
            image = self._load_rgb(item)
            label = batch_labels[idx_data]

            try:
                label["labels"] = torch.tensor([self.config.c2i[c] for c in label["labels"]], dtype=torch.long)
                label["boxes"] = torch.tensor(label["boxes"], dtype=torch.float32)
            except Exception:
                pass

            if self.encoder is not None:
                image = self.encoder.processor(image, return_tensors="pt").pixel_values.to(self.config.device)

            batch_images.append(image)

        self.current_idx += 1
        return batch_images, batch_labels