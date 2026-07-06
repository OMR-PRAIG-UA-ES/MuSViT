"""Data-loading and label-preparation helpers.

Covers the steps between raw files on disk and the tensors the model needs:
  * discover image / ground-truth pairs in a folder,
  * turn symbol strings into integer ids with a LabelEncoder,
  * pad variable-length sequences to a common length,
  * filter out sequences that are too long for the CTC time axis.
"""

import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import copy

def load_image_gt_pairs(folder_path):
    """Discover (image_path, symbol_list) pairs in a dataset folder.

    Expected file naming convention inside ``folder_path``:
        <id>_region.png   -> the cropped staff-region image
        <id>_gt.txt        -> its ground truth: whitespace-separated symbols

    An image with no matching ground-truth file is skipped with a warning.

    Args:
        folder_path (str): directory containing the paired files.

    Returns:
        list[tuple[str, list[str]]]: (image_path, list_of_symbol_strings).
    """
    data = []

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter for image files ending with '_region.png'
    image_files = [f for f in files if f.endswith('_region.png')]

    for img_file in image_files:
        # Extract the <id> from the filename
        base_id = img_file.replace('_region.png', '')

        # Construct the corresponding text filename
        txt_file = f"{base_id}_gt.txt"
        txt_path = os.path.join(folder_path, txt_file)
        img_path = os.path.join(folder_path, img_file)

        if os.path.exists(txt_path):
            # Read the ground truth and split on whitespace into a symbol list.
            with open(txt_path, 'r', encoding='utf-8') as f:
                symbols = f.read().strip().split()
            data.append((img_path, symbols))
        else:
            # No label for this image -> warn and skip it.
            print(f"Warning: No ground truth found for image {img_file}")

    return data


def encode_sequences(sequences, bias=0):
    """Map symbol strings to integer ids using a fitted LabelEncoder.

    Args:
        sequences (list[list[str]]): tokenised symbol sequences.
        bias (int): value added to every encoded id. Use bias=1 to reserve id 0
            for the CTC blank symbol.

    Returns:
        tuple: (le, encoded_sequences) where ``le`` is the fitted LabelEncoder
        (its ``classes_`` gives the vocabulary) and ``encoded_sequences`` is the
        list of integer-id arrays.
    """
    # Collect the full symbol vocabulary across all sequences.
    unique_symbols = set()
    for sequence in sequences:
        unique_symbols.update(sequence)

    # Fit LabelEncoder only on unique symbols
    le = LabelEncoder()
    le.fit(list(unique_symbols))

    # Map symbols to ids with bias
    encoded_sequences = list(map(le.transform, sequences))
    if bias>0:
        # Shift all ids up so that lower ids (e.g. 0) can be reserved for blank.
        encoded_sequences = list(map(lambda seq: seq+bias, encoded_sequences))

    return le, encoded_sequences


def pad_sequences(sequences, pad_token):
    """Right-pad all sequences to the longest length in the batch.

    Args:
        sequences (list[np.ndarray]): integer-encoded sequences.
        pad_token (int): id used for padding (here the blank id).

    Returns:
        tuple: (padded_sequences, sequences_len) where ``sequences_len`` holds
        each sequence's original (pre-padding) length for use by CTC.
    """
    # Record true lengths before padding (deepcopy avoids side effects).
    sequences_len = list(map(len, copy.deepcopy(sequences)))
    max_len = max([len(seq) for seq in sequences])
    # Append (max_len - len) pad tokens to each sequence.
    pad_sequences = list(map(lambda seq: np.concat((seq, np.array([pad_token] * (max_len-len(seq))))), sequences))

    return pad_sequences, sequences_len


def filter_max_len(imgs, seqs, len_seqs, max_len):
    """Drop samples whose target sequence is longer than ``max_len``.

    CTC requires the input length (number of column frames) to be at least the
    target length. Sequences that violate this are removed so they never break
    the loss during training.

    Args:
        imgs (list): image paths.
        seqs (list): padded encoded sequences.
        len_seqs (list[int]): true sequence lengths (the filter key).
        max_len (int): maximum allowed sequence length (== number of columns).

    Returns:
        tuple: the three lists filtered in lock-step.
    """
    imgs = [img for idx, img in enumerate(imgs) if len_seqs[idx]<=max_len]
    seqs = [seq for idx, seq in enumerate(seqs) if len_seqs[idx]<=max_len]
    len_seqs = [len_seq for idx, len_seq in enumerate(len_seqs) if len_seqs[idx]<=max_len]

    return imgs, seqs, len_seqs
