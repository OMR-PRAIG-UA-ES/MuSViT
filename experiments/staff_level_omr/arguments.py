"""Command-line argument definitions for the training entry point.

This module centralises every hyper-parameter and experiment switch that
``train.py`` reads at start-up.  Keeping the parser in its own file means the
same argument specification can be imported from other scripts (e.g. an
evaluation-only script) without duplicating the option list.

The resulting ``parser_train`` object is consumed in ``train.py`` via
``args = parser_train.parse_args()``.
"""

import argparse


# A single ArgumentParser instance shared across the project.
# It is created at import time so that simply importing this module gives you
# a fully-configured parser.
parser_train = argparse.ArgumentParser(description="Arguments for training")

# ---------------------------------------------------------------------------
# Experiment identity: what to train and how.
# ---------------------------------------------------------------------------

# Which dataset to use. The string is used as a key into ``config.data_paths``
# to resolve the folder that holds the image / ground-truth pairs.
parser_train.add_argument("--ds_name", type=str, default='catedrales', help="Dataset name")

# Which pre-trained backbone to load. The string is a key into
# ``config.data_models`` and selects the Hugging Face checkpoint plus its
# patch size / embedding dimension. Typical values: 'musvit', 'musvit_light'.
parser_train.add_argument("--model_name", type=str, default='musvit', help="Model name")

# Fine-tuning strategy for the backbone:
#   * 'linear_prob' -> freeze the ViT entirely and train only the LSTM+CTC head.
#   * 'lora'        -> keep the ViT frozen but inject trainable LoRA adapters
#                      into its attention projections (query/key/value).
parser_train.add_argument('--method', default='lora', type=str, choices=['linear_prob', 'lora'], help="""Method to use""")

# Number of ViT patches kept along (rows, cols) of the score crop.
#   * rows  -> vertical patches spanning the staff height.
#   * cols  -> horizontal patches; this becomes the CTC time axis length.
# The actual pixel size fed to the model is patch_size * (rows, cols).
parser_train.add_argument("--shape_patches", nargs=2, type=int, default=[8, 64],  help="Two integers as a tuple")

# ---------------------------------------------------------------------------
# Optimisation hyper-parameters.
# ---------------------------------------------------------------------------

# Mini-batch size used by every DataLoader (train / val / test).
parser_train.add_argument("--batch_size", type=int, default=8, help="Batch size in dataloader")

# First epoch at which validation (and therefore checkpointing / early
# stopping) begins. Skipping evaluation for the first epochs avoids wasting
# time while the head is still essentially random.
parser_train.add_argument("--start_eval", type=int, default=20, help="Epoch to start eval")

# Adam learning rate.
parser_train.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
