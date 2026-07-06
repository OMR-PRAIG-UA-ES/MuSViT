# models/token_grid.py
# -----------------------------------------------------------------------------
# Token-grid utility for MuSViT
# -----------------------------------------------------------------------------
# Goal
# - Enforce a fixed RCNN input size so MuSViT produces a fixed token grid.
# - For MuSViT 1024x1024 with patch_size=16:
#       64 x 64 = 4096 patch tokens.
#
# Notes
# - MuSViT expects fixed-size RGB inputs.
# - Positional embeddings are not resized here because the MuSViT checkpoints
#   used in this project are already trained for their configured image_size
#   usually 1024x1024.
# - This file intentionally keeps the public function name
#   `set_model_fixed_size_and_resize_pe` for compatibility with existing
#   training scripts.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

import torch.nn as nn


# ---- public API --------------------------------------------------------------

def desired_fixed_size_from_patch(patch_size: int, target_tokens: int = 4096) -> int:
    """
    Compute the square input side so that the backbone produces `target_tokens`
    patch tokens.

    Example:
        patch_size=16, target_tokens=4096
        sqrt(4096)=64
        fixed_size=64*16=1024
    """
    side_tokens = int(round(math.sqrt(target_tokens)))

    if side_tokens * side_tokens != target_tokens:
        raise ValueError(
            f"target_tokens must be a perfect square, got {target_tokens}."
        )

    return int(side_tokens * patch_size)


def set_model_fixed_size_and_resize_pe(
    model: nn.Module,
    kind: str,
    target_tokens: int = 4096,
    verbose: bool = True,
) -> int:
    """
    Enforce a fixed square input size for MuSViT-based Faster R-CNN models.

    Returns:
        The chosen fixed_size.

    This function does not resize positional embeddings. MuSViT checkpoints in
    this project are expected to already match the configured image size.
    """
    kind = (kind or "").lower()

    if kind not in ("musvit_small", "musvit_base", "musvit"):
        _vprint(verbose, f"[token-grid] '{kind}': not a MuSViT model; skipping.")
        return _get_transform_fixed_size(model)

    core = _get_musvit_core(model)
    if core is None:
        _vprint(verbose, "[token-grid] MuSViT core not found; skipping.")
        return _get_transform_fixed_size(model)

    if not hasattr(core, "patch_size"):
        _vprint(verbose, "[token-grid] MuSViT core lacks patch_size; skipping.")
        return _get_transform_fixed_size(model)

    patch_size = int(core.patch_size)
    fixed_size = desired_fixed_size_from_patch(patch_size, target_tokens)

    # Keep MuSViT metadata aligned with the detector transform.
    if hasattr(core, "image_size"):
        current_image_size = getattr(core, "image_size")

        if current_image_size is not None and int(current_image_size) != fixed_size:
            _vprint(
                verbose,
                f"[token-grid] warning: MuSViT checkpoint image_size={current_image_size}, "
                f"but requested fixed_size={fixed_size}."
            )

        core.image_size = int(fixed_size)

    _set_transform_fixed_size(model, fixed_size)

    _vprint(
        verbose,
        f"[token-grid] MuSViT: patch={patch_size} -> "
        f"fixed_size={fixed_size} for {target_tokens} tokens."
    )

    return fixed_size


# ---- helpers ----------------------------------------------------------------

def _get_musvit_core(model: nn.Module):
    """
    Return model.backbone.core when available.

    Expected structure:
        FasterRCNN
          └── backbone: MuSViTBackboneFPN
                └── core: SharedMuSViTBackbone
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None

    return getattr(backbone, "core", None)


def _get_transform_fixed_size(model: nn.Module) -> int:
    """
    Read the current fixed size from torchvision's GeneralizedRCNNTransform.
    Returns -1 if no fixed size is available.
    """
    transform = getattr(model, "transform", None)

    if transform is None:
        return -1

    fixed_size = getattr(transform, "fixed_size", None)
    if fixed_size is None:
        return -1

    if isinstance(fixed_size, (tuple, list)):
        return int(fixed_size[0])

    return int(fixed_size)


def _set_transform_fixed_size(model: nn.Module, size: int) -> None:
    """
    Update torchvision GeneralizedRCNNTransform to use a fixed square size.
    """
    transform = getattr(model, "transform", None)

    if transform is None:
        return

    size = int(size)

    transform.min_size = [size]
    transform.max_size = size

    if hasattr(transform, "fixed_size"):
        transform.fixed_size = (size, size)


def _vprint(verbose: bool, message: str) -> None:
    if verbose:
        print(message)