# models/musvit_frcnn.py
from collections import OrderedDict
from typing import Dict, Optional

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from models.musvit_backbone import SharedMuSViTBackbone


class MuSViTBackboneFPN(nn.Module):
    """
    Tiny FPN-like wrapper around ViT.
    - core.forward_features(x) returns B x D x h x w at stride~patch (e.g., 64x64 for 1024 input, patch=16).
    - Projects to out_channels (P3), then creates coarser maps (P4, P5) via strided convs.
    - Finally passes through torchvision's FPN for smoothing/alignment.
    """
    def __init__(self, vit_core: SharedMuSViTBackbone, out_channels: int = 256):
        super().__init__()
        self.core = vit_core
        d = vit_core.hidden_size

        # 1x1 projection to detector's channel width
        self.proj = nn.Conv2d(d, out_channels, 1)

        # Downsample blocks to synthesize P4 (/2) and P5 (/4) from P3
        self.down2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        # FPN: merges/smooths the three inputs
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels, out_channels, out_channels],
            out_channels=out_channels,
            extra_blocks=None,
        )
        self.out_channels = out_channels
        self.feat_names = ['p3', 'p4', 'p5']

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # base: B x D x h x w (e.g., h=w=64 for 1024 input w/ patch=16)
        base = self.core.forward_features(x)

        # Create three pyramid inputs:
        p3_in = self.proj(base)       # stride ~16 (finest)
        p4_in = self.down2(p3_in)     # stride ~32 (medium)
        p5_in = self.down4(p4_in)     # stride ~64 (coarsest)

        inputs = OrderedDict(zip(self.feat_names, (p3_in, p4_in, p5_in)))
        return self.fpn(inputs)


def _make_stride_scaled_anchors(patch: int, fixed_size: int = 1024) -> AnchorGenerator:
    """
    Build anchor sizes scaled to the backbone's patch stride AND image size.

    For high-resolution images (1024+), we need smaller anchors to detect
    small objects that become relatively smaller in the image.

    Args:
        patch: Patch size (e.g., 16)
        fixed_size: Input image size

    Returns:
        AnchorGenerator with appropriate anchor sizes
    """
    s = int(patch)

    if fixed_size >= 1024:
        sizes = (
            (s, 2*s, 4*s),        # (16, 32, 64) - P3: stride ~16
            (2*s, 4*s, 8*s),      # (32, 64, 128) - P4: stride ~32
            (4*s, 8*s, 16*s),     # (64, 128, 256) - P5: stride ~64
        )
        print(f"[anchors] Using SMALL anchors for {fixed_size}x{fixed_size}: {sizes}")
    else:
        sizes = (
            (2*s, 4*s, 8*s),      # (32, 64, 128) - P3: stride ~patch
            (4*s, 8*s, 16*s),     # (64, 128, 256) - P4: stride ~2*patch
            (8*s, 16*s, 32*s),    # (128, 256, 512) - P5: stride ~4*patch
        )
        print(f"[anchors] Using STANDARD anchors for {fixed_size}x{fixed_size}: {sizes}")

    aspects = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)  # Music-notation-friendly aspect ratios
    return AnchorGenerator(sizes=sizes, aspect_ratios=(aspects, aspects, aspects))


def build_model_musvit_frcnn(
    vit_model_path: str,
    hf_token: Optional[str],
    num_classes: int,
    out_channels: int = 256,
    fixed_size: int = 1024
):
    """
    Build a Faster R-CNN with MuSViT+FPN backbone.

    Notes:
    - fixed_size: Enforces square inputs because MuSViT expects fixed size
    - Normalization: Dataset handles normalization; transform uses neutral stats
    - Anchors: Scaled to patch size AND image size for proper coverage
    """
    vit_core = SharedMuSViTBackbone(vit_model_path, token=hf_token)

    backbone = MuSViTBackboneFPN(vit_core, out_channels=out_channels)

    anchor_generator = _make_stride_scaled_anchors(vit_core.patch_size, fixed_size=fixed_size)

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['p3', 'p4', 'p5'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    # I/O policy: fixed size; neutral stats (dataset already normalized)
    model.transform = GeneralizedRCNNTransform(
        min_size=fixed_size,
        max_size=fixed_size,
        image_mean=[0., 0., 0.],
        image_std=[1., 1., 1.],
        size_divisible=32,
        fixed_size=(fixed_size, fixed_size),
    )

    return model