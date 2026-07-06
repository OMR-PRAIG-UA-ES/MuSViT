# models/musvit_backbone.py
"""
MuSViT backbone for object detection.

- RGB input (3 channels)
- 1024x1024 image size
- Patch size 16x16
- Simple preprocessing: ToTensor + Resize(1024, 1024)
"""
from transformers import ViTModel
import torch
import torch.nn as nn


class SharedMuSViTBackbone(nn.Module):
    """
    MuSViT backbone for detection.

    Requirements:
    - Inputs must be exactly config.image_size x config.image_size (e.g., 1024x1024)
    - RGB images (3 channels)

    Returns a single feature map at stride ~patch_size (1/16 for patch_size=16).
    Exposes out_channels for TorchVision compatibility.
    """
    def __init__(
        self,
        model_path: str,
        token: str = None,
        proj_dim: int = 256,
        require_fixed_size: bool = True,
        freeze_pos: bool = False,
        freeze_until: int | None = None
    ):
        """
        Args:
            model_path: HuggingFace model path or local path
            token: HuggingFace token (if needed)
            proj_dim: Output channels after projection (default 256 for FPN)
            require_fixed_size: Enforce exact image_size from config
            freeze_pos: Freeze positional embeddings
            freeze_until: Freeze encoder layers 0 to freeze_until-1
        """
        super().__init__()

        print(f"[MuSViT] Loading model: {model_path}")

        self.vit = ViTModel.from_pretrained(
            model_path,
            token=token,
            add_pooling_layer=False,  # No pooling needed for detection
        )

        self.hidden_size = self.vit.config.hidden_size
        self.patch_size = self.vit.config.patch_size
        self.image_size = getattr(self.vit.config, "image_size", None)
        self.require_fixed_size = require_fixed_size
        self.model_path = model_path

        print(f"[MuSViT] Configuration:")
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  patch_size: {self.patch_size}")
        print(f"  image_size: {self.image_size}")
        print(f"  require_fixed_size: {require_fixed_size}")

        if self.image_size is not None and self.image_size != 1024:
            print(f"[MuSViT] Warning: Expected image_size=1024, got {self.image_size}")

        if self.patch_size != 16:
            print(f"[MuSViT] Warning: Expected patch_size=16, got {self.patch_size}")

        # Keep pretrained pos embeddings (if they exist) and optionally freeze
        if freeze_pos and getattr(self.vit.embeddings, "position_embeddings", None) is not None:
            self.vit.embeddings.position_embeddings.requires_grad_(False)
            print(f"[MuSViT] Froze positional embeddings")

        # Projection to detector channel width (256 by default)
        self.proj = nn.Conv2d(self.hidden_size, proj_dim, kernel_size=1)
        self.out_channels = proj_dim
        print(f"[MuSViT] Added projection: {self.hidden_size} -> {proj_dim} channels")

        # Optionally freeze early encoder layers
        if freeze_until is not None and hasattr(self.vit.encoder, "layer"):
            frozen_count = 0
            for i, blk in enumerate(self.vit.encoder.layer):
                if i < freeze_until:
                    for p in blk.parameters():
                        p.requires_grad = False
                    frozen_count += 1
            print(f"[MuSViT] Froze first {frozen_count} encoder layers")

        print(f"[MuSViT] Backbone ready")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from MuSViT.

        Args:
            x: Input tensor [B, C, H, W]
               - C must be 3 (RGB)
               - H, W should be config.image_size (e.g., 1024)

        Returns:
            Spatial feature map [B, hidden_size, h, w]
            where h = H // patch_size, w = W // patch_size
        """
        B, C, H, W = x.shape

        if C != 3:
            raise ValueError(
                f"Expected 3 channels (RGB), got {C}."
            )

        if self.require_fixed_size and self.image_size is not None:
            if H != self.image_size or W != self.image_size:
                raise ValueError(
                    f"Expected {self.image_size}x{self.image_size}, got {H}x{W}. "
                    f"Ensure preprocessing includes Resize({self.image_size}, {self.image_size})."
                )

        # ViT output: [B, 1+N, D] where [0] is the CLS token
        out = self.vit(pixel_values=x).last_hidden_state  # [B, 1+N, D]
        tokens = out[:, 1:, :]  # [B, N, D] - remove CLS token

        h = H // self.patch_size
        w = W // self.patch_size
        N = tokens.shape[1]

        expected_N = h * w
        if N != expected_N:
            raise RuntimeError(
                f"Token count mismatch: got {N} tokens, expected {expected_N} "
                f"(h={h}, w={w}, patch_size={self.patch_size})"
            )

        # Reshape: [B, N, D] -> [B, h, w, D] -> [B, D, h, w]
        feat = tokens.reshape(B, h, w, self.hidden_size).permute(0, 3, 1, 2).contiguous()

        return feat  # [B, hidden_size, h, w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass with projection.

        Args:
            x: Input tensor [B, C, H, W]
               - Must be RGB (C=3)
               - Must be square (H=W=image_size)

        Returns:
            Projected features [B, out_channels, h, w]
        """
        feat = self.forward_features(x)  # [B, hidden_size, h, w]
        feat = self.proj(feat)           # [B, out_channels, h, w]
        return feat