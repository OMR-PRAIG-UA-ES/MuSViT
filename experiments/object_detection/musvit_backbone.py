# models/detector_factory.py
from __future__ import annotations
from typing import Tuple, Dict, Any


def build_detector(kind: str, num_classes: int, **kwargs) -> Tuple[object, Dict[str, Any]]:
    """
    Build detector model and return (model, defaults).

    defaults dict contains:
      - opt_kind: "sgd" | "adamw"
      - wd: weight decay
      - lr_backbone, lr_head: learning rates for backbone and head
      - backbone_prefix: parameter name prefix for backbone (for freezing/LR)
      - fixed_size: recommended input size for this model

    Args:
        kind: Model type ("musvit_small" or "musvit_base")
        num_classes: Number of classes including background
        **kwargs: Model-specific args (musvit_small_model_path, musvit_base_model_path,
                  hf_token, out_channels, fixed_size)
    """
    kind = kind.lower()

    if kind not in ("musvit_small", "musvit_base"):
        raise ValueError(f"Unknown detector kind: {kind}")

    from models.musvit_frcnn import build_model_musvit_frcnn

    fixed_size = kwargs.get("fixed_size", 1024)

    if kind == "musvit_small":
        vit_model_path = kwargs.get("vit_small_model_path")
    else:  # musvit_base
        vit_model_path = kwargs.get("vit_base_model_path")

    model = build_model_musvit_frcnn(
        vit_model_path=vit_model_path,
        hf_token=kwargs.get("hf_token"),
        num_classes=num_classes,
        out_channels=kwargs.get("out_channels", 256),
        fixed_size=fixed_size,
    )

    defaults = dict(
        opt_kind="adamw",
        wd=kwargs.get("wd_default", 1e-2),
        lr_backbone=kwargs.get("lr_backbone_default", 5e-5),
        lr_head=kwargs.get("lr_head_default", 2.5e-4),
        backbone_prefix="backbone.core.vit",
        fixed_size=fixed_size,
    )
    return model, defaults