"""Preset vision-encoder lists for the embeddings sweep.

Migrated verbatim from the former ``script_run_experiments.sh`` so the sweep is
available cross-platform from the ``musvit`` CLI.
"""

LIGHT_MODELS = [
    "microsoft/beit-base-patch16-224-pt22k",
    "microsoft/dit-base",
    "facebook/dinov2-base",
    "google/siglip-so400m-patch14-224",
    "google/siglip2-so400m-patch14-224",
]

DEFAULT_MODELS = [
    "microsoft/beit-base-patch16-224-pt22k",
    "microsoft/beit-large-patch16-512",
    "microsoft/dit-base",
    "google/siglip-so400m-patch14-224",
    "google/siglip2-so400m-patch14-224",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-so400m-patch14-384",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "microsoft/kosmos-2.5",
    "carlospm12/LSMT-MAE-Small-1024-16",
    "carlospm12/LSMT-MAE-Base-1024-16",
    "carlospm12/LSMT-MAE-Large-1024-16",
]

FULL_MODELS = [
    "microsoft/beit-base-patch16-224-pt22k",
    "microsoft/beit-large-patch16-512",
    "microsoft/dit-base",
    "models/MAE-8X8-Small",
    "google/siglip-so400m-patch14-224",
    "google/siglip2-so400m-patch14-224",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-so400m-patch14-384",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
    "google/paligemma2-3b-pt-224",
    "google/paligemma2-3b-pt-448",
    "google/paligemma2-3b-pt-896",
    "Qwen/Qwen3-VL-8B-Instruct",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "microsoft/kosmos-2.5",
    "carlospm12/LSMT-MAE-Small-1024-16",
    "carlospm12/LSMT-MAE-Base-1024-16",
    "carlospm12/LSMT-MAE-Large-1024-16",
]

MODEL_SETS = {
    "light": LIGHT_MODELS,
    "default": DEFAULT_MODELS,
    "full": FULL_MODELS,
}
