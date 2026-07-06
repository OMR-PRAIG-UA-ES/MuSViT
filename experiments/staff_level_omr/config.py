"""Static configuration: dataset locations and backbone definitions.

Two dictionaries are exported:

* ``data_paths``  -> maps a short dataset name to the folder on disk that
                     contains its ``*_region.png`` / ``*_gt.txt`` file pairs.
* ``data_models`` -> maps a short model name to everything needed to load and
                     wire up the corresponding pre-trained ViT backbone.

Edit the paths below to point at your local copies of the data before running
any training.
"""

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
# NOTE: The paths are intentionally placeholders. The datasets themselves are
# distributed on request or are private, so you must obtain them from the
# original authors and then update these paths to your local folders.
data_paths = {
    'capitan': '/path/capitan/data',
    'catedrales': '/path/catedrales/data',
    'fmt': '/path/fmt/data',
    'guatemala': '/path/guatemala/data',
    'seils': '/path/seils/data',
}

# ---------------------------------------------------------------------------
# Pre-trained backbone definitions
# ---------------------------------------------------------------------------
# Each entry describes one MusViT variant hosted on the Hugging Face Hub:
#   * link        -> Hub repo id passed to ``ViTModel.from_pretrained``.
#   * patch_size  -> side length (in pixels) of one ViT patch. Combined with
#                    --shape_patches it fixes the input resolution.
#   * dim         -> hidden/embedding size of the ViT, i.e. the width of each
#                    patch token; used to size the projection layer.
#   * start_patch -> index of the first *content* token. 1 skips the leading
#                    [CLS] token so only spatial patch tokens are kept.
data_models = {
    'musvit_light': {
        'link': 'PRAIG/musvit-light',
        'patch_size': 16,
        'dim': 384,
        'start_patch': 1
    },
    'musvit': {
        'link': 'PRAIG/musvit',
        'patch_size': 16,
        'dim': 768,
        'start_patch': 1
    }
}
