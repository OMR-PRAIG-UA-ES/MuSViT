"""Centralized environment / credential setup for the MuSViT monorepo.

Loads the single root ``.env`` once and normalizes the Hugging Face token name
so every experiment authenticates from the same key, regardless of which
variable it reads. This only populates ``os.environ`` (no network calls): wandb
picks up ``WANDB_API_KEY`` and the Hugging Face stack picks up ``HF_TOKEN`` /
``HUGGING_FACE_HUB_TOKEN`` from the environment automatically.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT: Path | None = None
_DONE = False


def find_repo_root() -> Path:
    """Locate the repository root (the directory holding ``pyproject.toml``)."""
    global _REPO_ROOT
    if _REPO_ROOT is not None:
        return _REPO_ROOT

    for base in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for parent in (base, *base.parents):
            if (parent / "pyproject.toml").is_file():
                _REPO_ROOT = parent
                return parent

    _REPO_ROOT = Path.cwd().resolve()
    return _REPO_ROOT


def setup(force: bool = False) -> None:
    """Load the root ``.env`` and bridge credential variable names (idempotent)."""
    global _DONE
    if _DONE and not force:
        return

    env_file = find_repo_root() / ".env"
    if env_file.is_file():
        load_dotenv(env_file)

    # One key, many readers: embeddings reads HF_TOKEN, the rest read
    # HUGGINGFACE_KEY, and the HF stack reads HUGGING_FACE_HUB_TOKEN.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_KEY")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_KEY"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    _DONE = True
