# MuSViT

Official repository for the ECCV 2026 publication: *"MuSViT: A Foundation Vision Model for Sheet Music Representation"*.

This is a **monorepo** collecting the experiments around MuSViT. Every experiment
lives under [`experiments/`](experiments/) and is launched from the repository
root through a single cross-platform CLI: **`musvit`**.

## Repository layout

| Folder | Description |
| --- | --- |
| [`musvit/`](musvit/) | Central launcher (`musvit` CLI): discovers and runs the experiments. |
| [`experiments/full_page_omr/`](experiments/full_page_omr/) | Fine-tune MuSViT for full-page Optical Music Recognition (OMR). |
| [`experiments/embeddings_test/`](experiments/embeddings_test/) | Compare vision-encoder embedding distances against transcription distances. |

## Installation

The whole monorepo shares a single [uv](https://docs.astral.sh/uv/) project
defined at the repository root ([`pyproject.toml`](pyproject.toml) /
[`uv.lock`](uv.lock)). Install the environment once from the root:

```bash
uv sync
```

This also installs the `musvit` console command into the project environment.

## Environment variables

Experiments authenticate against [Weights & Biases](https://wandb.ai) and the
[Hugging Face Hub](https://huggingface.co). These credentials are shared across
the whole monorepo through a single `.env` file at the repository root.

1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Fill in your keys in `.env`:
   ```dotenv
   WANDB_API_KEY=...
   HUGGINGFACE_KEY=...
   ```

`.env` is git-ignored; only `.env.example` is committed. The `musvit` launcher
loads this `.env` once at startup and automatically bridges `HUGGINGFACE_KEY` to
`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, so every experiment authenticates from the
same key regardless of which variable it reads.

## Running experiments

All experiments are launched from the repository root with `musvit`. List what is
available:

```bash
uv run musvit list
```

Each subcommand maps its options directly onto the experiment's Python
entrypoint, and `--help` is generated automatically:

```bash
uv run musvit full-page-omr --help
uv run musvit embeddings run --help
uv run musvit embeddings sweep --help
```

### Full-page OMR (fine-tuning)

```bash
uv run musvit full-page-omr \
  --config_path experiments/full_page_omr/config/Polish_Scores/finetuning.json \
  --experiment_name my_experiment \
  --finetuning CL
```

Checkpoints and W&B logs are written inside
`experiments/full_page_omr/` (`weights/`, `wandb_logs/`).

### Embeddings analysis (single encoder)

```bash
uv run musvit embeddings run --model facebook/dinov2-base --device 0
```

Embeddings, results and logs are written inside
`experiments/embeddings_test/` (`embeddings/`, `results/`).

### Embeddings sweep (several encoders)

Cross-platform replacement for the old `script_run_experiments.sh` (works on
Windows too). Runs a preset list of encoders, writing a per-model log and a
`summary.tsv`:

```bash
uv run musvit embeddings sweep --models_set light
uv run musvit embeddings sweep --models_set default --device 0
uv run musvit embeddings sweep --only facebook/dinov2-base
```

Model sets: `light`, `default`, `full` (defined in
[`experiments/embeddings_test/model_sets.py`](experiments/embeddings_test/model_sets.py)).

> **Note on GPU memory:** the sweep loads encoders one after another in the same
> process and frees GPU memory between models. If you hit out-of-memory errors
> with the largest sets, run the encoders individually with `musvit embeddings run`.
