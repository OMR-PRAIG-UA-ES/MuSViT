<p align="center">
  <a href=""><img src="https://raw.githubusercontent.com/OMR-PRAIG-UA-ES/MuSViT/refs/heads/main/resources/musvit_logo.svg" alt="MuSViT-logo" width="100"></a>
</p>
<h1 align="center">MuSViT: A Foundation Vision Model for Sheet Music Representation</h1>
<h4 align="center">📄 Official publication at ECCV 2026.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.x-orange" alt="Python">
  <img src="https://img.shields.io/badge/uv-managed-6340ac?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/-Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black" alt="W&B">
  <img src="https://img.shields.io/static/v1?label=License&message=CC%20BY-NC-SA%204.0&color=blue" alt="License">
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#repository-layout">Repository Layout</a> •
  <a href="#installation">Installation</a> •
  <a href="#environment-variables">Environment Variables</a> •
  <a href="#running-experiments">Running Experiments</a> •
  <a href="#citations">Citations</a> •
  <a href="#license">License</a>
</p>

<a name=about></a>
## 📚 About

This repository contains the official code for **MuSViT: A Foundation Vision Model for Sheet Music Representation**, presented at ECCV 2026.

This is a **monorepo** collecting the experiments around MuSViT. Every experiment lives under [`experiments/`](experiments/) and is launched from the repository root through a single cross-platform CLI: **`musvit`**.

This project includes:
- A central CLI (`musvit`) to discover and run all experiments from the repository root.
- A full-page Optical Music Recognition (OMR) fine-tuning pipeline.
- An embeddings analysis pipeline to compare vision-encoder embedding distances against transcription distances.
- An Object Detection (OD) fine-tuning pipeline.
- Cross-platform sweeps over multiple vision encoders with automatic result logging.

<a name=repository-layout></a>
## ⚙️ Repository Layout

| Folder | Description |
| --- | --- |
| [`musvit/`](musvit/) | Central launcher (`musvit` CLI): discovers and runs the experiments. |
| [`experiments/full_page_omr/`](experiments/full_page_omr/) | Fine-tune MuSViT for full-page Optical Music Recognition (OMR). |
| [`experiments/embeddings_test/`](experiments/embeddings_test/) | Compare vision-encoder embedding distances against transcription distances. |
| [`experiments/object_detection/`](experiments/object_detection/) | Fine-tune MuSViT for Object Detection (OD). |

<a name=installation></a>
## 🔧 Installation

The whole monorepo shares a single [uv](https://docs.astral.sh/uv/) project defined at the repository root ([`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock)). Install the environment once from the root:

```bash
uv sync
```

This also installs the `musvit` console command into the project environment.

<a name=environment-variables></a>
## 🔑 Environment Variables

Experiments authenticate against [Weights & Biases](https://wandb.ai) and the [Hugging Face Hub](https://huggingface.co). These credentials are shared across the whole monorepo through a single `.env` file at the repository root.

1. Copy the template:
```bash
   cp .env.example .env
```
2. Fill in your keys in `.env`:
```dotenv
   WANDB_API_KEY=...
   HUGGINGFACE_KEY=...
```

`.env` is git-ignored; only `.env.example` is committed. The `musvit` launcher loads this `.env` once at startup and automatically bridges `HUGGINGFACE_KEY` to `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, so every experiment authenticates from the same key regardless of which variable it reads.

<a name=running-experiments></a>
## 🚀 Running Experiments

All experiments are launched from the repository root with `musvit`. List what is available:

```bash
uv run musvit list
```

Each subcommand maps its options directly onto the experiment's Python entrypoint, and `--help` is generated automatically:

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

Checkpoints and W&B logs are written inside `experiments/full_page_omr/` (`weights/`, `wandb_logs/`).

### Object Detection (Faster R-CNN fine-tuning)

```bash
uv run musvit object-detection \
  --model musvit_base \
  --dataset deepscores \
  --yaml_path experiments/object_detection/data/deepscores.yaml \
  --out_dir outputs/musvit_frcnn/deepscores
```

Checkpoints, per-epoch metrics and validation visualizations are written inside `--out_dir`. Supports LoRA fine-tuning (`--use_lora`), resuming from a checkpoint (`--resume_from --finetune_mode`) and W&B logging (`--use_wandb`).

### Embeddings analysis (single encoder)

```bash
uv run musvit embeddings run --model facebook/dinov2-base --device 0
```

Embeddings, results and logs are written inside `experiments/embeddings_test/` (`embeddings/`, `results/`).

### Embeddings sweep (several encoders)

Cross-platform replacement for the old `script_run_experiments.sh` (works on Windows too). Runs a preset list of encoders, writing a per-model log and a `summary.tsv`:

```bash
uv run musvit embeddings sweep --models_set light
uv run musvit embeddings sweep --models_set default --device 0
uv run musvit embeddings sweep --only facebook/dinov2-base
```

Model sets: `light`, `default`, `full` (defined in [`experiments/embeddings_test/model_sets.py`](experiments/embeddings_test/model_sets.py)).

> ⚠️ **Note on GPU memory**  
> The sweep loads encoders one after another in the same process and frees GPU memory between models. If you hit out-of-memory errors with the largest sets, run the encoders individually with `musvit embeddings run`.

<a name=citations></a>
## 📖 Citations

```bibtex
@inproceedings{penarrubia2026musvit,
  title     = {MuSViT: A Foundation Vision Model for Sheet Music Representation},
  author    = {Penarrubia, Carlos and Rios-Vila, Antonio and Fuentes-Martinez, Eliseo 
              and Martinez-Sevilla, Juan C. and Castellanos, Francisco J. and 
              Alfaro-Contreras, Maria and Calvo-Zaragoza, Jorge},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

<a name=license></a>
## 📝 License

This work is under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

> ⚠️ **Disclaimer**  
> This github is under active development. Some parts of the codebase may not work as expected or could change without notice.  
> Please proceed with caution and check back for updates.
