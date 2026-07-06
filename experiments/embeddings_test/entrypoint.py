"""Public entrypoints for the embeddings experiment.

- :func:`run`   - analyse a single vision encoder.
- :func:`sweep` - analyse several encoders in a row (cross-platform replacement
  for the former ``script_run_experiments.sh``), writing a per-model log and a
  ``summary.tsv``.

Both are called in-process by the ``musvit`` launcher and anchor their output
(``embeddings/``, ``results/``, ``logs/``) inside this experiment folder.
"""

import argparse
import gc
import shutil
import subprocess
import sys
import traceback
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path

from .model_sets import MODEL_SETS
from .run_embedding_analysis import run_analysis

PACKAGE_DIR = Path(__file__).resolve().parent


def _make_config(model, device, dataset, split, batch_size, n_neighbors,
                 embeddings_dir, results_dir, no_save_gt):
    return argparse.Namespace(
        weights_encoder=model,
        device=device,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        n_neighbors=n_neighbors,
        embeddings_dir=embeddings_dir,
        results_dir=results_dir,
        no_save_gt=no_save_gt,
    )


def run(model: str, device: int = 0, dataset: str = "PRAIG/polish-scores",
        split: str = "train", batch_size: int = 1, n_neighbors: int = 1,
        embeddings_dir: str = "embeddings", results_dir: str = "results",
        no_save_gt: bool = False):
    """Compute embeddings for a single encoder and correlate distances."""
    config = _make_config(model, device, dataset, split, batch_size,
                          n_neighbors, embeddings_dir, results_dir, no_save_gt)
    with chdir(PACKAGE_DIR):
        run_analysis(config)


class _Tee:
    """Write to several text streams at once (in-process replacement for `tee`)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _free_gpu_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _print_gpu_status():
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                 "--format=csv,noheader"],
                check=False,
            )
        except Exception:
            pass


def sweep(models_set: str = "default", only=None, device: int = 0,
          dataset: str = "PRAIG/polish-scores", split: str = "train",
          batch_size: int = 1, n_neighbors: int = 1,
          embeddings_dir: str = "embeddings", results_dir: str = "results",
          log_dir: str = "logs", no_save_gt: bool = False,
          stop_on_error: bool = False) -> None:
    """Run the embedding analysis for several encoders.

    Args:
        models_set: One of "light", "default", "full" (ignored when --only is given).
        only: A single model id or a list/tuple of ids to run instead of a preset.
        stop_on_error: Stop at the first failing encoder (default: continue).

    Exits with a non-zero status if a model fails while stop_on_error is set.
    """
    if only:
        models = [only] if isinstance(only, str) else list(only)
    elif models_set in MODEL_SETS:
        models = MODEL_SETS[models_set]
    else:
        valid = ", ".join(MODEL_SETS)
        raise SystemExit(f"Unknown --models_set '{models_set}'. Valid values: {valid}")

    safe_dataset = dataset.replace("/", "_")
    real_out, real_err = sys.stdout, sys.stderr
    failures = 0

    with chdir(PACKAGE_DIR):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        summary_file = log_path / "summary.tsv"
        with open(summary_file, "w", encoding="utf-8") as summary:
            summary.write("model\tstatus\tlog_file\n")

        print(f"Running {len(models)} encoder(s)")
        print(f"Dataset: {dataset} | Split: {split} | Device: {device} | Batch size: {batch_size}")
        print(f"Embeddings: {embeddings_dir} | Results: {results_dir} | Logs: {log_dir}\n")

        for model in models:
            safe_model = model.replace("/", "_")
            log_file = log_path / f"{safe_model}__{safe_dataset}.log"

            print("=" * 60)
            print(f"Encoder: {model}")
            print(f"Log: {log_file}")
            print("=" * 60)

            status = "OK"
            config = _make_config(model, device, dataset, split, batch_size,
                                  n_neighbors, embeddings_dir, results_dir, no_save_gt)
            with open(log_file, "w", encoding="utf-8") as fh:
                tee_out, tee_err = _Tee(real_out, fh), _Tee(real_err, fh)
                with redirect_stdout(tee_out), redirect_stderr(tee_err):
                    try:
                        run_analysis(config)
                    except Exception:
                        status = "FAILED"
                        traceback.print_exc()

            with open(summary_file, "a", encoding="utf-8") as summary:
                summary.write(f"{model}\t{status}\t{log_file}\n")

            if status == "OK":
                print(f"OK: {model}")
            else:
                failures += 1
                print(f"FAILED: {model}", file=real_err)

            _free_gpu_memory()
            _print_gpu_status()
            print()

            if status != "OK" and stop_on_error:
                print("Stopping because stop_on_error is enabled.", file=real_err)
                break

        print(f"Finished. Summary written to: {summary_file} ({failures} failed)")

    if failures and stop_on_error:
        raise SystemExit(1)
