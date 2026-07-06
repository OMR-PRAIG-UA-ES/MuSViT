"""Root entrypoint for the MuSViT monorepo CLI (``musvit``).

Dispatches each subcommand to the in-process entrypoint function of the
corresponding experiment under ``experiments/``. Powered by Fire, so CLI
arguments map directly onto each entrypoint's parameters and ``--help`` is
generated automatically.
"""

import sys

from . import env

# (command, location, description) for `musvit list`.
_EXPERIMENTS = [
    ("full-page-omr <config_path> <experiment_name>", "experiments/full_page_omr",
     "Fine-tune MuSViT for full-page OMR."),
    ("embeddings run --model <id>", "experiments/embeddings_test",
     "Embedding-distance vs transcription-distance analysis (single encoder)."),
    ("embeddings sweep --models_set <light|default|full>", "experiments/embeddings_test",
     "Same analysis over a preset list of encoders (cross-platform sweep)."),
]


def list_experiments():
    """List the experiments runnable from this CLI."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="MuSViT experiments")
        table.add_column("command", style="bold cyan", no_wrap=True)
        table.add_column("location", style="dim")
        table.add_column("description")
        for command, location, desc in _EXPERIMENTS:
            table.add_row(f"musvit {command}", location, desc)
        Console().print(table)
    except Exception:
        # Fallback if rich is unavailable for any reason.
        print("MuSViT experiments:")
        for command, location, desc in _EXPERIMENTS:
            print(f"  musvit {command}\n      {desc}  ({location})")


def main():
    """Console-script entrypoint (``musvit``)."""
    env.setup()

    argv = sys.argv[1:]
    # Fast path: discovery must not pay the cost of importing the heavy
    # experiment modules (torch, transformers, ...).
    if not argv or argv[0] == "list":
        return list_experiments()

    from experiments.embeddings_test.entrypoint import run as embeddings_run
    from experiments.embeddings_test.entrypoint import sweep as embeddings_sweep
    from experiments.full_page_omr.entrypoint import run as full_page_omr_run
    from fire import Fire

    Fire({
        "list": list_experiments,
        "full-page-omr": full_page_omr_run,
        "embeddings": {"run": embeddings_run, "sweep": embeddings_sweep},
    })


if __name__ == "__main__":
    main()
