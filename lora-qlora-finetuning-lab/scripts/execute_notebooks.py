"""Execute tutorial notebooks and persist outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "01_data_and_baselines.ipynb",
    "02_lora_finetuning_tutorial.ipynb",
    "03_qlora_finetuning_tutorial.ipynb",
    "04_comparison_and_report.ipynb",
]


def execute_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=False)
    client.execute(cwd=str(Path(".").resolve()))
    nbformat.write(nb, path)


def main() -> None:
    base = Path("notebooks")
    for name in NOTEBOOKS:
        execute_notebook(base / name)


if __name__ == "__main__":
    main()
