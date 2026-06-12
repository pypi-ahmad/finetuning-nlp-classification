"""CLI interface for LoRA/QLoRA lab."""

from __future__ import annotations

import json
import subprocess
import sys

import typer

from lora_qlora_lab.config import get_settings
from lora_qlora_lab.logging_utils import configure_logging
from lora_qlora_lab.pipeline import run_pipeline

app = typer.Typer(help="LoRA vs QLoRA Fine-Tuning Lab")


@app.callback()
def callback() -> None:
    configure_logging()


@app.command("run-all")
def run_all_cmd() -> None:
    settings = get_settings()
    summary = run_pipeline(settings)
    typer.echo(json.dumps(summary["deltas"], indent=2))


@app.command("serve-app")
def serve_app_cmd(port: int = 8502) -> None:
    settings = get_settings()
    app_path = settings.resolve_path(settings.project_root / "app" / "streamlit_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
