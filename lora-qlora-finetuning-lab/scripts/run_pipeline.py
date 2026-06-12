"""Convenience entrypoint.

Usage:
    uv run python scripts/run_pipeline.py
"""

from __future__ import annotations

import json

from lora_qlora_lab.config import get_settings
from lora_qlora_lab.logging_utils import configure_logging
from lora_qlora_lab.pipeline import run_pipeline


if __name__ == "__main__":
    configure_logging()
    settings = get_settings()
    result = run_pipeline(settings)
    print(json.dumps(result["deltas"], indent=2))
