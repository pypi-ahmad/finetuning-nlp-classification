"""Configuration for LoRA/QLoRA project."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")

    dataset_name: str = "dair-ai/emotion"
    lora_model_name: str = "distilgpt2"
    qlora_model_name: str = "facebook/opt-350m"

    train_samples: int = 1800
    validation_samples: int = 300
    test_samples: int = 300
    baseline_eval_samples: int = 40
    tuned_eval_samples: int = 80

    max_length: int = 192
    lora_max_steps: int = 30
    qlora_max_steps: int = 12
    seed: int = 42

    hf_token: str | None = None

    def resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def resolved_data_dir(self) -> Path:
        return self.resolve_path(self.data_dir)

    @property
    def resolved_raw_dir(self) -> Path:
        return self.resolve_path(self.raw_dir)

    @property
    def resolved_artifacts_dir(self) -> Path:
        return self.resolve_path(self.artifacts_dir)

    @property
    def resolved_models_dir(self) -> Path:
        return self.resolve_path(self.models_dir)

    def ensure_directories(self) -> None:
        self.resolved_data_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_raw_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.resolved_artifacts_dir / "charts").mkdir(parents=True, exist_ok=True)
        (self.resolved_artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.resolved_artifacts_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.resolved_artifacts_dir / "tables").mkdir(parents=True, exist_ok=True)
        self.resolved_models_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
