import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppSettings:
    camera_index: int = 0
    model_path: str = "yolov8n.pt"
    confidence: float = 0.25
    image_size: int = 640


DEFAULT_SETTINGS = AppSettings()


def settings_file_path() -> Path:
    if os.name == "nt":
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "AIHumanDetected" / "settings.json"
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base_path = Path(config_home) if config_home else Path.home() / ".config"
    return base_path / "ai-human-detected" / "settings.json"


def validate_settings(settings: AppSettings) -> AppSettings:
    if type(settings.camera_index) is not int or settings.camera_index < 0:
        raise ValueError("Camera index must be a non-negative integer.")
    if (
        isinstance(settings.confidence, bool)
        or not isinstance(settings.confidence, (int, float))
        or not 0 <= settings.confidence <= 1
    ):
        raise ValueError("Confidence must be between 0 and 1.")
    if type(settings.image_size) is not int or settings.image_size <= 0:
        raise ValueError("Image size must be greater than zero.")
    if not isinstance(settings.model_path, str) or not settings.model_path.strip():
        raise ValueError("Model path cannot be empty.")
    return settings


def load_settings(path: Path | None = None) -> AppSettings:
    settings_path = path or settings_file_path()
    if not settings_path.exists():
        return DEFAULT_SETTINGS

    try:
        data: Any = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid settings file: {settings_path}") from error

    if not isinstance(data, dict):
        raise ValueError(f"Settings file must contain a JSON object: {settings_path}")

    try:
        settings = replace(DEFAULT_SETTINGS, **data)
    except TypeError as error:
        raise ValueError(f"Unsupported setting in file: {settings_path}") from error
    return validate_settings(settings)


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    settings_path = path or settings_file_path()
    validated = validate_settings(settings)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = settings_path.with_suffix(f"{settings_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(asdict(validated), indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(settings_path)
    return settings_path
