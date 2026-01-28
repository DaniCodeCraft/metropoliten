"""Конфигурация приложения через Pydantic."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Настройки приложения с поддержкой переменных окружения."""

    # Пути
    input_dir: Path = Field(default=Path("data/input"))
    output_file: Path = Field(default=Path("data/output/results.json"))

    # Настройки OCR
    tesseract_lang: str = Field(default="eng")
    tesseract_psm: int = Field(default=6)

    # Предобработка изображений
    clahe_clip_limit: float = Field(default=2.0)
    clahe_tile_size: int = Field(default=8)
    min_image_height: int = Field(default=800)  # Увеличивать если меньше
    upscale_method: str = Field(default="cubic")  # linear, cubic, lanczos

    # Извлечение регионов (процент от высоты изображения)
    reg_number_region_start: float = Field(default=0.15)
    reg_number_region_end: float = Field(default=0.30)
    vin_region_start: float = Field(default=0.25)
    vin_region_end: float = Field(default=0.60)
    body_number_region_start: float = Field(default=0.52)
    body_number_region_end: float = Field(default=0.68)

    # Валидация
    vin_length: int = Field(default=17)
    min_reg_number_length: int = Field(default=8)
    max_reg_number_length: int = Field(default=10)

    model_config = {
        "env_prefix": "VOCR_",
        "env_file": ".env",
        "extra": "ignore"
    }


def get_settings() -> Settings:
    """Получить синглтон настроек."""
    return Settings()
