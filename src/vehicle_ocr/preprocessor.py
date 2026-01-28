"""Предобработка изображений для OCR."""

import cv2
import numpy as np
from vehicle_ocr.config import Settings


class ImagePreprocessor:
    """Обработка изображений для улучшения качества OCR."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

        # Маппинг латиницы в кириллицу для российских номеров
        self.char_map = {
            'O': 'О', 'o': 'о', '0': 'О',
            'B': 'В', 'b': 'в',
            'A': 'А', 'a': 'а',
            'E': 'Е', 'e': 'е',
            'K': 'К', 'k': 'к',
            'M': 'М', 'm': 'м',
            'H': 'Н', 'h': 'н',
            'P': 'Р', 'p': 'р',
            'C': 'С', 'c': 'с',
            'T': 'Т', 't': 'т',
            'Y': 'У', 'y': 'у',
            'X': 'Х', 'x': 'х',
        }

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Улучшение контраста через CLAHE."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(
            clipLimit=self.settings.clahe_clip_limit,
            tileGridSize=(self.settings.clahe_tile_size, self.settings.clahe_tile_size)
        )
        return clahe.apply(gray)

    def extract_region(
        self,
        image: np.ndarray,
        start_ratio: float,
        end_ratio: float
    ) -> np.ndarray:
        """Извлечение горизонтального региона по проценту высоты."""
        height = image.shape[0]
        start_y = int(height * start_ratio)
        end_y = int(height * end_ratio)
        return image[start_y:end_y, :]

    def normalize_to_cyrillic(self, text: str) -> str:
        """Конвертация латинских символов в кириллические эквиваленты."""
        for lat, cyr in self.char_map.items():
            text = text.replace(lat, cyr)
        return text

    def upscale_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Увеличение маленьких изображений для лучшего OCR."""
        height = image.shape[0]
        min_height = self.settings.min_image_height

        if height >= min_height:
            return image

        scale = min_height / height
        new_width = int(image.shape[1] * scale)
        new_height = int(height * scale)

        interpolation_methods = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }
        method = interpolation_methods.get(
            self.settings.upscale_method,
            cv2.INTER_CUBIC
        )

        return cv2.resize(image, (new_width, new_height), interpolation=method)

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Полный пайплайн предобработки для OCR."""
        upscaled = self.upscale_if_needed(image)
        enhanced = self.enhance_contrast(upscaled)
        return enhanced
