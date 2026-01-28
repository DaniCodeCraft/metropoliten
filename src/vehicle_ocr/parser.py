"""Парсер документов ТС — извлечение данных из СТС."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from vehicle_ocr.config import Settings, get_settings
from vehicle_ocr.preprocessor import ImagePreprocessor


class ExtractionResult:
    """Контейнер результатов извлечения."""

    def __init__(
        self,
        file: str,
        reg_number: Optional[str] = None,
        vin: Optional[str] = None,
        body_number: Optional[str] = None
    ):
        self.file = file
        self.reg_number = reg_number
        self.vin = vin
        self.body_number = body_number

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "reg_number": self.reg_number,
            "vin": self.vin,
            "body_number": self.body_number
        }


class VehicleParser:
    """Основной парсер документов регистрации ТС (СТС)."""

    # Паттерн российского гос.номера: Х000ХХ00 или Х000ХХ000
    REG_NUMBER_PATTERN = re.compile(r'[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}')

    # Паттерн VIN: 17 буквенно-цифровых символов (без I, O, Q по ISO 3779)
    VIN_PATTERN = re.compile(r'[A-HJ-NPR-Z0-9]{17}')

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.preprocessor = ImagePreprocessor(self.settings)

    def _run_ocr(self, image: np.ndarray, psm: int | None = None) -> str:
        """Запуск Tesseract OCR на изображении."""
        psm = psm or self.settings.tesseract_psm
        config = f'--psm {psm}'

        pil_image = Image.fromarray(image)
        text = pytesseract.image_to_string(
            pil_image,
            lang=self.settings.tesseract_lang,
            config=config
        )
        return text

    def extract_reg_number(self, image: np.ndarray) -> Optional[str]:
        """Извлечение гос.номера из изображения."""
        height = image.shape[0]

        # Извлекаем регион где обычно находится гос.номер
        region = self.preprocessor.extract_region(
            image,
            self.settings.reg_number_region_start,
            self.settings.reg_number_region_end
        )

        # Для маленьких изображений увеличиваем регион
        if height < self.settings.min_image_height:
            region = self.preprocessor.upscale_if_needed(region)

        processed = self.preprocessor.enhance_contrast(region)
        text = self._run_ocr(processed)

        # Нормализуем латиницу в кириллицу
        normalized = self.preprocessor.normalize_to_cyrillic(text.upper())

        # Ищем совпадения по паттерну
        matches = self.REG_NUMBER_PATTERN.findall(normalized)
        if matches:
            return matches[0].replace(' ', '').replace('\n', '')

        # Запасной вариант: поиск построчно
        for line in normalized.split('\n'):
            cleaned = re.sub(r'[^А-Я0-9]', '', line)

            if self.settings.min_reg_number_length <= len(cleaned) <= self.settings.max_reg_number_length:
                has_letters = bool(re.search(r'[А-Я]', cleaned))
                has_digits = bool(re.search(r'\d', cleaned))

                if has_letters and has_digits:
                    letter_count = sum(1 for c in cleaned if c.isalpha())
                    digit_count = sum(1 for c in cleaned if c.isdigit())

                    if 3 <= letter_count <= 4 and 4 <= digit_count <= 6:
                        return cleaned

        return None

    def extract_vin(self, image: np.ndarray) -> Optional[str]:
        """Извлечение VIN из документа."""
        # Извлекаем регион где обычно находится VIN
        region = self.preprocessor.extract_region(
            image,
            self.settings.vin_region_start,
            self.settings.vin_region_end
        )

        # Всегда увеличиваем регион VIN для стабильности OCR
        region = self.preprocessor.upscale_if_needed(region)
        processed = self.preprocessor.enhance_contrast(region)

        candidates = []

        # Пробуем несколько PSM режимов и собираем кандидатов
        for psm in [3, 6]:
            try:
                text = self._run_ocr(processed, psm=psm).upper()
                cleaned = re.sub(r'[^A-Z0-9]', '', text)

                # Ищем все возможные 17-символьные последовательности VIN
                for i in range(len(cleaned) - 16):
                    candidate = cleaned[i:i + 17]
                    if self.VIN_PATTERN.match(candidate):
                        candidates.append(candidate)

            except Exception:
                pass

        if not candidates:
            return None

        # Оценка кандидатов на основе структуры VIN
        def vin_score(vin: str) -> int:
            score = 0
            # VIN должен начинаться с буквы (WMI код страны)
            if vin[0].isalpha():
                score += 10
            # Позиции 10-17 (серийный номер) должны содержать цифры
            serial = vin[9:17]
            score += sum(1 for c in serial if c.isdigit())
            return score

        # Возвращаем кандидата с лучшим скором
        return max(candidates, key=vin_score)

    def _extract_with_multiple_strategies(
        self,
        image: np.ndarray,
        region_start: float,
        region_end: float
    ) -> Optional[str]:
        """Извлечение номера с использованием множества стратегий предобработки.

        Использует консенсус: пробует 36 комбинаций параметров и выбирает
        кандидата, который встречается чаще всего.
        """
        region = self.preprocessor.extract_region(image, region_start, region_end)

        all_candidates = []

        # Вариации стратегий
        clahe_params = [(2.0, 8), (3.0, 8), (2.0, 4), (1.5, 16)]
        scale_factors = [1.0, 1.5, 2.0] if region.shape[0] >= 200 else [2.0, 2.5, 3.0]
        psm_modes = [3, 6, 11]

        for clip_limit, tile_size in clahe_params:
            for scale in scale_factors:
                # Масштабирование
                if scale != 1.0:
                    new_h = int(region.shape[0] * scale)
                    new_w = int(region.shape[1] * scale)
                    scaled = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                else:
                    scaled = region

                # Конвертация в grayscale
                if len(scaled.shape) == 3:
                    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                else:
                    gray = scaled

                # Применяем CLAHE с текущими параметрами
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                enhanced = clahe.apply(gray)

                # Пробуем разные PSM режимы
                for psm in psm_modes:
                    try:
                        text = pytesseract.image_to_string(
                            Image.fromarray(enhanced),
                            lang=self.settings.tesseract_lang,
                            config=f'--psm {psm}'
                        ).upper()

                        cleaned = re.sub(r'[^A-Z0-9]', '', text)

                        for i in range(len(cleaned) - 16):
                            candidate = cleaned[i:i + 17]
                            if self.VIN_PATTERN.match(candidate):
                                all_candidates.append(candidate)
                    except Exception:
                        pass

        if not all_candidates:
            return None

        # Подсчёт частоты — консенсусный подход
        counter = Counter(all_candidates)

        # Скор: частота + качество структуры
        def combined_score(item):
            candidate, count = item
            struct_score = 0
            if candidate[0].isalpha():
                struct_score += 5
            struct_score += sum(1 for c in candidate[9:17] if c.isdigit())
            return count * 10 + struct_score

        best = max(counter.items(), key=combined_score)
        return best[0]

    def extract_body_number(self, image: np.ndarray) -> Optional[str]:
        """Извлечение номера кузова из документа."""
        return self._extract_with_multiple_strategies(
            image,
            self.settings.body_number_region_start,
            self.settings.body_number_region_end
        )

    def parse_document(self, image_path: str | Path) -> ExtractionResult:
        """Обработка одного документа."""
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        reg_number = self.extract_reg_number(image)
        vin = self.extract_vin(image)
        body_number = self.extract_body_number(image)

        return ExtractionResult(
            file=image_path.name,
            reg_number=reg_number,
            vin=vin,
            body_number=body_number
        )

    def process_directory(self, directory: str | Path) -> list[ExtractionResult]:
        """Обработка всех изображений в директории."""
        directory = Path(directory)
        results = []

        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        image_files = sorted([
            f for f in directory.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        for image_file in image_files:
            try:
                result = self.parse_document(image_file)
                results.append(result)
            except Exception as e:
                print(f"Ошибка обработки {image_file.name}: {e}")
                results.append(ExtractionResult(file=image_file.name))

        return results

    def save_results(self, results: list[ExtractionResult], output_path: str | Path):
        """Сохранение результатов в JSON файл."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "documents": [r.to_dict() for r in results],
            "total_processed": len(results),
            "statistics": {
                "reg_numbers_found": sum(1 for r in results if r.reg_number),
                "vins_found": sum(1 for r in results if r.vin),
                "body_numbers_found": sum(1 for r in results if r.body_number)
            },
            "version": "1.0.0"
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_data
