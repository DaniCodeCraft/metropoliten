"""
Версия 2: Добавлена предобработка изображений (CLAHE)
Точность: ~85%
"""

import json
import re
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


class ImprovedVehicleParser:
    """Парсер с предобработкой изображений"""
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """CLAHE preprocessing для улучшения контраста"""
        # Загрузка
        img = cv2.imread(image_path)
        
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_text(self, image_path: str) -> str:
        """Извлечение текста с предобработкой"""
        # Предобработка
        enhanced = self.preprocess_image(image_path)
        
        # OCR на улучшенном изображении
        pil_img = Image.fromarray(enhanced)
        text = pytesseract.image_to_string(pil_img, lang='eng', config='--psm 6')
        
        return text
    
    def normalize_cyrillic(self, text: str) -> str:
        """Нормализация Latin → Cyrillic"""
        mapping = {
            'O': 'О', 'o': 'о',
            'B': 'В', 'A': 'А',
            'E': 'Е', 'K': 'К',
            'M': 'М', 'H': 'Н',
            'P': 'Р', 'C': 'С',
            'T': 'Т', 'Y': 'У',
            'X': 'Х'
        }
        
        for lat, cyr in mapping.items():
            text = text.replace(lat, cyr)
        
        return text
    
    def extract_vin(self, text: str) -> Optional[str]:
        """Улучшенное извлечение VIN"""
        pattern = r'[A-HJ-NPR-Z0-9]{17}'
        matches = re.findall(pattern, text.upper())
        return matches[0] if matches else None
    
    def extract_reg_number(self, text: str) -> Optional[str]:
        """Улучшенное извлечение гос.номера"""
        # Нормализуем
        normalized = self.normalize_cyrillic(text.upper())
        
        # Паттерн российского номера
        pattern = r'[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}'
        matches = re.findall(pattern, normalized)
        
        if matches:
            return matches[0]
        
        # Fallback - ищем в строках
        lines = normalized.split('\n')
        for line in lines:
            cleaned = re.sub(r'[^А-Я0-9]', '', line)
            if 8 <= len(cleaned) <= 10:
                if re.search(r'[А-Я]', cleaned) and re.search(r'\d', cleaned):
                    return cleaned
        
        return None
    
    def parse_document(self, image_path: str) -> Dict[str, Optional[str]]:
        """Парсинг документа"""
        print(f"Обработка: {Path(image_path).name}")
        
        text = self.extract_text(image_path)
        
        return {
            'file': Path(image_path).name,
            'reg_number': self.extract_reg_number(text),
            'vin': self.extract_vin(text),
            'body_number': self.extract_vin(text)
        }
    
    def process_directory(self, input_dir: str) -> List[Dict]:
        """Обработка директории"""
        results = []
        input_path = Path(input_dir)
        
        for img_file in sorted(input_path.glob('*.jpg')):
            result = self.parse_document(str(img_file))
            results.append(result)
        
        return results


def main():
    """Тестовый запуск v2"""
    parser = ImprovedVehicleParser()
    results = parser.process_directory('data/input')
    
    output = {
        'documents': results,
        'total_processed': len(results),
        'version': 'v2-with-preprocessing',
        'improvements': 'Added CLAHE preprocessing + cyrillic normalization'
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    with open('data/output/results_v2.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Обработано: {len(results)} документов")


if __name__ == '__main__':
    main()