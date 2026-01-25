import json
import re
import pytesseract
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, List


class BasicVehicleParser:
    """Базовый парсер - простой Tesseract OCR"""
    
    def extract_text(self, image_path: str) -> str:
        """Простое извлечение текста"""
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        return text
    
    def extract_vin(self, text: str) -> Optional[str]:
        """Поиск VIN (17 символов)"""
        pattern = r'[A-HJ-NPR-Z0-9]{17}'
        matches = re.findall(pattern, text.upper())
        return matches[0] if matches else None
    
    def extract_reg_number(self, text: str) -> Optional[str]:
        """Поиск гос.номера (базовый)"""
        # Простой поиск последовательности букв и цифр
        lines = text.split('\n')
        for line in lines:
            cleaned = re.sub(r'[^A-ZА-Я0-9]', '', line.upper())
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
            'body_number': self.extract_vin(text)  # Обычно совпадает с VIN
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
    """Тестовый запуск"""
    parser = BasicVehicleParser()
    results = parser.process_directory('data/input')
    
    # Вывод результатов
    output = {
        'documents': results,
        'total_processed': len(results),
        'version': 'v1-basic-ocr'
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    # Сохранение
    with open('data/output/results_v1.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Обработано: {len(results)} документов")


if __name__ == '__main__':
    main()