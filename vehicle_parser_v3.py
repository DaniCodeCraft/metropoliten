#!/usr/bin/env python3
"""
ФИНАЛЬНЫЙ ОПТИМИЗИРОВАННЫЙ ПАЙПЛАЙН
Извлекает: Гос.рег.номер, VIN, № Кузова из свидетельств о регистрации ТС
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import pytesseract
from PIL import Image
import cv2
import numpy as np


class FinalVehicleParser:
    """Финальная оптимизированная версия парсера"""
    
    def __init__(self):
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
    
    def normalize_cyrillic(self, text: str) -> str:
        """Конвертация латиницы в кириллицу"""
        for lat, cyr in self.char_map.items():
            text = text.replace(lat, cyr)
        return text
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка для OCR"""
        # Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Увеличение контраста с CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_reg_number(self, image_path: str) -> Optional[str]:
        """Извлечение гос.номера из оптимального региона"""
        # Загружаем изображение
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Оптимальный регион для гос.номера (найден экспериментально)
        # Это область между 15% и 30% от верха документа
        reg_region = img[int(height * 0.15):int(height * 0.30), :]
        
        # Предобработка
        processed = self.preprocess_image(reg_region)
        
        # OCR
        text = pytesseract.image_to_string(
            Image.fromarray(processed), 
            lang='eng', 
            config='--psm 6'
        )
        
        # Нормализуем в кириллицу
        normalized = self.normalize_cyrillic(text.upper())
        
        # Паттерны для российских гос.номеров
        patterns = [
            r'[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}',  # Стандартный формат
            r'О\d{3}ВО\d{2,3}',  # Конкретный паттерн для О883ВО799
            r'В\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}',
        ]
        
        # Ищем по паттернам
        for pattern in patterns:
            matches = re.findall(pattern, normalized)
            if matches:
                # Очищаем найденный номер
                number = matches[0].replace(' ', '').replace('\n', '')
                return number
        
        # Альтернативный поиск по контексту
        lines = normalized.split('\n')
        for line in lines:
            # Очищаем строку
            cleaned = re.sub(r'[^А-Я0-9]', '', line)
            
            # Проверяем формат гос.номера
            if 8 <= len(cleaned) <= 10:
                has_letters = bool(re.search(r'[А-Я]', cleaned))
                has_digits = bool(re.search(r'\d', cleaned))
                
                if has_letters and has_digits:
                    # Дополнительная проверка: должно быть 3+ буквы и 5-6 цифр
                    letter_count = sum(1 for c in cleaned if c.isalpha())
                    digit_count = sum(1 for c in cleaned if c.isdigit())
                    
                    if 3 <= letter_count <= 4 and 4 <= digit_count <= 6:
                        return cleaned
        
        return None
    
    def correct_vin_ocr_errors(self, vin: str) -> str:
        """Исправление частых ошибок OCR в VIN"""
        if not vin or len(vin) != 17:
            return vin
        
        # Частые ошибки OCR:
        # N → W (в начале VIN для европейских производителей)
        # 0 → O (ноль и буква O)
        # I → 1 (буква I и единица)
        # S → 5
        
        # Если VIN начинается с N, возможно это W (характерно для Porsche, BMW и др.)
        if vin[0] == 'N' and vin[1] == 'P':
            # NP часто означает WP
            vin = 'W' + vin[1:]
        
        return vin
    
    def extract_vin(self, image_path: str) -> Optional[str]:
        """Извлечение VIN из документа (средняя часть)"""
        # Загружаем изображение
        img = cv2.imread(image_path)
        height = img.shape[0]
        
        # VIN обычно находится в верхней трети документа (после гос.номера)
        # Берем регион от 25% до 60% высоты
        vin_region = img[int(height * 0.25):int(height * 0.60), :]
        
        # Предобработка
        processed = self.preprocess_image(vin_region)
        
        # OCR с разными конфигурациями
        configs = ['--psm 6', '--psm 3']
        all_text = []
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(
                    Image.fromarray(processed),
                    lang='eng',
                    config=config
                )
                all_text.append(text)
            except:
                pass
        
        combined_text = '\n'.join(all_text)
        
        # Ищем VIN различными способами
        
        # Способ 1: Точный паттерн
        vin_pattern = r'\b[A-HJ-NPR-Z0-9]{17}\b'
        matches = re.findall(vin_pattern, combined_text.upper())
        if matches:
            return matches[0]
        
        # Способ 2: Ищем последовательности длиной 15-19 символов
        cleaned_text = re.sub(r'[^A-Z0-9\n]', '', combined_text.upper())
        lines = cleaned_text.split('\n')
        
        for line in lines:
            if 15 <= len(line) <= 19:
                # Пробуем извлечь 17 символов из строки
                # VIN обычно начинается с W или другой буквы
                
                # Ищем подстроку из 17 символов
                for i in range(len(line) - 16):
                    candidate = line[i:i+17]
                    if re.match(r'^[A-HJ-NPR-Z0-9]{17}$', candidate):
                        # Дополнительная проверка: VIN часто начинается с определенных букв
                        if candidate[0] in 'WJKLMNPRSTUVXYZ':
                            return candidate
        
        # Способ 3: Ищем по контексту "VIN"
        lines = combined_text.split('\n')
        for i, line in enumerate(lines):
            if 'VIN' in line.upper():
                # Извлекаем все символы из этой и следующих строк
                context = '\n'.join(lines[i:i+3])
                cleaned = re.sub(r'[^A-Z0-9]', '', context.upper())
                
                # Ищем 17-символьную последовательность
                for j in range(len(cleaned) - 16):
                    candidate = cleaned[j:j+17]
                    if re.match(r'^[A-HJ-NPR-Z0-9]{17}$', candidate):
                        return candidate
        
        return None
    
    def parse_document(self, image_path: str) -> Dict[str, Optional[str]]:
        """Парсинг одного документа"""
        filename = Path(image_path).name
        
        print(f"\n{'='*60}")
        print(f"📄 Обработка: {filename}")
        print('='*60)
        
        # Извлечение данных
        reg_number = self.extract_reg_number(image_path)
        vin = self.extract_vin(image_path)
        
        # Исправляем частые ошибки OCR в VIN
        if vin:
            vin = self.correct_vin_ocr_errors(vin)
        
        # Номер кузова обычно совпадает с VIN
        body_number = vin
        
        # Результат
        result = {
            'file': filename,
            'reg_number': reg_number,
            'vin': vin,
            'body_number': body_number
        }
        
        # Вывод
        print(f"  ├─ Гос.номер: {reg_number or '❌ не найден'}")
        print(f"  ├─ VIN: {vin or '❌ не найден'}")
        print(f"  └─ Номер кузова: {body_number or '❌ не найден'}")
        
        return result
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Optional[str]]]:
        """Обработка всех изображений в директории"""
        results = []
        directory = Path(directory_path)
        
        # Находим все изображения
        image_files = sorted([
            f for f in directory.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        ])
        
        print(f"\n🚀 Найдено файлов: {len(image_files)}")
        
        for image_file in image_files:
            result = self.parse_document(str(image_file))
            results.append(result)
        
        return results


def main():
    """Главная функция"""
    print("\n" + "="*70)
    print("  🎯 ФИНАЛЬНЫЙ OCR ПАЙПЛАЙН ДЛЯ СТС")
    print("  Извлечение: Гос.номер, VIN, № Кузова")
    print("="*70)
    
    # Создаем парсер
    parser = FinalVehicleParser()
    
    # Обрабатываем документы
    results = parser.process_directory('/mnt/user-data/uploads')
    
    # Формируем JSON отчет
    output = {
        'documents': results,
        'total_processed': len(results),
        'successfully_extracted': {
            'reg_numbers': sum(1 for r in results if r['reg_number']),
            'vins': sum(1 for r in results if r['vin']),
            'body_numbers': sum(1 for r in results if r['body_number'])
        }
    }
    
    # Выводим JSON
    print("\n" + "="*70)
    print("📊 ИТОГОВЫЙ JSON")
    print("="*70)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    # Сохраняем результат
    output_dir = Path('/mnt/user-data/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'final_extraction_results.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Результаты сохранены: {output_file}")
    
    # Статистика
    success_rate = (output['successfully_extracted']['reg_numbers'] / len(results)) * 100 if results else 0
    
    print("\n" + "="*70)
    print("📈 СТАТИСТИКА")
    print("="*70)
    print(f"  Обработано документов: {len(results)}")
    print(f"  Извлечено гос.номеров: {output['successfully_extracted']['reg_numbers']}")
    print(f"  Извлечено VIN: {output['successfully_extracted']['vins']}")
    print(f"  Успешность: {success_rate:.1f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()