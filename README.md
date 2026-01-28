# Vehicle OCR Pipeline

OCR-пайплайн для извлечения данных из свидетельств о регистрации ТС (СТС):
- Государственный регистрационный номер
- VIN (идентификационный номер)
- Номер кузова (кабины, прицепа)

## Быстрый старт

### Docker (рекомендуется)

```bash
# Запуск одной командой
docker-compose up --build

# Результаты в data/output/results.json
```

### Локальная установка

**Требования:** Python 3.10+, Tesseract OCR

**macOS:**
```bash
# 1. Установить Tesseract
brew install tesseract tesseract-lang

# 2. Установить Poetry
pip install poetry

# 3. Установить зависимости
poetry install

# 4. Запуск
poetry run vehicle-ocr -i data/input -o data/output/results.json
```

**Ubuntu/Debian:**
```bash
# 1. Установить Tesseract
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

# 2. Установить Poetry
pip install poetry

# 3. Установить зависимости
poetry install

# 4. Запуск
poetry run vehicle-ocr -i data/input -o data/output/results.json
```

## Использование

### CLI

```bash
# Базовый запуск
vehicle-ocr --input ./images --output ./results.json

# С параметрами
vehicle-ocr -i ./images -o ./results.json --lang eng

# Тихий режим
vehicle-ocr -i ./images -o ./results.json -q
```

### Python API

```python
from vehicle_ocr import VehicleParser, Settings

# С настройками по умолчанию
parser = VehicleParser()

# Или с кастомными настройками
settings = Settings(
    tesseract_lang="eng",
    reg_number_region_start=0.10,
    reg_number_region_end=0.35
)
parser = VehicleParser(settings)

# Обработка одного файла
result = parser.parse_document("document.jpg")
print(result.reg_number)   # В883ВО799
print(result.vin)          # WP1ZZZ9PZ9LA42290
print(result.body_number)  # WP1ZZZ9PZ9LA42290

# Обработка директории
results = parser.process_directory("./images")
parser.save_results(results, "results.json")
```

## Конфигурация

Настройки через переменные окружения (префикс `VOCR_`):

```bash
export VOCR_TESSERACT_LANG=eng
export VOCR_CLAHE_CLIP_LIMIT=2.5
export VOCR_MIN_IMAGE_HEIGHT=800
```

Или файл `.env`:

```env
VOCR_TESSERACT_LANG=eng
VOCR_CLAHE_CLIP_LIMIT=2.5
```

### Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `tesseract_lang` | `eng` | Языки OCR |
| `tesseract_psm` | `6` | Page segmentation mode |
| `clahe_clip_limit` | `2.0` | CLAHE порог контраста |
| `min_image_height` | `800` | Мин. высота для upscale |
| `reg_number_region_start` | `0.15` | Начало региона гос.номера |
| `reg_number_region_end` | `0.30` | Конец региона гос.номера |
| `vin_region_start` | `0.25` | Начало региона VIN |
| `vin_region_end` | `0.60` | Конец региона VIN |
| `body_number_region_start` | `0.52` | Начало региона номера кузова |
| `body_number_region_end` | `0.68` | Конец региона номера кузова |

## Выходной формат

```json
{
  "documents": [
    {
      "file": "test.jpg",
      "reg_number": "В883ВО799",
      "vin": "WP1ZZZ9PZ9LA42290",
      "body_number": "WP1ZZZ9PZ9LA42290"
    }
  ],
  "total_processed": 1,
  "statistics": {
    "reg_numbers_found": 1,
    "vins_found": 1,
    "body_numbers_found": 1
  },
  "version": "1.0.0"
}
```

## Структура проекта

```
vehicle-ocr/
├── pyproject.toml          # Poetry конфигурация
├── Dockerfile
├── docker-compose.yml
├── src/
│   └── vehicle_ocr/
│       ├── __init__.py     # Экспорт публичного API
│       ├── cli.py          # CLI интерфейс
│       ├── config.py       # Настройки (Pydantic)
│       ├── parser.py       # Основной парсер
│       └── preprocessor.py # Предобработка изображений
├── tests/
│   └── test_parser.py
└── data/
    ├── input/              # Входные изображения
    └── output/             # Результаты
```

## Архитектура

```
Изображение → Извлечение региона → Предобработка → OCR → Постобработка → JSON
                     │                   │           │          │
                По % высоты         CLAHE +      Tesseract   Нормализация
                                   Upscaling    (PSM 3,6)    + Скоринг
```

### Ключевые техники

1. **CLAHE** — адаптивное улучшение контраста
2. **ROI extraction** — извлечение регионов по % высоты документа
3. **Multi-strategy consensus** — 36 комбинаций параметров, выбор по частоте
4. **Latin→Cyrillic нормализация** — для гос.номеров (O→О, B→В и т.д.)
5. **VIN scoring** — оценка по структуре ISO 3779

## Тестирование

```bash
# Запуск тестов
poetry run pytest

# С coverage
poetry run pytest --cov=vehicle_ocr
```

## Технологии

- **Python** 3.10+
- **OpenCV** — обработка изображений
- **Tesseract OCR** — распознавание текста
- **Pydantic** — валидация и настройки
- **Poetry** — управление зависимостями
- **Docker** — контейнеризация
