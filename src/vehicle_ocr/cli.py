"""Интерфейс командной строки для Vehicle OCR."""

import argparse
import sys
from pathlib import Path

from vehicle_ocr import VehicleParser, Settings


def main():
    """Главная точка входа CLI."""
    parser = argparse.ArgumentParser(
        description="Извлечение данных из свидетельств о регистрации ТС (СТС)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("data/input"),
        help="Директория с изображениями"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/output/results.json"),
        help="Путь к выходному JSON файлу"
    )

    parser.add_argument(
        "--lang",
        default="eng",
        help="Язык(и) Tesseract"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Тихий режим (без вывода прогресса)"
    )

    args = parser.parse_args()

    # Проверка входной директории
    if not args.input.exists():
        print(f"Ошибка: директория не найдена: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Настройка
    settings = Settings(
        input_dir=args.input,
        output_file=args.output,
        tesseract_lang=args.lang
    )

    if not args.quiet:
        print("=" * 60)
        print("  VEHICLE OCR PIPELINE")
        print("=" * 60)

    # Обработка
    ocr_parser = VehicleParser(settings)

    image_count = len(list(args.input.glob("*.[jJpP][pPnN][gG]")) +
                      list(args.input.glob("*.[jJ][pP][eE][gG]")))

    if not args.quiet:
        print(f"\nНайдено изображений: {image_count} в {args.input}\n")

    results = ocr_parser.process_directory(args.input)

    # Вывод результатов
    for result in results:
        if not args.quiet:
            print(f"{'=' * 50}")
            print(f"Файл: {result.file}")
            print(f"  Гос.номер:    {result.reg_number or 'не найден'}")
            print(f"  VIN:          {result.vin or 'не найден'}")
            print(f"  Номер кузова: {result.body_number or 'не найден'}")

    # Сохранение
    output_data = ocr_parser.save_results(results, args.output)

    if not args.quiet:
        print(f"\n{'=' * 60}")
        print("ИТОГО")
        print("=" * 60)
        stats = output_data["statistics"]
        total = output_data["total_processed"]
        print(f"  Обработано:     {total}")
        print(f"  Гос.номеров:    {stats['reg_numbers_found']}/{total}")
        print(f"  VIN:            {stats['vins_found']}/{total}")
        print(f"  Номеров кузова: {stats['body_numbers_found']}/{total}")
        print(f"\nРезультаты сохранены: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
