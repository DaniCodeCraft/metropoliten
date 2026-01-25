"""
Production OCR Pipeline
Использует финальную версию v3
"""

import argparse
from vehicle_parser_v3 import FinalVehicleParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/input')
    parser.add_argument('--output', default='data/output/results.json')
    args = parser.parse_args()
    
    print("="*70)
    print("  VEHICLE OCR PIPELINE - PRODUCTION VERSION")
    print("="*70)
    
    pipeline = FinalVehicleParser()
    results = pipeline.process_directory(args.input)
    
    # Сохранение
    import json
    from pathlib import Path
    
    output_data = {
        'documents': results,
        'total_processed': len(results),
        'version': 'v3-production'
    }
    
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Результаты: {args.output}")


if __name__ == '__main__':
    main()