"""Vehicle OCR - Extract data from Russian vehicle registration certificates."""

__version__ = "1.0.0"

from vehicle_ocr.parser import VehicleParser
from vehicle_ocr.config import Settings

__all__ = ["VehicleParser", "Settings"]
