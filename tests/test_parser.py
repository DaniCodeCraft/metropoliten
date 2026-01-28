"""Tests for Vehicle OCR parser."""

import pytest
from pathlib import Path

from vehicle_ocr import VehicleParser, Settings
from vehicle_ocr.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Tests for image preprocessing."""

    def test_normalize_to_cyrillic(self):
        """Test Latin to Cyrillic conversion."""
        preprocessor = ImagePreprocessor()

        # Test basic conversions
        assert preprocessor.normalize_to_cyrillic("A") == "А"
        assert preprocessor.normalize_to_cyrillic("B") == "В"
        assert preprocessor.normalize_to_cyrillic("O") == "О"

        # Test mixed string
        result = preprocessor.normalize_to_cyrillic("B883BO799")
        assert "В" in result  # Cyrillic В
        assert "О" in result  # Cyrillic О


class TestSettings:
    """Tests for configuration."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.tesseract_lang == "rus+eng"
        assert settings.vin_length == 17
        assert 0 < settings.reg_number_region_start < settings.reg_number_region_end < 1

    def test_custom_settings(self):
        """Test custom settings."""
        settings = Settings(
            tesseract_lang="eng",
            clahe_clip_limit=3.0
        )

        assert settings.tesseract_lang == "eng"
        assert settings.clahe_clip_limit == 3.0


class TestVehicleParser:
    """Tests for main parser functionality."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return VehicleParser()

    def test_reg_number_pattern(self, parser):
        """Test registration number regex pattern."""
        # Valid patterns
        assert parser.REG_NUMBER_PATTERN.match("А123АА77")
        assert parser.REG_NUMBER_PATTERN.match("В883ВО799")
        assert parser.REG_NUMBER_PATTERN.match("О883ВО799")

        # Invalid patterns
        assert not parser.REG_NUMBER_PATTERN.match("12345")
        assert not parser.REG_NUMBER_PATTERN.match("АААААА")

    def test_vin_pattern(self, parser):
        """Test VIN regex pattern."""
        # Valid VIN
        assert parser.VIN_PATTERN.match("WP1ZZZ9PZ9LA42290")
        assert parser.VIN_PATTERN.match("WVWZZZ3CZWE123456")

        # Invalid (contains I, O, Q)
        assert not parser.VIN_PATTERN.match("WP1ZZZ9PZOLA42290")  # Contains O
        assert not parser.VIN_PATTERN.match("WP1ZZZ9PZ9IA42290")  # Contains I


class TestIntegration:
    """Integration tests with actual images."""

    @pytest.fixture
    def test_images_dir(self):
        """Path to test images."""
        return Path(__file__).parent.parent / "data" / "input"

    def test_process_directory(self, test_images_dir):
        """Test processing a directory of images."""
        if not test_images_dir.exists():
            pytest.skip("Test images not found")

        parser = VehicleParser()
        results = parser.process_directory(test_images_dir)

        assert len(results) > 0

        # At least some VINs should be extracted
        vins_found = sum(1 for r in results if r.vin)
        assert vins_found > 0, "No VINs extracted from test images"
