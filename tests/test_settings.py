import json
import tempfile
import unittest
from pathlib import Path

from ai_human_detected.settings import (
    AppSettings,
    DEFAULT_SETTINGS,
    load_settings,
    save_settings,
)


class SettingsTests(unittest.TestCase):
    def test_load_returns_defaults_when_file_does_not_exist(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "settings.json"

            self.assertEqual(load_settings(path), DEFAULT_SETTINGS)

    def test_save_and_load_round_trip(self):
        settings = AppSettings(
            camera_index=2,
            model_path="custom.pt",
            confidence=0.6,
            image_size=416,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "settings.json"

            save_settings(settings, path)

            self.assertEqual(load_settings(path), settings)

    def test_invalid_settings_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "settings.json"
            path.write_text(json.dumps({"confidence": 2}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Confidence"):
                load_settings(path)

    def test_malformed_json_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "settings.json"
            path.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid settings file"):
                load_settings(path)


if __name__ == "__main__":
    unittest.main()
