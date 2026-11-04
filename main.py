import argparse
from dataclasses import replace

from ai_human_detected.app import run_webcam
from ai_human_detected.detector import PersonDetector
from ai_human_detected.settings import load_settings, settings_file_path, validate_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time human detection and counting from a webcam using YOLOv8."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Index of the webcam device (overrides the saved setting).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ultralytics model weights (overrides the saved setting).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Minimum detection confidence from 0 to 1 (overrides the saved setting).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Inference image size (overrides the saved setting).",
    )
    parser.add_argument(
        "--window-name",
        default="YOLOv8 Human Detection",
        help="Window title for the live webcam preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    settings = replace(
        settings,
        camera_index=args.camera_index if args.camera_index is not None else settings.camera_index,
        model_path=args.model if args.model is not None else settings.model_path,
        confidence=args.confidence if args.confidence is not None else settings.confidence,
        image_size=args.image_size if args.image_size is not None else settings.image_size,
    )
    settings = validate_settings(settings)
    detector = PersonDetector(
        model_path=settings.model_path,
        confidence=settings.confidence,
        image_size=settings.image_size,
    )
    print("Use Space to pause/resume; press 'q' or Esc to exit.")
    frames_processed = run_webcam(
        detector,
        camera_index=settings.camera_index,
        window_name=args.window_name,
        display=True,
        settings_path=settings_file_path(),
    )
    print(f"Processed {frames_processed} webcam frames.")


if __name__ == "__main__":
    main()
