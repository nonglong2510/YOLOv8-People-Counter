import argparse

from ai_human_detected.app import run_webcam
from ai_human_detected.detector import PersonDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time human detection and counting from a webcam using YOLOv8."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the webcam device to use (default: 0).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--window-name",
        default="YOLOv8 Human Detection",
        help="Window title for the live webcam preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = PersonDetector(
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
    )
    frames_processed = run_webcam(
        detector,
        camera_index=args.camera_index,
        window_name=args.window_name,
        display=True,
    )
    print(f"Processed {frames_processed} webcam frames.")
    print("Press 'q' in the video window to exit.")


if __name__ == "__main__":
    main()
