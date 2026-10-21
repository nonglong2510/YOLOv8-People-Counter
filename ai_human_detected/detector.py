from typing import Any

import cv2
from ultralytics import YOLO


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        image_size: int = 640,
        model: Any = None,
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("Image size must be greater than zero.")

        self.model = model if model is not None else YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def process_frame(self, frame: Any) -> tuple[Any, int]:
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        result = results[0]
        count = len(result.boxes) if result.boxes is not None else 0
        annotated_frame = result.plot()
        cv2.putText(
            annotated_frame,
            f"People: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated_frame, count

    def run_webcam(
        self,
        camera_index: int = 0,
        window_name: str = "YOLOv8 Human Detection",
        display: bool = True,
    ) -> int:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open webcam device index: {camera_index}")

        total_frames = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                annotated_frame, count = self.process_frame(frame)
                total_frames += 1

                if display:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if not display:
                    print(f"Frame {total_frames}: People = {count}")
        finally:
            capture.release()
            if display:
                cv2.destroyAllWindows()

        if total_frames == 0:
            raise ValueError("No frames were received from the webcam.")

        return total_frames
