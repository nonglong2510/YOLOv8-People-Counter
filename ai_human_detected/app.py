from pathlib import Path

from ai_human_detected.detector import PersonDetector
from ai_human_detected.ui import run_opencv_dashboard


def run_webcam(
    detector: PersonDetector,
    camera_index: int = 0,
    window_name: str = "YOLOv8 Human Detection",
    display: bool = True,
    settings_path: Path | None = None,
) -> int:
    if display:
        return run_opencv_dashboard(
            detector,
            camera_index=camera_index,
            window_name=window_name,
            settings_path=settings_path,
        )
    return detector.run_webcam(
        camera_index=camera_index,
        window_name=window_name,
        display=False,
    )
