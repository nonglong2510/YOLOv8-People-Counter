from ai_human_detected.detector import PersonDetector


def run_webcam(
    detector: PersonDetector,
    camera_index: int = 0,
    window_name: str = "YOLOv8 Human Detection",
    display: bool = True,
) -> int:
    return detector.run_webcam(
        camera_index=camera_index,
        window_name=window_name,
        display=display,
    )
