import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ai_human_detected.detector import PersonDetector
from ai_human_detected.settings import (
    AppSettings,
    DEFAULT_SETTINGS,
    save_settings,
)


class OpenCVDashboard:
    WIDTH = 1280
    HEIGHT = 800
    SIDEBAR_WIDTH = 190
    HEADER_HEIGHT = 64
    VIDEO_RECT = (220, 130, 730, 500)
    PAUSE_BUTTON = (975, 76, 130, 40)
    STOP_BUTTON = (1115, 76, 135, 40)
    SAVE_SETTINGS_BUTTON = (975, 648, 130, 48)
    RESTORE_DEFAULTS_BUTTON = (1115, 648, 135, 48)
    CAMERA_DECREASE_BUTTON = (24, 321, 42, 34)
    CAMERA_INCREASE_BUTTON = (75, 321, 42, 34)
    CONFIDENCE_SLIDER = (997, 460, 230)
    IMAGE_SIZE_OPTIONS = (320, 416, 640)

    BACKGROUND = (245, 247, 252)
    NAVY = (34, 48, 69)
    BLUE = (205, 111, 28)
    GREEN = (70, 145, 53)
    RED = (58, 65, 225)
    TEXT = (42, 48, 59)
    MUTED = (110, 119, 132)
    WHITE = (255, 255, 255)
    BORDER = (224, 229, 238)

    def __init__(
        self,
        detector: PersonDetector,
        camera_index: int,
        settings_path: Path | None = None,
    ) -> None:
        self.detector = detector
        self.camera_index = camera_index
        self.settings_path = settings_path
        self.paused = False
        self.should_close = False
        self.requested_camera_index: int | None = None
        self.status_message: str | None = None

    @staticmethod
    def _inside(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
        x, y = point
        left, top, width, height = rect
        return left <= x <= left + width and top <= y <= top + height

    def handle_mouse(self, event: int, x: int, y: int, _flags: int, _data: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._inside((x, y), self.PAUSE_BUTTON):
            self.paused = not self.paused
        elif self._inside((x, y), self.STOP_BUTTON):
            self.should_close = True
        elif self._inside((x, y), self.CAMERA_DECREASE_BUTTON):
            self.requested_camera_index = max(0, self.camera_index - 1)
        elif self._inside((x, y), self.CAMERA_INCREASE_BUTTON):
            self.requested_camera_index = self.camera_index + 1
        elif self._inside((x, y), self.SAVE_SETTINGS_BUTTON):
            self.save_current_settings()
        elif self._inside((x, y), self.RESTORE_DEFAULTS_BUTTON):
            self.restore_defaults()
        elif abs(y - self.CONFIDENCE_SLIDER[1]) <= 12:
            start_x, slider_y, width = self.CONFIDENCE_SLIDER
            if start_x <= x <= start_x + width:
                self.detector.confidence = round((x - start_x) / width, 2)
        else:
            self._select_image_size(x, y)

    def _select_image_size(self, x: int, y: int) -> None:
        button_y = 500
        button_width = 72
        spacing = 6
        start_x = 997
        if not button_y <= y <= button_y + 26:
            return
        for index, image_size in enumerate(self.IMAGE_SIZE_OPTIONS):
            left = start_x + index * (button_width + spacing)
            if left <= x <= left + button_width:
                self.detector.image_size = image_size
                return

    def save_current_settings(self) -> None:
        settings = AppSettings(
            camera_index=self.camera_index,
            model_path=self.detector.model_path,
            confidence=self.detector.confidence,
            image_size=self.detector.image_size,
        )
        try:
            save_settings(settings, self.settings_path)
        except OSError as error:
            self.status_message = f"Unable to save settings: {error}"
            return
        self.status_message = "Settings saved for the next launch."

    def restore_defaults(self) -> None:
        self.detector.confidence = DEFAULT_SETTINGS.confidence
        self.detector.image_size = DEFAULT_SETTINGS.image_size
        self.requested_camera_index = DEFAULT_SETTINGS.camera_index
        try:
            save_settings(DEFAULT_SETTINGS, self.settings_path)
        except OSError as error:
            self.status_message = f"Unable to restore settings: {error}"
            return
        self.status_message = "Defaults saved; restart to reload the default model."

    def _text(
        self,
        image: Any,
        text: str,
        position: tuple[int, int],
        scale: float = 0.55,
        color: tuple[int, int, int] | None = None,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color or self.TEXT,
            thickness,
            cv2.LINE_AA,
        )

    def _panel(
        self,
        image: Any,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int] | None = None,
    ) -> None:
        x, y, width, height = rect
        cv2.rectangle(image, (x, y), (x + width, y + height), color or self.WHITE, -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), self.BORDER, 1)

    def _button(
        self,
        image: Any,
        rect: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        x, y, width, height = rect
        cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
        self._text(image, label, (x + 18, y + 31), 0.55, self.WHITE, 2)

    def render(
        self,
        frame: Any,
        people_count: int,
        fps: float,
        processed_frames: int,
        frame_status: str,
    ) -> Any:
        canvas = np.full((self.HEIGHT, self.WIDTH, 3), self.BACKGROUND, dtype=np.uint8)
        canvas[:, : self.SIDEBAR_WIDTH] = self.NAVY
        canvas[: self.HEADER_HEIGHT, self.SIDEBAR_WIDTH :] = self.WHITE
        cv2.line(
            canvas,
            (self.SIDEBAR_WIDTH, self.HEADER_HEIGHT),
            (self.WIDTH, self.HEADER_HEIGHT),
            self.BORDER,
            1,
        )

        self._text(canvas, "YOLOv8 Vision", (22, 38), 0.72, self.WHITE, 2)
        self._text(canvas, "Real-time people detection", (20, 58), 0.35, (205, 213, 224))
        cv2.rectangle(canvas, (12, 91), (178, 132), (53, 72, 98), -1)
        self._text(canvas, "Live monitoring", (24, 117), 0.47, self.WHITE, 1)
        self._text(canvas, "Controls", (24, 170), 0.48, (205, 213, 224), 1)
        self._text(canvas, "Space: Pause", (24, 198), 0.4, (177, 188, 202))
        self._text(canvas, "Q / Esc: Exit", (24, 222), 0.4, (177, 188, 202))
        self._text(canvas, "Camera index", (24, 281), 0.42, (177, 188, 202))
        self._text(canvas, str(self.camera_index), (24, 307), 0.65, self.WHITE, 2)
        self._button(canvas, self.CAMERA_DECREASE_BUTTON, "-", (72, 83, 102))
        self._button(canvas, self.CAMERA_INCREASE_BUTTON, "+", (72, 83, 102))
        self._text(canvas, "Switch camera", (24, 378), 0.4, (177, 188, 202))
        self._text(canvas, "OpenCV Desktop", (20, 755), 0.4, (177, 188, 202))

        self._text(canvas, "Live monitoring", (220, 39), 0.72, self.TEXT, 2)
        state_text = "PAUSED" if self.paused else "RUNNING"
        state_color = self.MUTED if self.paused else self.GREEN
        cv2.circle(canvas, (690, 33), 6, state_color, -1)
        self._text(canvas, state_text, (705, 39), 0.48, state_color, 2)
        self._text(canvas, f"Camera index: {self.camera_index}", (930, 39), 0.45, self.MUTED)

        self._panel(canvas, self.VIDEO_RECT)
        video_x, video_y, video_width, video_height = self.VIDEO_RECT
        if frame is None:
            self._text(canvas, "Waiting for camera frame...", (video_x + 170, video_y + 250), 0.65, self.MUTED)
        else:
            available_width = video_width - 12
            available_height = video_height - 12
            frame_height, frame_width = frame.shape[:2]
            scale = min(available_width / frame_width, available_height / frame_height)
            resized_width = max(1, int(frame_width * scale))
            resized_height = max(1, int(frame_height * scale))
            resized = cv2.resize(frame, (resized_width, resized_height))
            offset_x = video_x + 6 + (available_width - resized_width) // 2
            offset_y = video_y + 6 + (available_height - resized_height) // 2
            canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
            if self.paused:
                cv2.rectangle(
                    canvas,
                    (video_x + 6, video_y + 6),
                    (video_x + video_width - 6, video_y + 48),
                    self.NAVY,
                    -1,
                )
                self._text(canvas, "DETECTION PAUSED", (video_x + 20, video_y + 34), 0.5, self.WHITE, 2)

        self._text(canvas, "WEBCAM VIDEO", (video_x, video_y - 14), 0.42, self.MUTED, 1)
        self._panel(canvas, (video_x, 648, video_width, 54))
        self._text(
            canvas,
            "Current count shows people detected in this frame.",
            (video_x + 14, 671),
            0.43,
            self.TEXT,
        )
        self._text(
            canvas,
            "This is not a cumulative entrance/exit count.",
            (video_x + 14, 690),
            0.43,
            self.MUTED,
        )

        self._panel(canvas, (975, 130, 275, 125))
        self._text(canvas, "PEOPLE IN CURRENT FRAME", (997, 160), 0.42, self.MUTED, 1)
        self._text(canvas, str(people_count), (997, 222), 1.8, self.BLUE, 2)
        self._text(canvas, "people / frame", (1050, 220), 0.45, self.MUTED)

        self._panel(canvas, (975, 270, 275, 100))
        self._text(canvas, "AVERAGE PROCESSING SPEED", (997, 300), 0.38, self.MUTED, 1)
        self._text(canvas, f"{fps:.1f} FPS", (997, 345), 0.95, self.GREEN, 2)

        self._panel(canvas, (975, 385, 275, 145))
        self._text(canvas, "DETECTION SETTINGS", (997, 415), 0.45, self.MUTED, 1)
        self._text(canvas, f"Confidence: {self.detector.confidence:.2f}", (997, 445), 0.44)
        slider_x, slider_y, slider_width = self.CONFIDENCE_SLIDER
        slider_end = slider_x + slider_width
        cv2.line(canvas, (slider_x, slider_y), (slider_end, slider_y), self.BORDER, 5)
        current_x = slider_x + int(self.detector.confidence * slider_width)
        cv2.line(canvas, (slider_x, slider_y), (current_x, slider_y), self.BLUE, 5)
        cv2.circle(canvas, (current_x, slider_y), 8, self.BLUE, -1)
        self._text(canvas, "Image size", (997, 490), 0.4, self.MUTED)
        for index, image_size in enumerate(self.IMAGE_SIZE_OPTIONS):
            x = 997 + index * 78
            color = self.BLUE if self.detector.image_size == image_size else (235, 239, 247)
            text_color = self.WHITE if self.detector.image_size == image_size else self.TEXT
            cv2.rectangle(canvas, (x, 500), (x + 72, 526), color, -1)
            self._text(canvas, str(image_size), (x + 17, 519), 0.42, text_color, 1)

        self._button(
            canvas,
            self.PAUSE_BUTTON,
            "Resume" if self.paused else "Pause",
            self.GREEN if self.paused else self.BLUE,
        )
        self._button(canvas, self.STOP_BUTTON, "Stop", self.RED)

        self._panel(canvas, (975, 548, 275, 80))
        self._text(canvas, "STATUS", (997, 576), 0.43, self.MUTED, 1)
        status_color = self.RED if "Unable to open" in frame_status else state_color
        if "Unable to" in frame_status:
            status_color = self.RED
        self._text(canvas, frame_status, (997, 604), 0.38, status_color)
        self._button(canvas, self.SAVE_SETTINGS_BUTTON, "Save settings", self.BLUE)
        self._button(canvas, self.RESTORE_DEFAULTS_BUTTON, "Defaults", (72, 83, 102))
        self._text(canvas, f"Processed frames: {processed_frames}", (220, 730), 0.43, self.MUTED)
        self._text(canvas, "Shortcuts: Space to pause | Q/Esc to exit", (220, 760), 0.43, self.MUTED)
        return canvas


def run_opencv_dashboard(
    detector: PersonDetector,
    camera_index: int = 0,
    window_name: str = "YOLOv8 Human Detection",
    settings_path: Path | None = None,
) -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open webcam device index: {camera_index}")

    try:
        dashboard = OpenCVDashboard(detector, camera_index, settings_path)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, dashboard.WIDTH, dashboard.HEIGHT)
        cv2.setMouseCallback(window_name, dashboard.handle_mouse)

        processed_frames = 0
        people_count = 0
        started_at = time.perf_counter()
        frame_status = "Camera connected"
        while not dashboard.should_close:
            if dashboard.requested_camera_index is not None:
                requested_index = dashboard.requested_camera_index
                dashboard.requested_camera_index = None
                candidate = cv2.VideoCapture(requested_index)
                if candidate.isOpened():
                    capture.release()
                    capture = candidate
                    dashboard.camera_index = requested_index
                    dashboard.status_message = None
                    people_count = 0
                else:
                    candidate.release()
                    dashboard.status_message = f"Unable to open camera {requested_index}"

            success, frame = capture.read()
            if not success:
                frame_status = "No frame received from camera"
                break
            if not dashboard.paused:
                annotated_frame, people_count = detector.process_frame(frame)
                processed_frames += 1
            elif dashboard.paused:
                annotated_frame = frame
                frame_status = "Detection is paused"

            elapsed = max(time.perf_counter() - started_at, 1e-9)
            fps = processed_frames / elapsed
            if dashboard.paused:
                frame_status = "Detection is paused"
            else:
                frame_status = dashboard.status_message or "Camera connected"

            cv2.imshow(
                window_name,
                dashboard.render(
                    annotated_frame,
                    people_count,
                    fps,
                    processed_frames,
                    frame_status,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                dashboard.paused = not dashboard.paused
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if processed_frames == 0:
        raise ValueError("No frames were processed from the webcam.")
    return processed_frames
