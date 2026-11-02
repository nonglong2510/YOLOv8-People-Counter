import time
from typing import Any

import cv2
import numpy as np

from ai_human_detected.detector import PersonDetector


class OpenCVDashboard:
    WIDTH = 1280
    HEIGHT = 800
    SIDEBAR_WIDTH = 190
    HEADER_HEIGHT = 64
    VIDEO_RECT = (220, 130, 730, 500)
    PAUSE_BUTTON = (975, 76, 130, 40)
    STOP_BUTTON = (1115, 76, 135, 40)

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
    ) -> None:
        self.detector = detector
        self.camera_index = camera_index
        self.paused = False
        self.should_close = False

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

        self._text(canvas, "YOLOv8 VISION", (22, 38), 0.72, self.WHITE, 2)
        self._text(canvas, "Nhan dien va dem nguoi", (20, 58), 0.35, (205, 213, 224))
        cv2.rectangle(canvas, (12, 91), (178, 132), (53, 72, 98), -1)
        self._text(canvas, "Giam sat truc tiep", (24, 117), 0.47, self.WHITE, 1)
        self._text(canvas, "Dieu khien", (24, 170), 0.48, (205, 213, 224), 1)
        self._text(canvas, "Space: Tam dung", (24, 198), 0.4, (177, 188, 202))
        self._text(canvas, "Q / Esc: Thoat", (24, 222), 0.4, (177, 188, 202))
        self._text(canvas, "Camera index", (24, 281), 0.42, (177, 188, 202))
        self._text(canvas, str(self.camera_index), (24, 307), 0.65, self.WHITE, 2)
        self._text(canvas, "OpenCV Desktop", (20, 755), 0.4, (177, 188, 202))

        self._text(canvas, "Giam sat truc tiep", (220, 39), 0.72, self.TEXT, 2)
        state_text = "TAM DUNG" if self.paused else "DANG CHAY"
        state_color = self.MUTED if self.paused else self.GREEN
        cv2.circle(canvas, (690, 33), 6, state_color, -1)
        self._text(canvas, state_text, (705, 39), 0.48, state_color, 2)
        self._text(canvas, f"Camera index: {self.camera_index}", (930, 39), 0.45, self.MUTED)

        self._panel(canvas, self.VIDEO_RECT)
        video_x, video_y, video_width, video_height = self.VIDEO_RECT
        if frame is None:
            self._text(canvas, "Dang cho khung hinh tu camera...", (video_x + 170, video_y + 250), 0.65, self.MUTED)
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
                self._text(canvas, "TAM DUNG NHAN DIEN", (video_x + 20, video_y + 34), 0.5, self.WHITE, 2)

        self._text(canvas, "VIDEO WEBCAM", (video_x, video_y - 14), 0.42, self.MUTED, 1)
        self._panel(canvas, (video_x, 648, video_width, 54))
        self._text(
            canvas,
            "So nguoi hien tai la so nguoi duoc phat hien trong frame.",
            (video_x + 14, 671),
            0.43,
            self.TEXT,
        )
        self._text(
            canvas,
            "Khong phai tong luot nguoi di qua.",
            (video_x + 14, 690),
            0.43,
            self.MUTED,
        )

        self._panel(canvas, (975, 130, 275, 125))
        self._text(canvas, "SO NGUOI HIEN TAI", (997, 160), 0.45, self.MUTED, 1)
        self._text(canvas, str(people_count), (997, 222), 1.8, self.BLUE, 2)
        self._text(canvas, "nguoi / frame", (1050, 220), 0.45, self.MUTED)

        self._panel(canvas, (975, 270, 275, 100))
        self._text(canvas, "FPS TRUNG BINH", (997, 300), 0.45, self.MUTED, 1)
        self._text(canvas, f"{fps:.1f} FPS", (997, 345), 0.95, self.GREEN, 2)

        self._panel(canvas, (975, 385, 275, 115))
        self._text(canvas, "CAU HINH NHAN DIEN", (997, 415), 0.45, self.MUTED, 1)
        self._text(canvas, f"Confidence: {self.detector.confidence:.2f}", (997, 447), 0.48)
        self._text(canvas, f"Image size: {self.detector.image_size}", (997, 476), 0.48)

        self._button(
            canvas,
            self.PAUSE_BUTTON,
            "Tiep tuc" if self.paused else "Tam dung",
            self.GREEN if self.paused else self.BLUE,
        )
        self._button(canvas, self.STOP_BUTTON, "Dung", self.RED)

        self._panel(canvas, (975, 540, 275, 80))
        self._text(canvas, "TRANG THAI", (997, 568), 0.43, self.MUTED, 1)
        self._text(canvas, frame_status, (997, 596), 0.43, state_color)
        self._text(canvas, f"So frame da xu ly: {processed_frames}", (220, 730), 0.43, self.MUTED)
        self._text(canvas, "Phim tat: Space tam dung | Q/Esc thoat", (220, 760), 0.43, self.MUTED)
        return canvas


def run_opencv_dashboard(
    detector: PersonDetector,
    camera_index: int = 0,
    window_name: str = "YOLOv8 Human Detection",
) -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open webcam device index: {camera_index}")

    try:
        dashboard = OpenCVDashboard(detector, camera_index)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, dashboard.WIDTH, dashboard.HEIGHT)
        cv2.setMouseCallback(window_name, dashboard.handle_mouse)

        processed_frames = 0
        people_count = 0
        started_at = time.perf_counter()
        frame_status = "Camera dang hoat dong"
        while not dashboard.should_close:
            success, frame = capture.read()
            if not success:
                frame_status = "Khong nhan duoc frame tu camera"
                break
            if not dashboard.paused:
                annotated_frame, people_count = detector.process_frame(frame)
                processed_frames += 1
            elif dashboard.paused:
                annotated_frame = frame
                frame_status = "Nhan dien dang tam dung"

            elapsed = max(time.perf_counter() - started_at, 1e-9)
            fps = processed_frames / elapsed
            if not dashboard.paused:
                frame_status = "Camera dang hoat dong"

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
