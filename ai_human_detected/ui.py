import sqlite3
import time
from dataclasses import replace
from datetime import datetime, time as datetime_time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ai_human_detected.capacity import CapacityMonitor, CapacityStatus
from ai_human_detected.detector import PersonDetector
from ai_human_detected.session_store import (
    SAMPLE_INTERVAL_SECONDS,
    SessionStore,
    SessionSummary,
    WeeklySchedule,
    find_active_schedule,
)
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
    SESSION_BUTTON = (230, 674, 150, 34)
    EXPORT_CSV_BUTTON = (390, 674, 140, 34)
    HISTORY_BUTTON = (540, 674, 125, 34)
    SCHEDULES_BUTTON = (680, 674, 130, 34)
    SCHEDULE_PAGE_SIZE = 4
    SCHEDULE_ROW_START = (240, 200)
    SCHEDULE_ROW_STEP = 36
    SCHEDULE_ROW_RECT = (240, 0, 680, 32)
    SCHEDULE_PREVIOUS_BUTTON = (790, 344, 50, 30)
    SCHEDULE_NEXT_BUTTON = (850, 344, 50, 30)
    SCHEDULE_START_DECREASE = (395, 386, 40, 30)
    SCHEDULE_START_INCREASE = (440, 386, 40, 30)
    SCHEDULE_END_DECREASE = (680, 386, 40, 30)
    SCHEDULE_END_INCREASE = (725, 386, 40, 30)
    SCHEDULE_DAY_BUTTONS = tuple((240 + index * 96, 454, 82, 32) for index in range(7))
    SCHEDULE_ADD_BUTTON = (240, 520, 100, 36)
    SCHEDULE_SAVE_BUTTON = (350, 520, 100, 36)
    SCHEDULE_TOGGLE_BUTTON = (460, 520, 130, 36)
    SCHEDULE_DELETE_BUTTON = (600, 520, 100, 36)
    WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    SAVE_SETTINGS_BUTTON = (975, 648, 130, 48)
    RESTORE_DEFAULTS_BUTTON = (1115, 648, 135, 48)
    CAPACITY_DECREASE_BUTTON = (1130, 558, 42, 30)
    CAPACITY_INCREASE_BUTTON = (1180, 558, 42, 30)
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
        capacity_limit: int = 30,
        session_store: SessionStore | None = None,
    ) -> None:
        self.detector = detector
        self.camera_index = camera_index
        self.settings_path = settings_path
        self.capacity_limit = capacity_limit
        self.capacity_monitor = CapacityMonitor(capacity_limit)
        self.session_store = session_store
        self.active_session_id: int | None = None
        self.last_sample_time: float | None = None
        self.history_visible = False
        self.history_sessions: list[SessionSummary] = []
        self.last_session_summary: SessionSummary | None = None
        self.schedule_manager_visible = False
        self.schedules: list[WeeklySchedule] = []
        self.schedule_page = 0
        self.schedule_draft = WeeklySchedule(
            None,
            "Schedule 1",
            datetime_time(7, 0),
            datetime_time(9, 30),
            tuple(range(7)),
            False,
        )
        self.active_schedule_id: int | None = None
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
        if self._inside((x, y), self.SCHEDULES_BUTTON):
            self.toggle_schedule_manager()
            return
        if self.schedule_manager_visible:
            self._handle_schedule_manager_mouse(x, y)
            return
        if self._inside((x, y), self.PAUSE_BUTTON):
            self.paused = not self.paused
        elif self._inside((x, y), self.STOP_BUTTON):
            self.should_close = True
        elif self._inside((x, y), self.SESSION_BUTTON):
            self.toggle_session()
        elif self._inside((x, y), self.EXPORT_CSV_BUTTON):
            self.export_session_report()
        elif self._inside((x, y), self.HISTORY_BUTTON):
            self.toggle_history()
        elif self._inside((x, y), self.CAMERA_DECREASE_BUTTON):
            self.requested_camera_index = max(0, self.camera_index - 1)
        elif self._inside((x, y), self.CAMERA_INCREASE_BUTTON):
            self.requested_camera_index = self.camera_index + 1
        elif self._inside((x, y), self.SAVE_SETTINGS_BUTTON):
            self.save_current_settings()
        elif self._inside((x, y), self.RESTORE_DEFAULTS_BUTTON):
            self.restore_defaults()
        elif self._inside((x, y), self.CAPACITY_DECREASE_BUTTON):
            if self.active_session_id is not None:
                self.status_message = "End the session before changing room capacity."
            else:
                self.set_capacity_limit(max(1, self.capacity_limit - 1))
        elif self._inside((x, y), self.CAPACITY_INCREASE_BUTTON):
            if self.active_session_id is not None:
                self.status_message = "End the session before changing room capacity."
            else:
                self.set_capacity_limit(self.capacity_limit + 1)
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
            capacity_limit=self.capacity_limit,
        )
        try:
            save_settings(settings, self.settings_path)
        except OSError as error:
            self.status_message = f"Unable to save settings: {error}"
            return
        self.status_message = "Settings saved for the next launch."

    def set_capacity_limit(self, capacity_limit: int) -> None:
        self.capacity_monitor.reset(capacity_limit)
        self.capacity_limit = capacity_limit

    def _ensure_session_store(self) -> SessionStore:
        if self.session_store is None:
            self.session_store = SessionStore()
        return self.session_store

    def toggle_schedule_manager(self) -> None:
        if self.schedule_manager_visible:
            self.schedule_manager_visible = False
            return
        try:
            self.schedules = self._ensure_session_store().list_schedules()
        except (OSError, sqlite3.Error) as error:
            self.status_message = f"Unable to load schedules: {error}"
            return
        self.schedule_page = 0
        self.history_visible = False
        self.schedule_manager_visible = True
        if self.schedules:
            self._load_schedule_draft(self.schedules[0])
        else:
            self._new_schedule_draft()

    def _new_schedule_draft(self) -> None:
        next_number = max(
            (schedule.schedule_id or 0 for schedule in self.schedules),
            default=0,
        ) + 1
        self.schedule_draft = WeeklySchedule(
            None,
            f"Schedule {next_number}",
            datetime_time(7, 0),
            datetime_time(9, 30),
            tuple(range(7)),
            False,
        )

    def _load_schedule_draft(self, schedule: WeeklySchedule) -> None:
        self.schedule_draft = schedule
        if schedule.schedule_id is not None:
            index = next(
                (
                    index
                    for index, item in enumerate(self.schedules)
                    if item.schedule_id == schedule.schedule_id
                ),
                0,
            )
            self.schedule_page = index // self.SCHEDULE_PAGE_SIZE

    def _shift_schedule_time(self, field: str, minutes: int) -> None:
        current = getattr(self.schedule_draft, field)
        total_minutes = current.hour * 60 + current.minute + minutes
        if total_minutes < 0 or total_minutes >= 24 * 60:
            self.status_message = "Schedule time must remain within the same day."
            return
        self.schedule_draft = replace(
            self.schedule_draft,
            **{
                field: datetime_time(
                    total_minutes // 60,
                    total_minutes % 60,
                )
            },
        )

    def _handle_schedule_manager_mouse(self, x: int, y: int) -> None:
        _, start_y = self.SCHEDULE_ROW_START
        page_start = self.schedule_page * self.SCHEDULE_PAGE_SIZE
        for visible_index, schedule in enumerate(
            self.schedules[page_start : page_start + self.SCHEDULE_PAGE_SIZE]
        ):
            row_rect = (
                self.SCHEDULE_ROW_RECT[0],
                start_y + visible_index * self.SCHEDULE_ROW_STEP,
                self.SCHEDULE_ROW_RECT[2],
                self.SCHEDULE_ROW_RECT[3],
            )
            if self._inside((x, y), row_rect):
                self._load_schedule_draft(schedule)
                return

        if self._inside((x, y), self.SCHEDULE_PREVIOUS_BUTTON):
            self.schedule_page = max(0, self.schedule_page - 1)
            return
        max_page = max(
            0,
            (len(self.schedules) - 1) // self.SCHEDULE_PAGE_SIZE,
        )
        if self._inside((x, y), self.SCHEDULE_NEXT_BUTTON):
            self.schedule_page = min(max_page, self.schedule_page + 1)
            return
        if self._inside((x, y), self.SCHEDULE_START_DECREASE):
            self._shift_schedule_time("start_time", -15)
            return
        if self._inside((x, y), self.SCHEDULE_START_INCREASE):
            self._shift_schedule_time("start_time", 15)
            return
        if self._inside((x, y), self.SCHEDULE_END_DECREASE):
            self._shift_schedule_time("end_time", -15)
            return
        if self._inside((x, y), self.SCHEDULE_END_INCREASE):
            self._shift_schedule_time("end_time", 15)
            return
        for day, rect in enumerate(self.SCHEDULE_DAY_BUTTONS):
            if self._inside((x, y), rect):
                weekdays = set(self.schedule_draft.weekdays)
                if day in weekdays:
                    weekdays.remove(day)
                else:
                    weekdays.add(day)
                self.schedule_draft = replace(
                    self.schedule_draft,
                    weekdays=tuple(sorted(weekdays)),
                )
                return
        if self._inside((x, y), self.SCHEDULE_ADD_BUTTON):
            self._new_schedule_draft()
            return
        if self._inside((x, y), self.SCHEDULE_SAVE_BUTTON):
            self._save_schedule_draft()
            return
        if self._inside((x, y), self.SCHEDULE_TOGGLE_BUTTON):
            self.schedule_draft = replace(
                self.schedule_draft,
                enabled=not self.schedule_draft.enabled,
            )
            return
        if self._inside((x, y), self.SCHEDULE_DELETE_BUTTON):
            self._delete_schedule_draft()

    def _save_schedule_draft(self) -> None:
        try:
            saved = self._ensure_session_store().save_schedule(
                self.schedule_draft.schedule_id,
                self.schedule_draft.name,
                self.schedule_draft.start_time.strftime("%H:%M"),
                self.schedule_draft.end_time.strftime("%H:%M"),
                self.schedule_draft.weekdays,
                self.schedule_draft.enabled,
            )
            self.schedules = self._ensure_session_store().list_schedules()
        except (OSError, sqlite3.Error, ValueError) as error:
            self.status_message = f"Unable to save schedule: {error}"
            return
        self._load_schedule_draft(saved)
        self.status_message = f"{saved.name} saved."

    def _delete_schedule_draft(self) -> None:
        schedule_id = self.schedule_draft.schedule_id
        if schedule_id is None:
            self.status_message = "Select a saved schedule to delete."
            return
        try:
            self._ensure_session_store().delete_schedule(schedule_id)
            self.schedules = self._ensure_session_store().list_schedules()
        except (OSError, sqlite3.Error, ValueError) as error:
            self.status_message = f"Unable to delete schedule: {error}"
            return
        if self.schedules:
            self._load_schedule_draft(self.schedules[0])
        else:
            self._new_schedule_draft()
        self.status_message = "Schedule deleted."

    def update_scheduled_session(self, current_time: datetime | None = None) -> None:
        if self.session_store is None:
            return
        matching_schedule = find_active_schedule(
            self.session_store.list_schedules(),
            current_time,
        )

        if self.active_session_id is not None:
            if self.active_schedule_id is None:
                if matching_schedule is None:
                    return
                self.end_active_session(current_time)
            elif (
                matching_schedule is not None
                and matching_schedule.schedule_id == self.active_schedule_id
            ):
                return
            else:
                self.end_active_session(current_time)

        if matching_schedule is None:
            return
        if matching_schedule.schedule_id is None:
            raise ValueError("A saved schedule must have an ID.")
        self.active_session_id = self.session_store.start_session(
            self.capacity_limit,
            self.camera_index,
            started_at=current_time,
            schedule_id=matching_schedule.schedule_id,
        )
        self.active_schedule_id = matching_schedule.schedule_id
        self.last_sample_time = None
        self.last_session_summary = None
        self.status_message = f"Scheduled session started: {matching_schedule.name}."

    def toggle_session(self) -> None:
        try:
            if self.active_session_id is not None:
                if self.active_schedule_id is not None:
                    self.status_message = "Scheduled sessions end automatically."
                    return
                self.last_session_summary = self._ensure_session_store().end_session(
                    self.active_session_id
                )
                self.active_session_id = None
                self.last_sample_time = None
                self.status_message = (
                    f"Session ended. Peak: {self.last_session_summary.peak_people}"
                )
                return

            self.active_session_id = self._ensure_session_store().start_session(
                self.capacity_limit,
                self.camera_index,
            )
            self.last_sample_time = None
            self.last_session_summary = None
            self.status_message = f"Session {self.active_session_id} started."
        except (OSError, sqlite3.Error) as error:
            self.status_message = f"Unable to manage session: {error}"

    def record_session_sample(self, monotonic_time: float | None = None) -> None:
        if self.active_session_id is None:
            return
        now = time.monotonic() if monotonic_time is None else monotonic_time
        if (
            self.last_sample_time is not None
            and now - self.last_sample_time < SAMPLE_INTERVAL_SECONDS
        ):
            return

        self._ensure_session_store().record_sample(
            self.active_session_id,
            self.capacity_monitor.smoothed_count,
            self.capacity_limit,
            self.camera_index,
            self.capacity_monitor.status,
        )
        self.last_sample_time = now

    def end_active_session(self, ended_at: datetime | None = None) -> None:
        if self.active_session_id is None:
            return
        self.last_session_summary = self._ensure_session_store().end_session(
            self.active_session_id,
            ended_at,
        )
        self.active_session_id = None
        self.active_schedule_id = None
        self.last_sample_time = None

    def export_session_report(self) -> None:
        try:
            export_path = self._ensure_session_store().export_csv()
        except (OSError, sqlite3.Error) as error:
            self.status_message = f"Unable to export session report: {error}"
            return
        self.status_message = f"CSV saved: {export_path.name}"

    def toggle_history(self) -> None:
        try:
            self.history_sessions = self._ensure_session_store().list_sessions(5)
        except (OSError, sqlite3.Error) as error:
            self.status_message = f"Unable to load session history: {error}"
            return
        self.history_visible = not self.history_visible

    def restore_defaults(self) -> None:
        if self.active_session_id is not None:
            self.status_message = "End the session before restoring defaults."
            return
        self.detector.confidence = DEFAULT_SETTINGS.confidence
        self.detector.image_size = DEFAULT_SETTINGS.image_size
        self.set_capacity_limit(DEFAULT_SETTINGS.capacity_limit)
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
        header_status = self.status_message or f"Camera {self.camera_index} connected"
        header_color = self.RED if "Unable to" in header_status else self.GREEN
        self._text(canvas, header_status, (930, 39), 0.38, header_color)

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

        if self.history_visible:
            self._panel(canvas, self.VIDEO_RECT, self.WHITE)
            self._text(canvas, "RECENT MONITORING SESSIONS", (video_x + 20, video_y + 36), 0.55, self.TEXT, 2)
            if not self.history_sessions:
                self._text(canvas, "No monitoring sessions have been saved.", (video_x + 20, video_y + 80), 0.48, self.MUTED)
            for index, summary in enumerate(self.history_sessions):
                started = summary.started_at.astimezone().strftime("%Y-%m-%d %H:%M")
                average = "-" if summary.average_people is None else f"{summary.average_people:.1f}"
                ended = "active" if summary.ended_at is None else "ended"
                source = (
                    "manual"
                    if summary.schedule_id is None
                    else f"schedule:{summary.schedule_id}"
                )
                row = (
                    f"#{summary.session_id}  {started}  {source} {ended}  "
                    f"samples:{summary.sample_count}  avg:{average}  peak:{summary.peak_people}"
                )
                self._text(
                    canvas,
                    row,
                    (video_x + 20, video_y + 86 + index * 42),
                    0.43,
                    self.TEXT,
                )

        if self.schedule_manager_visible:
            self._panel(canvas, self.VIDEO_RECT, self.WHITE)
            self._text(canvas, "WEEKLY SCHEDULES", (video_x + 20, video_y + 34), 0.58, self.TEXT, 2)
            page_start = self.schedule_page * self.SCHEDULE_PAGE_SIZE
            visible_schedules = self.schedules[
                page_start : page_start + self.SCHEDULE_PAGE_SIZE
            ]
            if not self.schedules:
                self._text(
                    canvas,
                    "No schedules yet. New creates a daily 07:00-09:30 draft.",
                    (video_x + 20, video_y + 80),
                    0.45,
                    self.MUTED,
                )
            for index, schedule in enumerate(visible_schedules):
                row_y = self.SCHEDULE_ROW_START[1] + index * self.SCHEDULE_ROW_STEP
                row_rect = (
                    self.SCHEDULE_ROW_RECT[0],
                    row_y,
                    self.SCHEDULE_ROW_RECT[2],
                    self.SCHEDULE_ROW_RECT[3],
                )
                selected = schedule.schedule_id == self.schedule_draft.schedule_id
                cv2.rectangle(
                    canvas,
                    (row_rect[0], row_rect[1]),
                    (row_rect[0] + row_rect[2], row_rect[1] + row_rect[3]),
                    (229, 239, 250) if selected else (245, 247, 252),
                    -1,
                )
                weekdays = ",".join(
                    self.WEEKDAY_LABELS[day] for day in schedule.weekdays
                )
                active_text = "ON" if schedule.enabled else "OFF"
                running_text = (
                    "  RUNNING"
                    if schedule.schedule_id == self.active_schedule_id
                    else ""
                )
                self._text(
                    canvas,
                    (
                        f"{schedule.schedule_id}. {schedule.name}  "
                        f"{schedule.start_time:%H:%M}-{schedule.end_time:%H:%M}  "
                        f"{weekdays}  {active_text}{running_text}"
                    ),
                    (row_rect[0] + 10, row_rect[1] + 22),
                    0.4,
                    self.TEXT,
                )

            max_page = max(
                0,
                (len(self.schedules) - 1) // self.SCHEDULE_PAGE_SIZE,
            )
            self._button(canvas, self.SCHEDULE_PREVIOUS_BUTTON, "<", (72, 83, 102))
            self._button(canvas, self.SCHEDULE_NEXT_BUTTON, ">", (72, 83, 102))
            self._text(
                canvas,
                f"Page {self.schedule_page + 1} / {max_page + 1}",
                (700, 369),
                0.38,
                self.MUTED,
            )

            self._text(
                canvas,
                f"Start {self.schedule_draft.start_time:%H:%M}",
                (video_x + 20, 407),
                0.45,
                self.TEXT,
            )
            self._button(canvas, self.SCHEDULE_START_DECREASE, "-", (72, 83, 102))
            self._button(canvas, self.SCHEDULE_START_INCREASE, "+", (72, 83, 102))
            self._text(
                canvas,
                f"End {self.schedule_draft.end_time:%H:%M}",
                (video_x + 330, 407),
                0.45,
                self.TEXT,
            )
            self._button(canvas, self.SCHEDULE_END_DECREASE, "-", (72, 83, 102))
            self._button(canvas, self.SCHEDULE_END_INCREASE, "+", (72, 83, 102))
            self._text(canvas, "Repeat on:", (video_x + 20, 442), 0.4, self.MUTED)
            for day, rect in enumerate(self.SCHEDULE_DAY_BUTTONS):
                selected = day in self.schedule_draft.weekdays
                color = self.BLUE if selected else (235, 239, 247)
                text_color = self.WHITE if selected else self.TEXT
                cv2.rectangle(
                    canvas,
                    (rect[0], rect[1]),
                    (rect[0] + rect[2], rect[1] + rect[3]),
                    color,
                    -1,
                )
                self._text(
                    canvas,
                    self.WEEKDAY_LABELS[day],
                    (rect[0] + 18, rect[1] + 22),
                    0.4,
                    text_color,
                )
            self._button(canvas, self.SCHEDULE_ADD_BUTTON, "New", self.GREEN)
            self._button(canvas, self.SCHEDULE_SAVE_BUTTON, "Save", self.BLUE)
            toggle_label = "Disable" if self.schedule_draft.enabled else "Enable"
            self._button(canvas, self.SCHEDULE_TOGGLE_BUTTON, toggle_label, (72, 83, 102))
            self._button(canvas, self.SCHEDULE_DELETE_BUTTON, "Delete", self.RED)
            self._text(
                canvas,
                "15-minute steps; Save applies edits. Same-day only; app and camera must stay running.",
                (video_x + 20, 586),
                0.38,
                self.MUTED,
            )

        self._text(canvas, "WEBCAM VIDEO", (video_x, video_y - 14), 0.42, self.MUTED, 1)
        self._panel(canvas, (video_x, 648, video_width, 66))
        session_state = "ACTIVE" if self.active_session_id is not None else "IDLE"
        self._text(
            canvas,
            f"SESSION: {session_state}  |  Samples every second; paused detection is not sampled.",
            (video_x + 14, 666),
            0.38,
            self.GREEN if self.active_session_id is not None else self.MUTED,
        )
        if self.active_schedule_id is not None:
            session_button_label = "Scheduled"
            session_button_color = self.MUTED
        elif self.active_session_id is not None:
            session_button_label = "End session"
            session_button_color = self.RED
        else:
            session_button_label = "Start session"
            session_button_color = self.GREEN
        self._button(
            canvas,
            self.SESSION_BUTTON,
            session_button_label,
            session_button_color,
        )
        self._button(canvas, self.EXPORT_CSV_BUTTON, "Export CSV", self.BLUE)
        self._button(
            canvas,
            self.HISTORY_BUTTON,
            "Close history" if self.history_visible else "History",
            (72, 83, 102),
        )
        self._button(
            canvas,
            self.SCHEDULES_BUTTON,
            "Close schedule" if self.schedule_manager_visible else "Schedules",
            (72, 83, 102),
        )
        if self.last_session_summary is not None and self.active_session_id is None:
            summary = self.last_session_summary
            avg = "-" if summary.average_people is None else f"{summary.average_people:.1f}"
            self._text(
                canvas,
                f"Avg {avg}, peak {summary.peak_people}",
                (820, 697),
                0.38,
                self.MUTED,
            )
        elif self.active_session_id is not None:
            session_source = (
                f"Schedule #{self.active_schedule_id}"
                if self.active_schedule_id is not None
                else f"Session #{self.active_session_id}"
            )
            self._text(
                canvas,
                session_source,
                (820, 697),
                0.4,
                self.TEXT,
            )

        self._panel(canvas, (975, 130, 275, 125))
        self._text(canvas, "PEOPLE IN CURRENT FRAME", (997, 160), 0.42, self.MUTED, 1)
        occupancy_status = self.capacity_monitor.status
        count_color = self.RED if occupancy_status in (
            CapacityStatus.AT_CAPACITY,
            CapacityStatus.OVER_CAPACITY,
        ) else (
            (0, 165, 235)
            if occupancy_status is CapacityStatus.NEAR_CAPACITY
            else self.BLUE
        )
        self._text(canvas, str(people_count), (997, 222), 1.8, count_color, 2)
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

        self._panel(canvas, (975, 548, 275, 88))
        self._text(canvas, "ROOM CAPACITY", (997, 574), 0.43, self.MUTED, 1)
        self._button(canvas, self.CAPACITY_DECREASE_BUTTON, "-", (72, 83, 102))
        self._button(canvas, self.CAPACITY_INCREASE_BUTTON, "+", (72, 83, 102))
        self._text(
            canvas,
            f"Smoothed: {self.capacity_monitor.smoothed_count} / {self.capacity_limit}",
            (997, 607),
            0.46,
        )
        capacity_colors = {
            CapacityStatus.NORMAL: self.GREEN,
            CapacityStatus.NEAR_CAPACITY: (0, 165, 235),
            CapacityStatus.AT_CAPACITY: self.RED,
            CapacityStatus.OVER_CAPACITY: self.RED,
        }
        self._text(
            canvas,
            occupancy_status.value,
            (997, 629),
            0.38,
            capacity_colors[occupancy_status],
            2,
        )
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
    capacity_limit: int = 30,
    session_store: SessionStore | None = None,
) -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open webcam device index: {camera_index}")

    dashboard: OpenCVDashboard | None = None
    try:
        dashboard = OpenCVDashboard(
            detector,
            camera_index,
            settings_path,
            capacity_limit,
            session_store or SessionStore(),
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, dashboard.WIDTH, dashboard.HEIGHT)
        cv2.setMouseCallback(window_name, dashboard.handle_mouse)

        processed_frames = 0
        people_count = 0
        started_at = time.perf_counter()
        last_schedule_check = 0.0
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
                    dashboard.capacity_monitor.reset()
                else:
                    candidate.release()
                    dashboard.status_message = f"Unable to open camera {requested_index}"

            success, frame = capture.read()
            if not success:
                frame_status = "No frame received from camera"
                break
            monotonic_now = time.monotonic()
            if monotonic_now - last_schedule_check >= SAMPLE_INTERVAL_SECONDS:
                dashboard.update_scheduled_session(datetime.now().astimezone())
                last_schedule_check = monotonic_now
            if not dashboard.paused:
                annotated_frame, people_count = detector.process_frame(frame)
                processed_frames += 1
                dashboard.capacity_monitor.update(people_count)
                dashboard.record_session_sample()
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
        try:
            if dashboard is not None:
                dashboard.end_active_session()
        finally:
            capture.release()
            cv2.destroyAllWindows()

    if processed_frames == 0:
        raise ValueError("No frames were processed from the webcam.")
    return processed_frames
