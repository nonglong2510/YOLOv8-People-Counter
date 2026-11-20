import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from ai_human_detected.capacity import CapacityStatus
from ai_human_detected.session_store import SessionStore
from ai_human_detected.ui import OpenCVDashboard, run_opencv_dashboard


class FakeDetector:
    confidence = 0.25
    image_size = 640
    model_path = "yolov8n.pt"

    def __init__(self):
        self.processed_frames = 0

    def process_frame(self, frame):
        self.processed_frames += 1
        return f"{frame}-annotated", 2


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class OpenCVDashboardTests(unittest.TestCase):
    def test_dashboard_renders_capacity_status_and_count(self):
        dashboard = OpenCVDashboard(FakeDetector(), 0, capacity_limit=30)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            dashboard.capacity_monitor.update(24)
        for _ in range(3):
            dashboard.capacity_monitor.update(24)

        with patch.object(dashboard, "_text") as mock_text:
            rendered = dashboard.render(frame, 24, 12.5, 10, "Camera connected")

        self.assertEqual(rendered.shape, (dashboard.HEIGHT, dashboard.WIDTH, 3))
        rendered_text = [call.args[1] for call in mock_text.call_args_list]
        self.assertIn("Smoothed: 24 / 30", rendered_text)
        self.assertIn("NEAR CAPACITY", rendered_text)

    @patch("ai_human_detected.ui.save_settings")
    def test_save_settings_button_persists_current_values(self, mock_save_settings):
        dashboard = OpenCVDashboard(FakeDetector(), 2)
        dashboard.detector.confidence = 0.6
        dashboard.detector.image_size = 416

        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            dashboard.SAVE_SETTINGS_BUTTON[0] + 1,
            dashboard.SAVE_SETTINGS_BUTTON[1] + 1,
            0,
            None,
        )

        settings = mock_save_settings.call_args.args[0]
        self.assertEqual(settings.camera_index, 2)
        self.assertEqual(settings.confidence, 0.6)
        self.assertEqual(settings.image_size, 416)
        self.assertIn("Settings saved", dashboard.status_message)

    @patch("ai_human_detected.ui.save_settings")
    def test_restore_defaults_resets_controls_and_persists_defaults(self, mock_save_settings):
        detector = FakeDetector()
        detector.confidence = 0.8
        detector.image_size = 416
        dashboard = OpenCVDashboard(detector, 2)

        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            dashboard.RESTORE_DEFAULTS_BUTTON[0] + 1,
            dashboard.RESTORE_DEFAULTS_BUTTON[1] + 1,
            0,
            None,
        )

        self.assertEqual(detector.confidence, 0.25)
        self.assertEqual(detector.image_size, 640)
        self.assertEqual(dashboard.capacity_limit, 30)
        self.assertEqual(dashboard.requested_camera_index, 0)
        mock_save_settings.assert_called_once()

    @patch("ai_human_detected.ui.save_settings")
    def test_capacity_controls_and_save_settings(self, mock_save_settings):
        dashboard = OpenCVDashboard(FakeDetector(), 0)
        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            dashboard.CAPACITY_INCREASE_BUTTON[0] + 1,
            dashboard.CAPACITY_INCREASE_BUTTON[1] + 1,
            0,
            None,
        )
        self.assertEqual(dashboard.capacity_limit, 31)

        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            dashboard.SAVE_SETTINGS_BUTTON[0] + 1,
            dashboard.SAVE_SETTINGS_BUTTON[1] + 1,
            0,
            None,
        )
        saved = mock_save_settings.call_args.args[0]
        self.assertEqual(saved.capacity_limit, 31)

    def test_session_controls_record_samples_and_finish_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            dashboard = OpenCVDashboard(FakeDetector(), 0, session_store=store)
            for _ in range(3):
                dashboard.capacity_monitor.update(24)

            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.SESSION_BUTTON[0] + 1,
                dashboard.SESSION_BUTTON[1] + 1,
                0,
                None,
            )
            self.assertIsNotNone(dashboard.active_session_id)

            dashboard.record_session_sample(monotonic_time=10.0)
            dashboard.record_session_sample(monotonic_time=10.5)
            dashboard.record_session_sample(monotonic_time=11.0)
            dashboard.toggle_session()

            self.assertIsNone(dashboard.active_session_id)
            self.assertEqual(dashboard.last_session_summary.sample_count, 2)
            self.assertEqual(dashboard.last_session_summary.peak_people, 24)

    def test_room_capacity_cannot_change_during_a_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            dashboard = OpenCVDashboard(FakeDetector(), 0, session_store=store)
            dashboard.set_capacity_limit(45)
            dashboard.toggle_session()

            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.CAPACITY_INCREASE_BUTTON[0] + 1,
                dashboard.CAPACITY_INCREASE_BUTTON[1] + 1,
                0,
                None,
            )
            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.RESTORE_DEFAULTS_BUTTON[0] + 1,
                dashboard.RESTORE_DEFAULTS_BUTTON[1] + 1,
                0,
                None,
            )

            self.assertEqual(dashboard.capacity_limit, 45)
            self.assertIn("End the session", dashboard.status_message)
            dashboard.end_active_session()

    def test_history_button_loads_recent_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            session_id = store.start_session(30, 0)
            store.record_sample(
                session_id,
                5,
                30,
                0,
                CapacityStatus.NORMAL,
            )
            store.end_session(session_id)
            dashboard = OpenCVDashboard(FakeDetector(), 0, session_store=store)

            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.HISTORY_BUTTON[0] + 1,
                dashboard.HISTORY_BUTTON[1] + 1,
                0,
                None,
            )

            self.assertTrue(dashboard.history_visible)
            self.assertEqual(len(dashboard.history_sessions), 1)

    def test_schedule_manager_saves_time_and_weekday_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            dashboard = OpenCVDashboard(FakeDetector(), 0, session_store=store)
            dashboard.toggle_schedule_manager()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(dashboard, "_text") as mock_text:
                rendered = dashboard.render(frame, 0, 0.0, 0, "Camera connected")
            self.assertEqual(rendered.shape, (dashboard.HEIGHT, dashboard.WIDTH, 3))
            self.assertIn(
                "WEEKLY SCHEDULES",
                [call.args[1] for call in mock_text.call_args_list],
            )

            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.SCHEDULE_START_INCREASE[0] + 1,
                dashboard.SCHEDULE_START_INCREASE[1] + 1,
                0,
                None,
            )
            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.SCHEDULE_DAY_BUTTONS[5][0] + 1,
                dashboard.SCHEDULE_DAY_BUTTONS[5][1] + 1,
                0,
                None,
            )
            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.SCHEDULE_TOGGLE_BUTTON[0] + 1,
                dashboard.SCHEDULE_TOGGLE_BUTTON[1] + 1,
                0,
                None,
            )
            dashboard.handle_mouse(
                cv2.EVENT_LBUTTONDOWN,
                dashboard.SCHEDULE_SAVE_BUTTON[0] + 1,
                dashboard.SCHEDULE_SAVE_BUTTON[1] + 1,
                0,
                None,
            )

            saved = store.list_schedules()[0]
            self.assertEqual(saved.start_time.strftime("%H:%M"), "07:15")
            self.assertNotIn(5, saved.weekdays)
            self.assertTrue(saved.enabled)

    def test_scheduler_starts_and_ends_session_at_schedule_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            local_start = datetime(2026, 1, 5, 7, 0).astimezone()
            local_end = datetime(2026, 1, 5, 9, 30).astimezone()
            schedule = store.save_schedule(
                None,
                "Morning class",
                "07:00",
                "09:30",
                (local_start.weekday(),),
                True,
            )
            dashboard = OpenCVDashboard(FakeDetector(), 0, session_store=store)

            dashboard.update_scheduled_session(local_start)
            self.assertEqual(dashboard.active_schedule_id, schedule.schedule_id)
            active_session_id = dashboard.active_session_id

            dashboard.update_scheduled_session(local_end)

            self.assertIsNone(dashboard.active_session_id)
            summary = store.get_summary(active_session_id)
            self.assertEqual(summary.ended_at, local_end)
            self.assertEqual(summary.schedule_id, schedule.schedule_id)

    def test_scheduled_start_takes_over_an_active_manual_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            local_start = datetime(2026, 1, 5, 7, 0).astimezone()
            schedule = store.save_schedule(
                None,
                "Morning class",
                "07:00",
                "09:30",
                (local_start.weekday(),),
                True,
            )
            manual_session_id = store.start_session(
                30,
                0,
                started_at=local_start.replace(hour=6),
            )
            dashboard = OpenCVDashboard(FakeDetector(), 0, session_store=store)
            dashboard.active_session_id = manual_session_id

            dashboard.update_scheduled_session(local_start)

            self.assertEqual(dashboard.active_schedule_id, schedule.schedule_id)
            manual_summary = store.get_summary(manual_session_id)
            self.assertEqual(manual_summary.ended_at, local_start)
            scheduled_summary = store.get_summary(dashboard.active_session_id)
            self.assertEqual(scheduled_summary.schedule_id, schedule.schedule_id)

    def test_run_ends_active_session_when_quitting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            capture = FakeCapture(["frame-1"])
            detector = FakeDetector()

            with (
                patch("ai_human_detected.ui.cv2.VideoCapture", return_value=capture),
                patch("ai_human_detected.ui.cv2.namedWindow"),
                patch("ai_human_detected.ui.cv2.resizeWindow"),
                patch(
                    "ai_human_detected.ui.cv2.setMouseCallback",
                    side_effect=lambda _name, callback: callback(
                        cv2.EVENT_LBUTTONDOWN,
                        OpenCVDashboard.SESSION_BUTTON[0] + 1,
                        OpenCVDashboard.SESSION_BUTTON[1] + 1,
                        0,
                        None,
                    ),
                ),
                patch(
                    "ai_human_detected.ui.OpenCVDashboard.render",
                    return_value="dashboard",
                ),
                patch("ai_human_detected.ui.cv2.imshow"),
                patch("ai_human_detected.ui.cv2.waitKey", return_value=ord("q")),
                patch("ai_human_detected.ui.cv2.destroyAllWindows"),
            ):
                run_opencv_dashboard(detector, 0, session_store=store)

            summary = store.list_sessions()[0]
            self.assertIsNotNone(summary.ended_at)
            self.assertEqual(summary.sample_count, 1)

    def test_controls_update_camera_confidence_and_image_size(self):
        detector = FakeDetector()
        dashboard = OpenCVDashboard(detector, 0)

        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            dashboard.CAMERA_INCREASE_BUTTON[0] + 1,
            dashboard.CAMERA_INCREASE_BUTTON[1] + 1,
            0,
            None,
        )
        self.assertEqual(dashboard.requested_camera_index, 1)

        slider_x, slider_y, slider_width = dashboard.CONFIDENCE_SLIDER
        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            slider_x + slider_width // 2,
            slider_y,
            0,
            None,
        )
        self.assertEqual(detector.confidence, 0.5)

        image_size_x = 997 + 2 * 78 + 10
        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            image_size_x,
            510,
            0,
            None,
        )
        self.assertEqual(detector.image_size, 640)

    def test_mouse_controls_toggle_pause_and_stop(self):
        dashboard = OpenCVDashboard(FakeDetector(), 0)

        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            dashboard.PAUSE_BUTTON[0] + 1,
            dashboard.PAUSE_BUTTON[1] + 1,
            0,
            None,
        )
        self.assertTrue(dashboard.paused)

        dashboard.handle_mouse(
            cv2.EVENT_LBUTTONDOWN,
            dashboard.STOP_BUTTON[0] + 1,
            dashboard.STOP_BUTTON[1] + 1,
            0,
            None,
        )
        self.assertTrue(dashboard.should_close)

    @patch("ai_human_detected.ui.cv2.destroyAllWindows")
    @patch("ai_human_detected.ui.cv2.waitKey", side_effect=[ord(" "), ord("q")])
    @patch("ai_human_detected.ui.cv2.imshow")
    @patch("ai_human_detected.ui.cv2.setMouseCallback")
    @patch("ai_human_detected.ui.cv2.resizeWindow")
    @patch("ai_human_detected.ui.cv2.namedWindow")
    @patch("ai_human_detected.ui.cv2.VideoCapture")
    @patch("ai_human_detected.ui.OpenCVDashboard.render", return_value="dashboard")
    def test_space_pauses_inference_and_q_exits(
        self,
        _render,
        mock_video_capture,
        _named_window,
        _resize_window,
        _mouse_callback,
        _imshow,
        _wait_key,
        _destroy,
    ):
        capture = FakeCapture(["frame-1", "frame-2", "frame-3"])
        mock_video_capture.return_value = capture
        detector = FakeDetector()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.sqlite3")
            processed = run_opencv_dashboard(
                detector,
                camera_index=0,
                session_store=store,
            )

        self.assertEqual(processed, 1)
        self.assertEqual(detector.processed_frames, 1)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
