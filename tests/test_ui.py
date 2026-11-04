import unittest
from unittest.mock import patch

import cv2

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
        self.assertEqual(dashboard.requested_camera_index, 0)
        mock_save_settings.assert_called_once()

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

        processed = run_opencv_dashboard(detector, camera_index=0)

        self.assertEqual(processed, 1)
        self.assertEqual(detector.processed_frames, 1)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
