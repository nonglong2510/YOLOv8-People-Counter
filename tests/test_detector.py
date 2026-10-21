import unittest
from unittest.mock import patch

from ai_human_detected.detector import PersonDetector


class FakeResult:
    def __init__(self, boxes, plotted_frame="annotated"):
        self.boxes = boxes
        self.plotted_frame = plotted_frame

    def plot(self):
        return self.plotted_frame


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.predict_args = None

    def predict(self, **kwargs):
        self.predict_args = kwargs
        return [self.result]


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def release(self):
        return None


class PersonDetectorTests(unittest.TestCase):
    @patch("ai_human_detected.detector.cv2.putText", side_effect=lambda frame, *args: frame)
    def test_counts_only_the_person_detections_returned_by_model(self, _put_text):
        model = FakeModel(FakeResult(boxes=[object(), object()]))
        detector = PersonDetector(model=model)

        frame, count = detector.process_frame("source-frame")

        self.assertEqual(frame, "annotated")
        self.assertEqual(count, 2)
        self.assertEqual(model.predict_args["source"], "source-frame")
        self.assertEqual(model.predict_args["classes"], [0])

    @patch("ai_human_detected.detector.cv2.putText", side_effect=lambda frame, *args: frame)
    def test_returns_zero_when_model_has_no_boxes(self, _put_text):
        detector = PersonDetector(model=FakeModel(FakeResult(boxes=None)))

        _frame, count = detector.process_frame("source-frame")

        self.assertEqual(count, 0)

    @patch("ai_human_detected.detector.cv2.destroyAllWindows")
    @patch("ai_human_detected.detector.cv2.waitKey", return_value=ord("q"))
    @patch("ai_human_detected.detector.cv2.imshow")
    @patch("ai_human_detected.detector.cv2.VideoCapture", return_value=FakeCapture(["frame1", "frame2"]))
    @patch("ai_human_detected.detector.cv2.putText", side_effect=lambda frame, *args: frame)
    def test_webcam_loop_stops_on_q(
        self,
        _put_text,
        _mock_video_capture,
        _mock_imshow,
        _mock_wait_key,
        _mock_destroy,
    ):
        detector = PersonDetector(model=FakeModel(FakeResult(boxes=[object()])))

        frames = detector.run_webcam(camera_index=0, display=True)

        self.assertEqual(frames, 1)

    def test_rejects_confidence_outside_range(self):
        with self.assertRaises(ValueError):
            PersonDetector(confidence=1.1, model=FakeModel(FakeResult(boxes=[])))


if __name__ == "__main__":
    unittest.main()
