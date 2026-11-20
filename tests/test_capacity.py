import unittest

from ai_human_detected.capacity import (
    CapacityMonitor,
    CapacityStatus,
    get_capacity_status,
)


class CapacityStatusTests(unittest.TestCase):
    def test_counts_below_near_capacity_threshold_as_normal(self):
        self.assertEqual(get_capacity_status(23, 30), CapacityStatus.NORMAL)

    def test_eighty_percent_threshold_is_near_capacity(self):
        self.assertEqual(get_capacity_status(24, 30), CapacityStatus.NEAR_CAPACITY)

    def test_exact_capacity_is_reported(self):
        self.assertEqual(get_capacity_status(30, 30), CapacityStatus.AT_CAPACITY)

    def test_count_over_capacity_is_reported(self):
        self.assertEqual(get_capacity_status(31, 30), CapacityStatus.OVER_CAPACITY)

    def test_rejects_invalid_values(self):
        for people_count, capacity_limit in ((-1, 30), (1.5, 30), (0, 0), (0, 1.5)):
            with self.subTest(people_count=people_count, capacity_limit=capacity_limit):
                with self.assertRaises(ValueError):
                    get_capacity_status(people_count, capacity_limit)

    def test_monitor_smooths_short_count_spikes(self):
        monitor = CapacityMonitor(30)

        smoothed, status = monitor.update(30)
        smoothed, status = monitor.update(0)
        smoothed, status = monitor.update(0)

        self.assertEqual(smoothed, 0)
        self.assertEqual(status, CapacityStatus.NORMAL)

    def test_monitor_requires_consecutive_frames_before_changing_status(self):
        monitor = CapacityMonitor(30)

        smoothed, status = monitor.update(25)
        self.assertEqual(status, CapacityStatus.NORMAL)

        smoothed, status = monitor.update(25)
        self.assertEqual(status, CapacityStatus.NORMAL)

        smoothed, status = monitor.update(25)
        self.assertEqual(smoothed, 25)
        self.assertEqual(status, CapacityStatus.NEAR_CAPACITY)

    def test_reset_clears_samples_and_uses_new_capacity(self):
        monitor = CapacityMonitor(30)
        for _ in range(5):
            monitor.update(25)

        monitor.reset(50)

        self.assertEqual(monitor.smoothed_count, 0)
        self.assertEqual(monitor.status, CapacityStatus.NORMAL)
        self.assertEqual(monitor.capacity_limit, 50)


if __name__ == "__main__":
    unittest.main()
