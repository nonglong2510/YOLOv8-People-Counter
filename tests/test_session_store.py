import csv
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_human_detected.capacity import CapacityStatus
from ai_human_detected.session_store import (
    SessionStore,
    find_active_schedule,
)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions.sqlite3")
        self.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_saves_and_summarizes_time_weighted_session_samples(self):
        session_id = self.store.start_session(30, 0, self.started_at)
        samples = (
            (10, CapacityStatus.NORMAL),
            (25, CapacityStatus.NEAR_CAPACITY),
            (30, CapacityStatus.AT_CAPACITY),
            (31, CapacityStatus.OVER_CAPACITY),
        )
        for index, (people_count, status) in enumerate(samples):
            self.store.record_sample(
                session_id,
                people_count,
                30,
                0,
                status,
                self.started_at + timedelta(seconds=index),
            )

        summary = self.store.end_session(
            session_id,
            self.started_at + timedelta(seconds=4),
        )

        self.assertEqual(summary.sample_count, 4)
        self.assertEqual(summary.average_people, 24)
        self.assertEqual(summary.peak_people, 31)
        self.assertEqual(summary.near_capacity_seconds, 1)
        self.assertEqual(summary.at_or_over_capacity_seconds, 2)
        self.assertEqual(summary.observed_seconds, 4)

    def test_rejects_samples_after_session_has_ended(self):
        session_id = self.store.start_session(30, 0, self.started_at)
        self.store.end_session(session_id, self.started_at + timedelta(seconds=1))

        with self.assertRaisesRegex(ValueError, "has ended"):
            self.store.record_sample(
                session_id,
                1,
                30,
                0,
                CapacityStatus.NORMAL,
                self.started_at + timedelta(seconds=2),
            )

    def test_rejects_out_of_order_samples(self):
        session_id = self.store.start_session(30, 0, self.started_at)
        self.store.record_sample(
            session_id,
            1,
            30,
            0,
            CapacityStatus.NORMAL,
            self.started_at + timedelta(seconds=2),
        )

        with self.assertRaisesRegex(ValueError, "chronological"):
            self.store.record_sample(
                session_id,
                1,
                30,
                0,
                CapacityStatus.NORMAL,
                self.started_at + timedelta(seconds=1),
            )

    def test_rejects_samples_before_session_start(self):
        session_id = self.store.start_session(30, 0, self.started_at)

        with self.assertRaisesRegex(ValueError, "before the session start"):
            self.store.record_sample(
                session_id,
                1,
                30,
                0,
                CapacityStatus.NORMAL,
                self.started_at - timedelta(seconds=1),
            )

    def test_exports_session_summary_csv(self):
        session_id = self.store.start_session(30, 0, self.started_at)
        self.store.record_sample(
            session_id,
            8,
            30,
            0,
            CapacityStatus.NORMAL,
            self.started_at,
        )
        self.store.end_session(session_id, self.started_at + timedelta(seconds=1))
        export_path = Path(self.temp_dir.name) / "reports" / "sessions.csv"

        result = self.store.export_csv(export_path)

        self.assertEqual(result, export_path)
        with export_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["peak_people"], "8")
        self.assertEqual(rows[0]["sample_count"], "1")

    def test_schedule_matches_selected_weekday_and_half_open_time_range(self):
        schedule = self.store.save_schedule(
            None,
            "Morning class",
            "07:00",
            "09:30",
            (0, 2, 4),
            True,
        )

        self.assertEqual(
            find_active_schedule(
                [schedule],
                datetime(2026, 1, 5, 7, 0).astimezone(),
            ),
            schedule,
        )
        self.assertIsNone(
            find_active_schedule(
                [schedule],
                datetime(2026, 1, 6, 8, 0).astimezone(),
            )
        )
        self.assertIsNone(
            find_active_schedule(
                [schedule],
                datetime(2026, 1, 5, 9, 30).astimezone(),
            )
        )

    def test_enabled_schedules_cannot_overlap_on_the_same_weekday(self):
        self.store.save_schedule(None, "First", "07:00", "09:30", (0,), True)

        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.store.save_schedule(None, "Second", "09:00", "10:00", (0,), True)

    def test_scheduled_session_is_linked_and_schedule_history_is_preserved(self):
        schedule = self.store.save_schedule(
            None,
            "Morning class",
            "07:00",
            "09:30",
            (0,),
            True,
        )
        session_id = self.store.start_session(
            30,
            0,
            self.started_at,
            schedule_id=schedule.schedule_id,
        )
        summary = self.store.end_session(session_id, self.started_at + timedelta(seconds=1))

        self.assertEqual(summary.schedule_id, schedule.schedule_id)
        with self.assertRaisesRegex(ValueError, "session history"):
            self.store.delete_schedule(schedule.schedule_id)

    def test_existing_database_is_migrated_with_schedule_support(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute(
                """
                CREATE TABLE monitoring_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    capacity_limit INTEGER NOT NULL,
                    camera_index INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE occupancy_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES monitoring_sessions(id),
                    sampled_at TEXT NOT NULL,
                    people_count INTEGER NOT NULL,
                    capacity_limit INTEGER NOT NULL,
                    camera_index INTEGER NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO monitoring_sessions
                    (started_at, ended_at, capacity_limit, camera_index)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self.started_at.isoformat(timespec="microseconds"),
                    (self.started_at + timedelta(seconds=1)).isoformat(
                        timespec="microseconds"
                    ),
                    30,
                    0,
                ),
            )
        connection.close()

        migrated_store = SessionStore(legacy_path)
        summary = migrated_store.get_summary(1)

        self.assertIsNone(summary.schedule_id)
        self.assertEqual(summary.capacity_limit, 30)


if __name__ == "__main__":
    unittest.main()
