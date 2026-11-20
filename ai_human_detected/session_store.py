import csv
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path

from ai_human_detected.capacity import CapacityStatus

SAMPLE_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class SessionSummary:
    session_id: int
    started_at: datetime
    ended_at: datetime | None
    capacity_limit: int
    camera_index: int
    sample_count: int
    average_people: float | None
    peak_people: int
    near_capacity_seconds: float
    at_or_over_capacity_seconds: float
    observed_seconds: float
    schedule_id: int | None


@dataclass(frozen=True)
class WeeklySchedule:
    schedule_id: int | None
    name: str
    start_time: time
    end_time: time
    weekdays: tuple[int, ...]
    enabled: bool


def find_active_schedule(
    schedules: list[WeeklySchedule],
    current_time: datetime | None = None,
) -> WeeklySchedule | None:
    now = current_time or datetime.now().astimezone()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Schedule evaluation requires a timezone-aware timestamp.")
    local_now = now.astimezone()
    weekday = local_now.weekday()
    clock = local_now.time().replace(second=0, microsecond=0, tzinfo=None)
    return next(
        (
            schedule
            for schedule in schedules
            if schedule.enabled
            and weekday in schedule.weekdays
            and schedule.start_time <= clock < schedule.end_time
        ),
        None,
    )


def session_database_path() -> Path:
    if os.name == "nt":
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "AIHumanDetected" / "sessions.sqlite3"
    data_home = os.environ.get("XDG_DATA_HOME")
    base_path = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base_path / "ai-human-detected" / "sessions.sqlite3"


def _utc_timestamp(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamps must include timezone information.")
    return timestamp.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SessionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or session_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS monitoring_schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS schedule_weekdays (
                    schedule_id INTEGER NOT NULL REFERENCES monitoring_schedules(id)
                        ON DELETE CASCADE,
                    weekday INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
                    PRIMARY KEY (schedule_id, weekday)
                );
                CREATE TABLE IF NOT EXISTS monitoring_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    capacity_limit INTEGER NOT NULL CHECK (capacity_limit > 0),
                    camera_index INTEGER NOT NULL CHECK (camera_index >= 0),
                    schedule_id INTEGER REFERENCES monitoring_schedules(id)
                );
                CREATE TABLE IF NOT EXISTS occupancy_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES monitoring_sessions(id),
                    sampled_at TEXT NOT NULL,
                    people_count INTEGER NOT NULL CHECK (people_count >= 0),
                    capacity_limit INTEGER NOT NULL CHECK (capacity_limit > 0),
                    camera_index INTEGER NOT NULL CHECK (camera_index >= 0),
                    status TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_occupancy_samples_session_time
                    ON occupancy_samples(session_id, sampled_at);
                """
            )
            session_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(monitoring_sessions)"
                )
            }
            if "schedule_id" not in session_columns:
                connection.execute(
                    """
                    ALTER TABLE monitoring_sessions
                    ADD COLUMN schedule_id INTEGER REFERENCES monitoring_schedules(id)
                    """
                )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def start_session(
        self,
        capacity_limit: int,
        camera_index: int,
        started_at: datetime | None = None,
        schedule_id: int | None = None,
    ) -> int:
        if type(capacity_limit) is not int or capacity_limit <= 0:
            raise ValueError("Capacity limit must be a positive integer.")
        if type(camera_index) is not int or camera_index < 0:
            raise ValueError("Camera index must be a non-negative integer.")
        if schedule_id is not None and (
            type(schedule_id) is not int or schedule_id <= 0
        ):
            raise ValueError("Schedule ID must be a positive integer.")

        timestamp = _format_timestamp(_utc_timestamp(started_at))
        with self._connect() as connection:
            if schedule_id is not None:
                schedule = connection.execute(
                    "SELECT enabled FROM monitoring_schedules WHERE id = ?",
                    (schedule_id,),
                ).fetchone()
                if schedule is None or not schedule["enabled"]:
                    raise ValueError("Scheduled sessions require an enabled schedule.")
            cursor = connection.execute(
                """
                INSERT INTO monitoring_sessions
                    (started_at, capacity_limit, camera_index, schedule_id)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, capacity_limit, camera_index, schedule_id),
            )
            return int(cursor.lastrowid)

    def save_schedule(
        self,
        schedule_id: int | None,
        name: str,
        start_time: str,
        end_time: str,
        weekdays: tuple[int, ...],
        enabled: bool,
    ) -> WeeklySchedule:
        if schedule_id is not None and (
            type(schedule_id) is not int or schedule_id <= 0
        ):
            raise ValueError("Schedule ID must be a positive integer.")
        clean_name = name.strip() if isinstance(name, str) else ""
        if not clean_name:
            raise ValueError("Schedule name cannot be empty.")
        if len(clean_name) > 40:
            raise ValueError("Schedule name cannot exceed 40 characters.")
        try:
            start = datetime.strptime(start_time, "%H:%M").time()
            end = datetime.strptime(end_time, "%H:%M").time()
        except (TypeError, ValueError) as error:
            raise ValueError("Schedule times must use the HH:MM format.") from error
        if start.strftime("%H:%M") != start_time or end.strftime("%H:%M") != end_time:
            raise ValueError("Schedule times must use the HH:MM format.")
        if start >= end:
            raise ValueError("Schedule end time must be later than its start time.")
        if type(enabled) is not bool:
            raise ValueError("Schedule enabled value must be a boolean.")
        if not isinstance(weekdays, (tuple, list, set)) or not weekdays:
            raise ValueError("Select at least one weekday for the schedule.")
        if any(type(day) is not int or day < 0 or day > 6 for day in weekdays):
            raise ValueError("Weekdays must be integers from 0 (Monday) to 6 (Sunday).")
        selected_days = tuple(sorted(set(weekdays)))

        with self._connect() as connection:
            if schedule_id is not None:
                existing = connection.execute(
                    "SELECT id FROM monitoring_schedules WHERE id = ?",
                    (schedule_id,),
                ).fetchone()
                if existing is None:
                    raise ValueError(f"Schedule {schedule_id} does not exist.")

            if enabled:
                candidates = connection.execute(
                    """
                    SELECT id, start_time, end_time FROM monitoring_schedules
                    WHERE enabled = 1 AND id != COALESCE(?, -1)
                    """,
                    (schedule_id,),
                ).fetchall()
                for candidate in candidates:
                    candidate_start = datetime.strptime(
                        candidate["start_time"], "%H:%M"
                    ).time()
                    candidate_end = datetime.strptime(
                        candidate["end_time"], "%H:%M"
                    ).time()
                    overlaps = start < candidate_end and end > candidate_start
                    if not overlaps:
                        continue
                    candidate_days = {
                        row["weekday"]
                        for row in connection.execute(
                            """
                            SELECT weekday FROM schedule_weekdays
                            WHERE schedule_id = ?
                            """,
                            (candidate["id"],),
                        )
                    }
                    if candidate_days.intersection(selected_days):
                        raise ValueError(
                            f"Schedule overlaps enabled schedule {candidate['id']}."
                        )

            if schedule_id is None:
                cursor = connection.execute(
                    """
                    INSERT INTO monitoring_schedules
                        (name, start_time, end_time, enabled)
                    VALUES (?, ?, ?, ?)
                    """,
                    (clean_name, start_time, end_time, int(enabled)),
                )
                saved_id = int(cursor.lastrowid)
            else:
                connection.execute(
                    """
                    UPDATE monitoring_schedules
                    SET name = ?, start_time = ?, end_time = ?, enabled = ?
                    WHERE id = ?
                    """,
                    (clean_name, start_time, end_time, int(enabled), schedule_id),
                )
                saved_id = schedule_id

            connection.execute(
                "DELETE FROM schedule_weekdays WHERE schedule_id = ?",
                (saved_id,),
            )
            connection.executemany(
                "INSERT INTO schedule_weekdays (schedule_id, weekday) VALUES (?, ?)",
                ((saved_id, day) for day in selected_days),
            )

        return WeeklySchedule(
            saved_id,
            clean_name,
            start,
            end,
            selected_days,
            enabled,
        )

    def list_schedules(self) -> list[WeeklySchedule]:
        with self._connect() as connection:
            schedules = connection.execute(
                "SELECT * FROM monitoring_schedules ORDER BY id"
            ).fetchall()
            weekday_rows = connection.execute(
                "SELECT schedule_id, weekday FROM schedule_weekdays ORDER BY weekday"
            ).fetchall()

        weekdays_by_schedule: dict[int, list[int]] = {}
        for row in weekday_rows:
            weekdays_by_schedule.setdefault(row["schedule_id"], []).append(row["weekday"])
        return [
            WeeklySchedule(
                schedule_id=row["id"],
                name=row["name"],
                start_time=datetime.strptime(row["start_time"], "%H:%M").time(),
                end_time=datetime.strptime(row["end_time"], "%H:%M").time(),
                weekdays=tuple(weekdays_by_schedule.get(row["id"], [])),
                enabled=bool(row["enabled"]),
            )
            for row in schedules
        ]

    def delete_schedule(self, schedule_id: int) -> None:
        if type(schedule_id) is not int or schedule_id <= 0:
            raise ValueError("Schedule ID must be a positive integer.")
        with self._connect() as connection:
            session_count = connection.execute(
                "SELECT COUNT(*) FROM monitoring_sessions WHERE schedule_id = ?",
                (schedule_id,),
            ).fetchone()[0]
            if session_count:
                raise ValueError(
                    "This schedule has session history; disable it instead of deleting it."
                )
            cursor = connection.execute(
                "DELETE FROM monitoring_schedules WHERE id = ?",
                (schedule_id,),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Schedule {schedule_id} does not exist.")

    def record_sample(
        self,
        session_id: int,
        people_count: int,
        capacity_limit: int,
        camera_index: int,
        status: CapacityStatus,
        sampled_at: datetime | None = None,
    ) -> None:
        if type(session_id) is not int or session_id <= 0:
            raise ValueError("Session ID must be a positive integer.")
        if type(people_count) is not int or people_count < 0:
            raise ValueError("People count must be a non-negative integer.")
        if type(capacity_limit) is not int or capacity_limit <= 0:
            raise ValueError("Capacity limit must be a positive integer.")
        if type(camera_index) is not int or camera_index < 0:
            raise ValueError("Camera index must be a non-negative integer.")
        if not isinstance(status, CapacityStatus):
            raise ValueError("Status must be a CapacityStatus value.")

        timestamp = _format_timestamp(_utc_timestamp(sampled_at))
        with self._connect() as connection:
            session = connection.execute(
                "SELECT started_at, ended_at FROM monitoring_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError(f"Monitoring session {session_id} does not exist.")
            if session["ended_at"] is not None:
                raise ValueError(f"Monitoring session {session_id} has ended.")
            if timestamp < session["started_at"]:
                raise ValueError("Sample timestamp cannot be before the session start.")

            previous = connection.execute(
                """
                SELECT sampled_at FROM occupancy_samples
                WHERE session_id = ?
                ORDER BY sampled_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if previous is not None and timestamp < previous["sampled_at"]:
                raise ValueError("Sample timestamps must be in chronological order.")

            connection.execute(
                """
                INSERT INTO occupancy_samples
                    (session_id, sampled_at, people_count, capacity_limit,
                     camera_index, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    timestamp,
                    people_count,
                    capacity_limit,
                    camera_index,
                    status.value,
                ),
            )

    def end_session(
        self,
        session_id: int,
        ended_at: datetime | None = None,
    ) -> SessionSummary:
        timestamp = _utc_timestamp(ended_at)
        with self._connect() as connection:
            session = connection.execute(
                "SELECT started_at, ended_at FROM monitoring_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError(f"Monitoring session {session_id} does not exist.")
            if session["ended_at"] is not None:
                raise ValueError(f"Monitoring session {session_id} has already ended.")
            if timestamp < _parse_timestamp(session["started_at"]):
                raise ValueError("Session end time cannot be before its start time.")
            last_sample = connection.execute(
                """
                SELECT sampled_at FROM occupancy_samples
                WHERE session_id = ?
                ORDER BY sampled_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if (
                last_sample is not None
                and timestamp < _parse_timestamp(last_sample["sampled_at"])
            ):
                raise ValueError("Session end time cannot be before its last sample.")

            connection.execute(
                "UPDATE monitoring_sessions SET ended_at = ? WHERE id = ?",
                (_format_timestamp(timestamp), session_id),
            )
        return self.get_summary(session_id)

    def get_summary(
        self,
        session_id: int,
        as_of: datetime | None = None,
    ) -> SessionSummary:
        current_time = _utc_timestamp(as_of)
        with self._connect() as connection:
            session = connection.execute(
                "SELECT * FROM monitoring_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError(f"Monitoring session {session_id} does not exist.")
            samples = connection.execute(
                """
                SELECT sampled_at, people_count, status
                FROM occupancy_samples
                WHERE session_id = ?
                ORDER BY sampled_at, id
                """,
                (session_id,),
            ).fetchall()

        started_at = _parse_timestamp(session["started_at"])
        ended_at = (
            _parse_timestamp(session["ended_at"])
            if session["ended_at"] is not None
            else None
        )
        summary_end = ended_at or current_time
        weighted_people = 0.0
        observed_seconds = 0.0
        near_capacity_seconds = 0.0
        at_or_over_capacity_seconds = 0.0

        for index, sample in enumerate(samples):
            sample_time = _parse_timestamp(sample["sampled_at"])
            next_time = (
                _parse_timestamp(samples[index + 1]["sampled_at"])
                if index + 1 < len(samples)
                else summary_end
            )
            interval = max(0.0, (next_time - sample_time).total_seconds())
            duration = min(interval, SAMPLE_INTERVAL_SECONDS)
            observed_seconds += duration
            weighted_people += sample["people_count"] * duration
            if sample["status"] == CapacityStatus.NEAR_CAPACITY.value:
                near_capacity_seconds += duration
            elif sample["status"] in (
                CapacityStatus.AT_CAPACITY.value,
                CapacityStatus.OVER_CAPACITY.value,
            ):
                at_or_over_capacity_seconds += duration

        average_people = (
            weighted_people / observed_seconds if observed_seconds else None
        )
        return SessionSummary(
            session_id=session_id,
            started_at=started_at,
            ended_at=ended_at,
            capacity_limit=session["capacity_limit"],
            camera_index=session["camera_index"],
            sample_count=len(samples),
            average_people=average_people,
            peak_people=max((sample["people_count"] for sample in samples), default=0),
            near_capacity_seconds=near_capacity_seconds,
            at_or_over_capacity_seconds=at_or_over_capacity_seconds,
            observed_seconds=observed_seconds,
            schedule_id=session["schedule_id"],
        )

    def list_sessions(self, limit: int = 20) -> list[SessionSummary]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("Session limit must be a positive integer.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM monitoring_sessions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get_summary(row["id"]) for row in rows]

    def export_csv(self, path: Path | None = None) -> Path:
        export_path = path or self.path.with_name("session-reports.csv")
        summaries = self.list_sessions(limit=2**31 - 1)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "session_id",
                    "started_at",
                    "ended_at",
                    "capacity_limit",
                    "camera_index",
                    "sample_count",
                    "average_people",
                    "peak_people",
                    "near_capacity_seconds",
                    "at_or_over_capacity_seconds",
                    "observed_seconds",
                    "schedule_id",
                )
            )
            for summary in reversed(summaries):
                writer.writerow(
                    (
                        summary.session_id,
                        _format_timestamp(summary.started_at),
                        _format_timestamp(summary.ended_at)
                        if summary.ended_at
                        else "",
                        summary.capacity_limit,
                        summary.camera_index,
                        summary.sample_count,
                        ""
                        if summary.average_people is None
                        else f"{summary.average_people:.2f}",
                        summary.peak_people,
                        f"{summary.near_capacity_seconds:.2f}",
                        f"{summary.at_or_over_capacity_seconds:.2f}",
                        f"{summary.observed_seconds:.2f}",
                        ""
                        if summary.schedule_id is None
                        else summary.schedule_id,
                    )
                )
        return export_path
