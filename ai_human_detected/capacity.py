from collections import deque
from enum import Enum
from statistics import median


class CapacityStatus(Enum):
    NORMAL = "NORMAL"
    NEAR_CAPACITY = "NEAR CAPACITY"
    AT_CAPACITY = "AT CAPACITY"
    OVER_CAPACITY = "OVER CAPACITY"


class CapacityMonitor:
    def __init__(
        self,
        capacity_limit: int,
        window_size: int = 5,
        confirmation_frames: int = 3,
    ) -> None:
        if type(window_size) is not int or window_size <= 0 or window_size % 2 == 0:
            raise ValueError("Window size must be a positive odd integer.")
        if type(confirmation_frames) is not int or confirmation_frames <= 0:
            raise ValueError("Confirmation frames must be a positive integer.")

        get_capacity_status(0, capacity_limit)
        self.capacity_limit = capacity_limit
        self.window_size = window_size
        self.confirmation_frames = confirmation_frames
        self._counts: deque[int] = deque(maxlen=window_size)
        self._status = CapacityStatus.NORMAL
        self._pending_status: CapacityStatus | None = None
        self._pending_frames = 0

    @property
    def smoothed_count(self) -> int:
        return int(median(self._counts)) if self._counts else 0

    @property
    def status(self) -> CapacityStatus:
        return self._status

    def reset(self, capacity_limit: int | None = None) -> None:
        if capacity_limit is not None:
            get_capacity_status(0, capacity_limit)
            self.capacity_limit = capacity_limit
        self._counts.clear()
        self._status = CapacityStatus.NORMAL
        self._pending_status = None
        self._pending_frames = 0

    def update(self, people_count: int) -> tuple[int, CapacityStatus]:
        get_capacity_status(people_count, self.capacity_limit)
        self._counts.append(people_count)
        candidate = get_capacity_status(self.smoothed_count, self.capacity_limit)

        if candidate is self._status:
            self._pending_status = None
            self._pending_frames = 0
        elif candidate is self._pending_status:
            self._pending_frames += 1
        else:
            self._pending_status = candidate
            self._pending_frames = 1

        if self._pending_frames >= self.confirmation_frames:
            self._status = candidate
            self._pending_status = None
            self._pending_frames = 0

        return self.smoothed_count, self._status


def get_capacity_status(people_count: int, capacity_limit: int) -> CapacityStatus:
    if type(people_count) is not int or people_count < 0:
        raise ValueError("People count must be a non-negative integer.")
    if type(capacity_limit) is not int or capacity_limit <= 0:
        raise ValueError("Capacity limit must be a positive integer.")

    near_limit = (capacity_limit * 80 + 99) // 100
    if people_count > capacity_limit:
        return CapacityStatus.OVER_CAPACITY
    if people_count == capacity_limit:
        return CapacityStatus.AT_CAPACITY
    if people_count >= near_limit:
        return CapacityStatus.NEAR_CAPACITY
    return CapacityStatus.NORMAL
