"""Thread-safe time-aligned buffers for channel outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Optional


@dataclass
class ChannelSample:
    """Single channel output with quality flag."""

    timestamp: float
    probability: float
    quality_ok: bool


class ChannelBuffer:
    """Rolling buffer storing the most recent ChannelSample entries."""

    def __init__(self, maxlen: int = 240, freshness_sec: float = 2.0) -> None:
        self._buffer: Deque[ChannelSample] = deque(maxlen=maxlen)
        self._lock = Lock()
        self.freshness_sec = float(freshness_sec)

    def push(self, sample: ChannelSample) -> None:
        with self._lock:
            self._buffer.append(sample)

    def latest(self, now_ts: Optional[float] = None) -> Optional[ChannelSample]:
        """Return the most recent sample if still considered fresh."""
        with self._lock:
            if not self._buffer:
                return None
            sample = self._buffer[-1]
            if now_ts is None:
                return sample
            if (now_ts - sample.timestamp) > self.freshness_sec:
                return None
            return sample

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
