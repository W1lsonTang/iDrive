"""Virtual ECG sensor that replays EDF data in real time."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .edf_loader import derive_annotation_path, load_button_onsets, load_ecg_signal
except ImportError:  # Allow direct execution imports from sibling files.
    from edf_loader import derive_annotation_path, load_button_onsets, load_ecg_signal


class VirtualECGSensor:
    """Replay ECG from EDF using wall-clock timing."""

    def __init__(
        self,
        edf_path: str,
        annotation_path: Optional[str] = None,
        playback_speed: float = 1.0,
        buffer_seconds: float = 60.0,
        alert_duration_sec: float = 600.0,
        drowsy_lead_sec: float = 60.0,
    ) -> None:
        self.edf_path = Path(edf_path)
        if not self.edf_path.is_file():
            raise FileNotFoundError(f"EDF file not found: {self.edf_path}")

        self.annotation_path = Path(annotation_path) if annotation_path else derive_annotation_path(self.edf_path)
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"Annotation EDF file not found: {self.annotation_path}")

        self.ecg_signal, self.sampling_rate, self.channel_name = load_ecg_signal(self.edf_path)
        self.button_onsets = load_button_onsets(self.annotation_path)
        self.duration_sec = len(self.ecg_signal) / self.sampling_rate

        self.playback_speed = float(playback_speed)
        self.alert_duration_sec = float(alert_duration_sec)
        self.drowsy_lead_sec = float(drowsy_lead_sec)

        self.buffer_samples = int(round(buffer_seconds * self.sampling_rate))
        self._ring = deque(maxlen=self.buffer_samples)

        self._playback_anchor_sec = 0.0
        self._wall_anchor_sec = time.perf_counter()
        self._paused = True
        self._last_read_idx = 0

    def start(self, reset: bool = False) -> None:
        if reset:
            self.reset()
        self._paused = False
        self._wall_anchor_sec = time.perf_counter()

    def pause(self) -> None:
        if self._paused:
            return
        self._playback_anchor_sec = self.get_current_time_sec()
        self._paused = True

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        self._wall_anchor_sec = time.perf_counter()

    def toggle_pause(self) -> None:
        if self._paused:
            self.resume()
        else:
            self.pause()

    def set_playback_speed(self, speed: float) -> None:
        speed = max(0.05, float(speed))
        current_t = self.get_current_time_sec()
        self.playback_speed = speed
        self._playback_anchor_sec = current_t
        self._wall_anchor_sec = time.perf_counter()

    def reset(self) -> None:
        self._playback_anchor_sec = 0.0
        self._wall_anchor_sec = time.perf_counter()
        self._last_read_idx = 0
        self._ring.clear()

    def seek(self, t_sec: float) -> None:
        t_sec = min(max(0.0, float(t_sec)), self.duration_sec)
        self._playback_anchor_sec = t_sec
        self._wall_anchor_sec = time.perf_counter()
        current_idx = int(round(t_sec * self.sampling_rate))
        self._last_read_idx = current_idx
        self._ring.clear()
        start_idx = max(0, current_idx - self.buffer_samples)
        for value in self.ecg_signal[start_idx:current_idx]:
            self._ring.append(float(value))

    def is_paused(self) -> bool:
        return self._paused

    def get_current_time_sec(self) -> float:
        if self._paused:
            return min(self._playback_anchor_sec, self.duration_sec)
        elapsed = time.perf_counter() - self._wall_anchor_sec
        now_t = self._playback_anchor_sec + elapsed * self.playback_speed
        if now_t >= self.duration_sec:
            self.pause()
            self._playback_anchor_sec = self.duration_sec
            return self.duration_sec
        return now_t

    def read_since_last(self) -> np.ndarray:
        current_t = self.get_current_time_sec()
        current_idx = int(round(current_t * self.sampling_rate))
        current_idx = min(current_idx, len(self.ecg_signal))
        if current_idx <= self._last_read_idx:
            return np.empty(0, dtype=np.float64)

        segment = self.ecg_signal[self._last_read_idx : current_idx].astype(np.float64)
        for value in segment:
            self._ring.append(float(value))
        self._last_read_idx = current_idx
        return segment

    def get_recent_waveform(self, seconds: float = 10.0) -> np.ndarray:
        n_samples = int(round(seconds * self.sampling_rate))
        n_samples = max(1, n_samples)
        current_idx = int(round(self.get_current_time_sec() * self.sampling_rate))
        start_idx = max(0, current_idx - n_samples)
        return self.ecg_signal[start_idx:current_idx].astype(np.float64)

    def get_hrv_buffer(self) -> Optional[np.ndarray]:
        if len(self._ring) < self.buffer_samples:
            return None
        return np.asarray(self._ring, dtype=np.float64)

    def get_ground_truth(self, t_sec: Optional[float] = None) -> int:
        if t_sec is None:
            t_sec = self.get_current_time_sec()

        # Drowsy has priority when windows overlap alert interval.
        for ts in self.button_onsets:
            if ts - self.drowsy_lead_sec <= t_sec <= ts:
                return 1
        if 0.0 <= t_sec <= self.alert_duration_sec:
            return 0
        return -1

    def get_progress_ratio(self) -> float:
        if self.duration_sec <= 0:
            return 0.0
        return min(max(self.get_current_time_sec() / self.duration_sec, 0.0), 1.0)
