"""EDF loading and labeling utilities for DD-Database ECG windows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import mne
import numpy as np

RECORDING_PATTERN = re.compile(r"(?P<subject>\d{2}[MF])_(?P<trial>\d)\.edf$", re.IGNORECASE)
ECG_CHANNEL_PATTERN = re.compile(r"(?:^|[^A-Z])(ECG|EKG)(?:[^A-Z]|$)", re.IGNORECASE)


def parse_subject_trial_from_path(path: Path) -> Tuple[str, int]:
    """Parse subject and trial identifiers from DD-Database filename."""
    match = RECORDING_PATTERN.search(path.name)
    if not match:
        return "unknown", -1
    return match.group("subject").upper(), int(match.group("trial"))


def derive_annotation_path(recording_path: Path) -> Path:
    """Infer annotation EDF path from a recording EDF path."""
    stem = recording_path.stem
    return recording_path.with_name(f"{stem}_annotations.edf")


def is_annotation_file(path: Path) -> bool:
    """Return True when the EDF file is an annotation sidecar."""
    return path.stem.lower().endswith("_annotations")


def iter_recording_pairs(data_dir: Path) -> Iterator[Tuple[Path, Path]]:
    """Yield (recording, annotation) pairs from DD-Database folder."""
    for edf_path in sorted(data_dir.glob("*.edf")):
        if is_annotation_file(edf_path):
            continue
        ann_path = derive_annotation_path(edf_path)
        if ann_path.is_file():
            yield edf_path, ann_path
        else:
            print(f"[WARN] Missing annotation file for recording: {edf_path.name}")


def find_ecg_channel_name(channel_names: Sequence[str]) -> str:
    """Resolve ECG channel name from EDF channel list."""
    for name in channel_names:
        if ECG_CHANNEL_PATTERN.search(name):
            return name
    raise ValueError(
        "Unable to find ECG/EKG channel in EDF file. "
        f"Available channels: {list(channel_names)}"
    )


def load_ecg_signal(recording_path: Path) -> Tuple[np.ndarray, float, str]:
    """Load ECG signal from EDF and return (signal, sampling_rate, channel_name)."""
    raw = mne.io.read_raw_edf(str(recording_path), preload=True, verbose="ERROR")
    channel_name = find_ecg_channel_name(raw.ch_names)
    ecg_signal = raw.get_data(picks=[channel_name])[0].astype(np.float64)
    sampling_rate = float(raw.info["sfreq"])
    return ecg_signal, sampling_rate, channel_name


def load_button_onsets(annotation_path: Path) -> np.ndarray:
    """Load drowsiness button timestamps (seconds) from annotation EDF."""
    annotations = mne.read_annotations(str(annotation_path))
    onsets = np.asarray(annotations.onset, dtype=np.float64)
    if onsets.size == 0:
        print(f"[WARN] No button events found in annotation file: {annotation_path.name}")
    return np.sort(onsets)


def _build_drowsy_intervals(
    button_onsets: np.ndarray, drowsy_lead_sec: float
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    for ts in button_onsets:
        intervals.append((max(0.0, ts - drowsy_lead_sec), float(ts)))
    return intervals


def _window_overlaps_any(
    start_sec: float, end_sec: float, intervals: Sequence[Tuple[float, float]]
) -> bool:
    for a_start, a_end in intervals:
        overlap = min(end_sec, a_end) - max(start_sec, a_start)
        if overlap > 0.0:
            return True
    return False


def label_window(
    start_sec: float,
    end_sec: float,
    button_onsets: np.ndarray,
    alert_duration_sec: float = 600.0,
    drowsy_lead_sec: float = 60.0,
) -> Optional[int]:
    """Assign label for one window.

    Priority:
    1) Drowsy (overlap with any button-preceding drowsy interval)
    2) Alert (entirely inside first alert_duration_sec)
    3) Gray-zone (None)
    """
    drowsy_intervals = _build_drowsy_intervals(button_onsets, drowsy_lead_sec)
    if _window_overlaps_any(start_sec, end_sec, drowsy_intervals):
        return 1
    if end_sec <= alert_duration_sec:
        return 0
    return None


def generate_labeled_windows(
    recording_path: Path,
    annotation_path: Optional[Path] = None,
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    alert_duration_sec: float = 600.0,
    drowsy_lead_sec: float = 60.0,
    drop_gray_zone: bool = True,
) -> List[Dict[str, object]]:
    """Generate ECG windows with labels for one recording."""
    if annotation_path is None:
        annotation_path = derive_annotation_path(recording_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation EDF not found: {annotation_path}")

    ecg_signal, sampling_rate, channel_name = load_ecg_signal(recording_path)
    button_onsets = load_button_onsets(annotation_path)
    subject, trial = parse_subject_trial_from_path(recording_path)

    window_size = int(round(window_sec * sampling_rate))
    step_size = int(round(step_sec * sampling_rate))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_sec and step_sec must create positive sample counts.")
    if len(ecg_signal) < window_size:
        return []

    samples: List[Dict[str, object]] = []
    for start_idx in range(0, len(ecg_signal) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        start_sec = start_idx / sampling_rate
        end_sec = end_idx / sampling_rate
        label = label_window(
            start_sec=start_sec,
            end_sec=end_sec,
            button_onsets=button_onsets,
            alert_duration_sec=alert_duration_sec,
            drowsy_lead_sec=drowsy_lead_sec,
        )
        if label is None and drop_gray_zone:
            continue

        samples.append(
            {
                "ecg": ecg_signal[start_idx:end_idx].copy(),
                "label": label,
                "subject": subject,
                "trial": trial,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "sampling_rate": sampling_rate,
                "channel_name": channel_name,
                "recording_path": str(recording_path),
                "annotation_path": str(annotation_path),
            }
        )
    return samples
