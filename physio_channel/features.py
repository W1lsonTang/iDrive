"""HRV feature extraction utilities for ECG windows."""

from __future__ import annotations

from typing import Dict, Optional

import neurokit2 as nk
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "mean_rr_ms",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
]


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return float(np.asarray(value).squeeze())


def _extract_first(df: pd.DataFrame, column: str, fallback: float = float("nan")) -> float:
    if column not in df.columns:
        return fallback
    return _safe_float(df.iloc[0][column])


def extract_hrv_features(
    ecg_window: np.ndarray,
    sampling_rate: float = 128.0,
    min_rpeaks: int = 20,
) -> Optional[Dict[str, float]]:
    """Extract robust HRV features from one ECG window.

    Returns None when ECG quality is too poor for stable HRV estimation.
    """
    if ecg_window.size < int(sampling_rate * 10):
        return None

    try:
        cleaned = nk.ecg_clean(ecg_window, sampling_rate=sampling_rate)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    except Exception:
        return None

    rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    if rpeaks.size < min_rpeaks:
        return None

    rr_ms = np.diff(rpeaks) / sampling_rate * 1000.0
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    if rr_ms.size < min_rpeaks - 1:
        return None

    # NeuroKit HRV APIs expect a binary peak series in dataframe form.
    peak_signal = np.zeros(ecg_window.shape[0], dtype=np.int32)
    peak_signal[rpeaks] = 1
    peaks_df = pd.DataFrame({"ECG_R_Peaks": peak_signal})

    try:
        hrv_time = nk.hrv_time(peaks_df, sampling_rate=sampling_rate, show=False)
        hrv_freq = nk.hrv_frequency(peaks_df, sampling_rate=sampling_rate, show=False)
    except Exception:
        return None

    mean_rr = _extract_first(hrv_time, "HRV_MeanNN", fallback=float(np.mean(rr_ms)))
    sdnn = _extract_first(hrv_time, "HRV_SDNN")
    rmssd = _extract_first(hrv_time, "HRV_RMSSD")
    pnn50 = _extract_first(hrv_time, "HRV_pNN50")
    lf_power = _extract_first(hrv_freq, "HRV_LF")
    hf_power = _extract_first(hrv_freq, "HRV_HF")
    lf_hf = _extract_first(hrv_freq, "HRV_LFHF")

    features = {
        "mean_rr_ms": mean_rr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf,
        "heart_rate_bpm": 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else float("nan"),
    }

    values = np.array([features[col] for col in FEATURE_COLUMNS], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        return None
    return features


def feature_vector_from_dict(feature_dict: Dict[str, float]) -> np.ndarray:
    """Convert a feature dictionary into fixed-order ndarray."""
    return np.array([feature_dict[name] for name in FEATURE_COLUMNS], dtype=np.float64)
