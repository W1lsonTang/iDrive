from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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


@dataclass
class SessionItem:
    subject: str
    session: str
    ibi_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract UL-DD HRV features from IBI csv files.")
    parser.add_argument("--uldd-dir", default="UL-DD", help="Path to UL-DD root directory.")
    parser.add_argument(
        "--output-csv",
        default="output/uldd_physio_series.csv",
        help="Output CSV path for extracted HRV windows.",
    )
    parser.add_argument("--window-sec", type=float, default=60.0, help="HRV window length in seconds.")
    parser.add_argument("--step-sec", type=float, default=30.0, help="Sliding step in seconds.")
    parser.add_argument("--min-rr-count", type=int, default=30, help="Minimum RR intervals per window.")
    parser.add_argument("--max-sessions", type=int, default=0, help="Optional limit for smoke tests (0 = all).")
    parser.add_argument("--log-every", type=int, default=1, help="Log progress every N sessions.")
    return parser.parse_args()


def iter_session_items(uldd_dir: Path) -> Iterator[SessionItem]:
    base = uldd_dir / "CSV_Files" / "CSV_Files"
    if not base.is_dir():
        raise FileNotFoundError(f"UL-DD csv directory not found: {base}")

    for subject_dir in sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name):
        subject = subject_dir.name
        for session_dir in sorted([d for d in subject_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
            session = session_dir.name
            ibi_path = session_dir / f"{subject}_IBI_{session}.csv"
            label_path = session_dir / f"{subject}_Labels_{session}.csv"
            if not ibi_path.is_file() or not label_path.is_file():
                continue
            yield SessionItem(subject=subject, session=session, ibi_path=ibi_path, label_path=label_path)


def load_session_duration_sec(label_path: Path) -> float:
    row = pd.read_csv(label_path, header=None).iloc[0].to_numpy(dtype=np.float64)
    return float(len(row) * 60.0)


def _extract_first(df: pd.DataFrame, column: str, fallback: float = float("nan")) -> float:
    if column not in df.columns:
        return fallback
    value = df.iloc[0][column]
    if value is None:
        return fallback
    return float(np.asarray(value).squeeze())


def extract_hrv_from_rr(rr_ms: np.ndarray, rr_times_sec: np.ndarray) -> dict[str, float] | None:
    if rr_ms.size < 3:
        return None

    rr_ms = rr_ms.astype(np.float64)
    rr_times_sec = rr_times_sec.astype(np.float64)

    peaks = {"RRI": rr_ms, "RRI_Time": rr_times_sec}
    try:
        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False, silent=True)
    except Exception:
        return None

    mean_rr = _extract_first(hrv_time, "HRV_MeanNN", fallback=float(np.mean(rr_ms)))
    sdnn = _extract_first(hrv_time, "HRV_SDNN")
    rmssd = _extract_first(hrv_time, "HRV_RMSSD")
    pnn50 = _extract_first(hrv_time, "HRV_pNN50")
    lf = _extract_first(hrv_freq, "HRV_LF")
    hf = _extract_first(hrv_freq, "HRV_HF")
    lf_hf = _extract_first(hrv_freq, "HRV_LFHF")

    feats = {
        "mean_rr_ms": mean_rr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": lf_hf,
        "heart_rate_bpm": 60000.0 / mean_rr if np.isfinite(mean_rr) and mean_rr > 0 else float("nan"),
    }

    arr = np.array([feats[c] for c in FEATURE_COLUMNS], dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        return None
    return feats


def extract_session_windows(
    item: SessionItem,
    window_sec: float,
    step_sec: float,
    min_rr_count: int,
) -> pd.DataFrame:
    ibi = pd.read_csv(item.ibi_path, header=None, names=["timestamp_sec", "rr_sec"])
    ibi = ibi.replace([np.inf, -np.inf], np.nan).dropna()
    ibi = ibi[(ibi["rr_sec"] > 0.0) & (ibi["rr_sec"] < 3.0)]
    if ibi.empty:
        return pd.DataFrame()

    t0 = float(ibi["timestamp_sec"].iloc[0])
    ibi["t_rel_sec"] = ibi["timestamp_sec"] - t0
    duration_sec = load_session_duration_sec(item.label_path)

    rows: list[dict[str, float | int | str]] = []
    start = 0.0
    while start + window_sec <= duration_sec + 1e-9:
        end = start + window_sec
        center = start + window_sec / 2.0
        sel = ibi[(ibi["t_rel_sec"] >= start) & (ibi["t_rel_sec"] < end)]
        rr_sec = sel["rr_sec"].to_numpy(dtype=np.float64)
        if rr_sec.size < min_rr_count:
            start += step_sec
            continue

        rr_ms = rr_sec * 1000.0
        rr_times = sel["t_rel_sec"].to_numpy(dtype=np.float64)
        feats = extract_hrv_from_rr(rr_ms=rr_ms, rr_times_sec=rr_times)
        if feats is None:
            start += step_sec
            continue

        row = {
            "subject": item.subject,
            "session": item.session,
            "window_start_sec": float(start),
            "window_end_sec": float(end),
            "t_sec": float(center),
            "rr_count": int(rr_sec.size),
            "heart_rate_bpm": float(feats["heart_rate_bpm"]),
        }
        for c in FEATURE_COLUMNS:
            row[c] = float(feats[c])
        rows.append(row)

        start += step_sec

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    uldd_dir = Path(args.uldd_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sessions = list(iter_session_items(uldd_dir))
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]
    if not sessions:
        raise RuntimeError("No UL-DD session csv found to process.")

    all_frames: list[pd.DataFrame] = []
    for idx, item in enumerate(sessions, start=1):
        df_session = extract_session_windows(
            item,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            min_rr_count=args.min_rr_count,
        )
        all_frames.append(df_session)

        if args.log_every > 0 and (idx % args.log_every == 0 or idx == len(sessions)):
            print(
                f"[INFO] {idx}/{len(sessions)} {item.subject}-{item.session}: "
                f"{len(df_session)} HRV windows"
            )

    out_df = pd.concat([df for df in all_frames if not df.empty], ignore_index=True)
    out_df = out_df.sort_values(["subject", "session", "t_sec"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    print(f"[INFO] Saved physio series to {output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
