"""Generate a 3-panel ECG comparison figure for presentation.

Panels:
  1. Alert segment  — first 60 s of recording (label = 0)
  2. Drowsy segment — 60 s window ending at first button press (label = 1)
  3. Simulated replay — same alert segment but annotated to show how
     VirtualSensor replays the signal in real time

Usage:
    python evaluation/plot_ecg_comparison.py
    python evaluation/plot_ecg_comparison.py --edf DD-Database/02F_1.edf --output output/ecg_comparison.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neurokit2 as nk
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from physio_channel.edf_loader import load_ecg_signal, load_button_onsets, derive_annotation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", default="DD-Database/01M_2.edf")
    parser.add_argument("--output", default="output/ecg_comparison.png")
    parser.add_argument("--window-sec", type=float, default=60.0)
    return parser.parse_args()


def get_rpeaks(ecg_window: np.ndarray, fs: float) -> np.ndarray:
    try:
        cleaned = nk.ecg_clean(ecg_window, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        return np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    except Exception:
        return np.array([], dtype=np.int64)


def draw_panel(
    ax: plt.Axes,
    ecg: np.ndarray,
    fs: float,
    rpeaks: np.ndarray,
    title: str,
    label_tag: str,
    label_color: str,
    show_window_box: bool = False,
    show_replay_arrow: bool = False,
) -> None:
    t = np.arange(len(ecg)) / fs
    ax.plot(t, ecg, color="#2196F3", linewidth=0.9, alpha=0.9, label="ECG signal")

    if rpeaks.size > 0:
        ax.scatter(
            rpeaks / fs,
            ecg[rpeaks],
            color="#F44336",
            s=40,
            zorder=5,
            label="R-peak",
        )
        if rpeaks.size > 1:
            rr_intervals = np.diff(rpeaks) / fs * 1000
            mean_rr = float(np.mean(rr_intervals))
            ax.annotate(
                f"mean RR = {mean_rr:.0f} ms",
                xy=(0.98, 0.06),
                xycoords="axes fraction",
                ha="right",
                fontsize=9,
                color="#888",
            )
            for i in range(min(3, rpeaks.size - 1)):
                x1, x2 = rpeaks[i] / fs, rpeaks[i + 1] / fs
                y_mid = float(np.min(ecg)) - 0.15 * (float(np.max(ecg)) - float(np.min(ecg)))
                ax.annotate(
                    "",
                    xy=(x2, y_mid),
                    xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="<->", color="#FF9800", lw=1.2),
                )

    if show_window_box:
        ax.axvspan(0, 30, alpha=0.06, color="#4CAF50", label="Window 1 (0–60 s)")
        ax.axvspan(30, 60, alpha=0.06, color="#9C27B0", label="Window 2 (30–90 s)")
        ax.text(15, float(np.max(ecg)) * 0.92, "Window 1", ha="center",
                fontsize=8, color="#4CAF50", fontweight="bold")
        ax.text(45, float(np.max(ecg)) * 0.92, "Window 2", ha="center",
                fontsize=8, color="#9C27B0", fontweight="bold")
        ax.text(30, float(np.max(ecg)) * 0.78, "← 30 s step →",
                ha="center", fontsize=8, color="#888")

    if show_replay_arrow:
        ax.annotate(
            "VirtualSensor:\nreplays at 128 Hz\n(real-time speed)",
            xy=(0.02, 0.85),
            xycoords="axes fraction",
            fontsize=8.5,
            color="#FF9800",
            bbox=dict(boxstyle="round,pad=0.3", fc="#2a2a2a", ec="#FF9800", lw=1.2),
        )

    patch_color = "#4CAF50" if label_color == "green" else "#F44336"
    tag_patch = mpatches.Patch(color=patch_color, label=label_tag)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [tag_patch], fontsize=8, loc="upper right")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude (μV)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_xlim(0, len(ecg) / fs)


def main() -> None:
    args = parse_args()

    edf_path = Path(args.edf)
    ann_path = derive_annotation_path(edf_path)

    print(f"[INFO] Loading ECG from {edf_path.name} ...")
    ecg_signal, fs, ch_name = load_ecg_signal(edf_path)
    button_onsets = load_button_onsets(ann_path)
    print(f"[INFO] Signal: {len(ecg_signal)} samples @ {fs:.0f} Hz, "
          f"{len(ecg_signal)/fs:.0f} s total")
    print(f"[INFO] Button onsets: {button_onsets}")

    win = int(args.window_sec * fs)

    # Panel 1 — Alert: first 60 s
    alert_seg = ecg_signal[:win]
    alert_peaks = get_rpeaks(alert_seg, fs)

    # Panel 2 — Drowsy: 60 s ending at a button press that is far enough in
    if button_onsets.size == 0:
        raise SystemExit("No button events found — choose another EDF.")
    # Pick the first button press that allows a full 60 s window
    valid_onsets = button_onsets[button_onsets > args.window_sec + 10]
    if valid_onsets.size == 0:
        valid_onsets = button_onsets
    first_btn = float(valid_onsets[0])
    drowsy_end = int(first_btn * fs)
    drowsy_start = max(0, drowsy_end - win)
    drowsy_seg = ecg_signal[drowsy_start:drowsy_end]
    if len(drowsy_seg) < win:
        drowsy_seg = np.pad(drowsy_seg, (0, win - len(drowsy_seg)))
    drowsy_peaks = get_rpeaks(drowsy_seg, fs)

    # Panel 3 — Simulated replay (same alert segment, annotated differently)
    replay_seg = alert_seg.copy()
    replay_peaks = alert_peaks.copy()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle(
        f"ECG Segments from DD-Database  ·  {edf_path.stem}  ·  {fs:.0f} Hz  (Simulated via VirtualSensor)",
        fontsize=12, fontweight="bold", y=0.995,
    )

    draw_panel(
        axes[0], alert_seg, fs, alert_peaks,
        title="① Alert Segment  (0 – 60 s, label = 0)",
        label_tag="Label: ALERT",
        label_color="green",
    )

    draw_panel(
        axes[1], drowsy_seg, fs, drowsy_peaks,
        title=f"② Drowsy Segment  ({first_btn-60:.0f} – {first_btn:.0f} s, 60 s before button press, label = 1)",
        label_tag="Label: DROWSY",
        label_color="red",
    )

    draw_panel(
        axes[2], replay_seg, fs, replay_peaks,
        title="③ VirtualSensor Replay  (60 s window, 30 s sliding step)",
        label_tag="Simulated real-time replay",
        label_color="green",
        show_window_box=True,
        show_replay_arrow=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
