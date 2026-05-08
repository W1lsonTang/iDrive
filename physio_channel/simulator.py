"""Standalone physiological channel simulator UI (OpenCV + matplotlib)."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

try:
    from .features import extract_hrv_features
    from .model import PhysioPredictor
    from .virtual_sensor import VirtualECGSensor
except ImportError:  # Allow direct execution: python physio_channel/simulator.py
    from features import extract_hrv_features
    from model import PhysioPredictor
    from virtual_sensor import VirtualECGSensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physiological-channel simulator UI.")
    parser.add_argument("--edf", required=True, help="Path to DD-Database recording EDF.")
    parser.add_argument("--annotations", default=None, help="Optional annotation EDF path.")
    parser.add_argument("--model", default=None, help="Optional trained model path.")
    parser.add_argument("--scaler", default="output/physio_scaler.pkl", help="Scaler path used with --model.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--inference-interval-sec", type=float, default=30.0, help="Inference period in simulated seconds.")
    parser.add_argument("--waveform-sec", type=float, default=10.0, help="Displayed ECG history length.")
    parser.add_argument("--buffer-sec", type=float, default=60.0, help="HRV feature buffer length.")
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=0.0,
        help="Optional auto-exit runtime in wall-clock seconds (0 disables).",
    )
    parser.add_argument("--width", type=int, default=1280, help="UI width.")
    parser.add_argument("--height", type=int, default=720, help="UI height.")
    return parser.parse_args()


def label_to_text(label: int) -> str:
    if label == 1:
        return "DROWSY"
    if label == 0:
        return "ALERT"
    return "UNKNOWN"


def draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.6,
    color: Tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_waveform(canvas: np.ndarray, waveform: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
    if waveform.size < 2:
        draw_text(canvas, "Waiting for ECG samples...", x + 10, y + 28, scale=0.6, color=(180, 180, 180))
        return

    values = waveform.astype(np.float64)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-8:
        values = values - vmin
        vmax = 1.0
        vmin = 0.0

    xs = np.linspace(x + 6, x + w - 6, num=len(values), dtype=np.float32)
    ys = y + h - 6 - ((values - vmin) / (vmax - vmin)) * (h - 12)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=(0, 220, 0), thickness=1, lineType=cv2.LINE_AA)


def render_hr_plot(
    history_t: Deque[float],
    history_hr: Deque[float],
    width: int,
    height: int,
) -> np.ndarray:
    fig = Figure(figsize=(max(width, 100) / 100.0, max(height, 100) / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Heart Rate Trend (last 5 min)", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.3)

    if history_t and history_hr:
        times = np.asarray(history_t, dtype=np.float64)
        hrs = np.asarray(history_hr, dtype=np.float64)
        t0 = float(times[-1]) - 300.0
        mask = times >= t0
        times = times[mask]
        hrs = hrs[mask]
        x = times - (times[0] if times.size > 0 else 0.0)
        ax.plot(x, hrs, color="tab:blue", linewidth=1.8)
        if hrs.size > 0:
            ax.set_ylim(max(35.0, float(np.min(hrs) - 8.0)), min(160.0, float(np.max(hrs) + 8.0)))
    else:
        ax.text(0.5, 0.5, "No HR data yet", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_probability_bar(canvas: np.ndarray, prob: Optional[float], rect: Tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 1)
    if prob is None or not np.isfinite(prob):
        draw_text(canvas, "P(drowsy): n/a", x + 8, y + h - 8, 0.55, (190, 190, 190))
        return
    p = min(max(float(prob), 0.0), 1.0)
    fill_w = int((w - 2) * p)
    fill_color = (0, 220, 220) if p < 0.5 else (0, 80, 255)
    cv2.rectangle(canvas, (x + 1, y + 1), (x + 1 + fill_w, y + h - 1), fill_color, -1)
    draw_text(canvas, f"P(drowsy): {p:.2f}", x + 8, y + h - 8, 0.55, (20, 20, 20), 1)


def main() -> None:
    args = parse_args()
    start_wall = time.perf_counter()

    predictor: Optional[PhysioPredictor] = None
    if args.model:
        if not Path(args.scaler).is_file():
            raise FileNotFoundError(f"Scaler file not found: {args.scaler}")
        predictor = PhysioPredictor(model_path=args.model, scaler_path=args.scaler)

    sensor = VirtualECGSensor(
        edf_path=args.edf,
        annotation_path=args.annotations,
        playback_speed=args.speed,
        buffer_seconds=max(args.buffer_sec, args.waveform_sec, 60.0),
    )
    sensor.start(reset=True)

    hr_history_t: Deque[float] = deque(maxlen=2400)
    hr_history_v: Deque[float] = deque(maxlen=2400)
    latest_features = None
    latest_prob: Optional[float] = None
    latest_pred = -1

    last_infer_t = -1e9
    last_plot_update_wall = 0.0
    plot_image = np.zeros((220, 400, 3), dtype=np.uint8)

    window_name = "Physio Channel Simulator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    try:
        while True:
            if args.max_runtime_sec > 0 and (time.perf_counter() - start_wall) >= args.max_runtime_sec:
                break

            _ = sensor.read_since_last()
            now_t = sensor.get_current_time_sec()
            gt = sensor.get_ground_truth(now_t)

            if now_t - last_infer_t >= args.inference_interval_sec:
                buffer_signal = sensor.get_hrv_buffer()
                if buffer_signal is not None:
                    feats = extract_hrv_features(buffer_signal, sampling_rate=sensor.sampling_rate)
                    if feats is not None:
                        latest_features = feats
                        hr_val = float(feats.get("heart_rate_bpm", np.nan))
                        if np.isfinite(hr_val):
                            hr_history_t.append(now_t)
                            hr_history_v.append(hr_val)
                        if predictor is not None:
                            latest_prob = predictor.predict_proba(feats)
                            latest_pred = int(latest_prob >= 0.5)
                last_infer_t = now_t

            wall_now = time.perf_counter()
            if wall_now - last_plot_update_wall >= 1.0:
                plot_image = render_hr_plot(hr_history_t, hr_history_v, width=420, height=230)
                last_plot_update_wall = wall_now

            canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            canvas[:] = (22, 22, 22)

            # Top banner
            status = label_to_text(latest_pred) if predictor is not None else "NO MODEL"
            if status == "DROWSY":
                banner_color = (0, 0, 220)
            elif status == "ALERT":
                banner_color = (0, 160, 0)
            else:
                banner_color = (70, 70, 70)
            cv2.rectangle(canvas, (0, 0), (args.width, 70), banner_color, -1)
            draw_text(canvas, f"Status: {status}", 20, 44, scale=1.0, color=(255, 255, 255), thickness=2)
            draw_text(
                canvas,
                f"Source: {Path(args.edf).name} | t={now_t:6.1f}s | speed={sensor.playback_speed:.2f}x",
                340,
                30,
                scale=0.55,
                color=(240, 240, 240),
                thickness=1,
            )
            draw_text(
                canvas,
                f"Progress: {sensor.get_progress_ratio() * 100:5.1f}% | {'PAUSED' if sensor.is_paused() else 'RUNNING'}",
                340,
                54,
                scale=0.55,
                color=(240, 240, 240),
                thickness=1,
            )

            # ECG waveform panel
            draw_text(canvas, "ECG Waveform (last 10s)", 20, 100, scale=0.65)
            wave = sensor.get_recent_waveform(seconds=args.waveform_sec)
            draw_waveform(canvas, wave, rect=(20, 115, args.width - 40, 250))

            # Metrics panel
            cv2.rectangle(canvas, (20, 390), (430, 640), (45, 45, 45), -1)
            cv2.rectangle(canvas, (20, 390), (430, 640), (90, 90, 90), 1)
            draw_text(canvas, "HRV Metrics", 34, 420, 0.72)

            if latest_features is None:
                draw_text(canvas, "Waiting for enough data (60s buffer)...", 34, 455, 0.56, (180, 180, 180))
            else:
                metric_lines = [
                    f"HR: {latest_features.get('heart_rate_bpm', np.nan):.1f} bpm",
                    f"Mean RR: {latest_features.get('mean_rr_ms', np.nan):.1f} ms",
                    f"SDNN: {latest_features.get('sdnn_ms', np.nan):.1f} ms",
                    f"RMSSD: {latest_features.get('rmssd_ms', np.nan):.1f} ms",
                    f"pNN50: {latest_features.get('pnn50', np.nan):.2f}",
                    f"LF/HF: {latest_features.get('lf_hf_ratio', np.nan):.2f}",
                ]
                for idx, line in enumerate(metric_lines):
                    draw_text(canvas, line, 34, 455 + idx * 28, 0.62)

            # HR trend panel (matplotlib rendered image)
            cv2.rectangle(canvas, (460, 390), (args.width - 20, 640), (45, 45, 45), -1)
            cv2.rectangle(canvas, (460, 390), (args.width - 20, 640), (90, 90, 90), 1)
            plot_h, plot_w = plot_image.shape[:2]
            target_x1, target_y1 = 472, 402
            target_x2, target_y2 = args.width - 32, 632
            target_w = target_x2 - target_x1
            target_h = target_y2 - target_y1
            resized_plot = cv2.resize(plot_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            canvas[target_y1:target_y2, target_x1:target_x2] = resized_plot

            # Bottom status strip
            cv2.rectangle(canvas, (20, 650), (args.width - 20, args.height - 20), (40, 40, 40), -1)
            cv2.rectangle(canvas, (20, 650), (args.width - 20, args.height - 20), (80, 80, 80), 1)
            draw_probability_bar(canvas, latest_prob, rect=(34, 665, 520, 26))

            gt_text = label_to_text(gt)
            pred_text = label_to_text(latest_pred) if predictor is not None else "N/A"
            if predictor is not None and gt in (0, 1) and latest_pred in (0, 1):
                match_text = "MATCH" if gt == latest_pred else "MISMATCH"
            else:
                match_text = "N/A"
            draw_text(
                canvas,
                f"GroundTruth: {gt_text} | Prediction: {pred_text} | Compare: {match_text}",
                580,
                684,
                0.62,
                (230, 230, 230),
            )

            draw_text(
                canvas,
                "Controls: q=quit  space=pause  +/-=speed  r=reset",
                34,
                args.height - 28,
                0.58,
                (180, 180, 180),
            )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                sensor.toggle_pause()
            if key in (ord("+"), ord("=")):
                sensor.set_playback_speed(sensor.playback_speed * 1.25)
            if key == ord("-"):
                sensor.set_playback_speed(sensor.playback_speed / 1.25)
            if key == ord("r"):
                sensor.reset()
                sensor.start(reset=False)
                hr_history_t.clear()
                hr_history_v.clear()
                latest_features = None
                latest_prob = None
                latest_pred = -1
                last_infer_t = -1e9
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
