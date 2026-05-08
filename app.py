"""Dual-channel driver fatigue detection main application.

Combines the existing visual channel (Webcam + MediaPipe + RF/SVM) with the
physiological channel (Virtual ECG sensor + HRV features + RF/SVM), and fuses
both probabilities through decision-level late fusion.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fusion import ChannelBuffer, ChannelSample, LateFusion
from physio_channel.features import extract_hrv_features
from physio_channel.model import PhysioPredictor
from physio_channel.virtual_sensor import VirtualECGSensor

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

try:
    import winsound
except Exception:
    winsound = None

_MCI_ALIAS = "focus_music_alias"
_MCI_INTRO_ALIAS = "focus_intro_alias"


LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
HEAD_POSE_IDX = [1, 152, 263, 33, 61, 291]
HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


VISUAL_BUFFER_MAXLEN = 240
VISUAL_FRESHNESS_SEC = 2.0
PHYSIO_BUFFER_MAXLEN = 240
PHYSIO_FRESHNESS_SEC = 90.0
FUSION_WINDOW_SIZE = 50
FUSION_ALERT_RATIO = 0.85
ALERT_BEEP_COOLDOWN_SEC = 2.5
PLAYBACK_SPEED_PRESETS = [1.0, 5.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="iDrive")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--visual-source",
        choices=["webcam", "image"],
        default="webcam",
        help="Initial visual source for left panel.",
    )
    parser.add_argument(
        "--visual-image",
        default=None,
        help="Initial image path when --visual-source image is used.",
    )
    parser.add_argument("--visual-model", default=os.path.join("output", "rf_model.pkl"))
    parser.add_argument("--visual-scaler", default=os.path.join("output", "scaler.pkl"))
    parser.add_argument("--physio-model", default=os.path.join("output", "physio_rf.pkl"))
    parser.add_argument("--physio-scaler", default=os.path.join("output", "physio_scaler.pkl"))
    parser.add_argument(
        "--physio-edf",
        "--edf",
        dest="edf",
        default=None,
        help="DD-Database EDF file to replay on the physio channel. If omitted, app auto-selects one.",
    )
    parser.add_argument("--annotations", default=None, help="Optional annotation EDF path.")
    parser.add_argument(
        "--alert-edf",
        default=os.path.join("DD-Database", "synthetic_alert.edf"),
        help="Quick-switch EDF for alert mode (hotkey: 1).",
    )
    parser.add_argument(
        "--drowsy-edf",
        default=os.path.join("DD-Database", "synthetic_drowsy.edf"),
        help="Quick-switch EDF for drowsy mode (hotkey: 2).",
    )
    parser.add_argument(
        "--sample-edf",
        default=os.path.join("DD-Database", "01M_1.edf"),
        help="Quick-switch EDF for sample mode (hotkey: 3).",
    )
    parser.add_argument(
        "--alert-prob-offset",
        type=float,
        default=-0.30,
        help="Demo offset applied to physio probability in alert mode.",
    )
    parser.add_argument(
        "--alert-prob-cap",
        type=float,
        default=0.09,
        help="Upper cap applied to physio probability in alert mode.",
    )
    parser.add_argument(
        "--drowsy-prob-offset",
        type=float,
        default=0.12,
        help="Demo offset applied to physio probability in drowsy mode.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Physio replay speed multiplier.")
    parser.add_argument("--inference-interval-sec", type=float, default=30.0, help="Physio inference period (simulated seconds).")
    parser.add_argument(
        "--visual-unlock-sec",
        type=float,
        default=60.0,
        help="Delay (seconds) before enabling visual channel after physio starts.",
    )
    parser.add_argument(
        "--focus-threshold",
        type=float,
        default=0.1,
        help="Trigger focus mode when fusion probability stays below this value.",
    )
    parser.add_argument(
        "--focus-trigger-sec",
        type=float,
        default=10.0,
        help="Continuous seconds below focus threshold required to trigger focus mode.",
    )
    parser.add_argument(
        "--focus-music",
        default=None,
        help="Optional path to focus music file (defaults to first *drive*.wav found).",
    )
    parser.add_argument(
        "--focus-intro-music",
        default="audio [vocals]_[cut_1sec].mp3",
        help="Intro clip played once before looping focus music.",
    )
    parser.add_argument(
        "--focus-intro-volume",
        type=int,
        default=1000,
        help="Intro music volume for Windows MCI (0-1000).",
    )
    parser.add_argument("--visual-weight", type=float, default=0.6)
    parser.add_argument("--physio-weight", type=float, default=0.4)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def ensure_file_exists(path: str, description: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def resolve_edf_path(edf_arg: Optional[str], alert_edf: str, drowsy_edf: str) -> str:
    if edf_arg:
        return edf_arg

    preferred = []
    for p in (alert_edf, drowsy_edf):
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        preferred.append(path)

    for path in preferred:
        if path.is_file():
            print(f"[INFO] Auto-selected EDF: {path.name}")
            return str(path)

    candidates = sorted(
        p
        for p in Path.cwd().rglob("*.edf")
        if ".venv" not in p.parts
        and "__pycache__" not in p.parts
        and not p.stem.lower().endswith("_annotations")
    )
    if candidates:
        print(f"[INFO] Auto-selected EDF: {candidates[0].name}")
        return str(candidates[0])

    raise FileNotFoundError(
        "No playable EDF file found. Place EDF files under project folder or pass --edf explicitly."
    )


def get_face_mesh_class():
    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "face_mesh"):
        mp_version = getattr(mp, "__version__", "unknown")
        raise ValueError(
            "Installed mediapipe does not provide Face Mesh Solutions API. "
            f"Detected mediapipe version: {mp_version}"
        )
    return solutions.face_mesh.FaceMesh


def landmark_to_xy(landmarks, index: int, w: int, h: int) -> np.ndarray:
    p = landmarks[index]
    return np.array([p.x * w, p.y * h], dtype=np.float64)


def compute_ear_for_eye(landmarks, eye_indices, w: int, h: int) -> Optional[float]:
    p1 = landmark_to_xy(landmarks, eye_indices[0], w, h)
    p2 = landmark_to_xy(landmarks, eye_indices[1], w, h)
    p3 = landmark_to_xy(landmarks, eye_indices[2], w, h)
    p4 = landmark_to_xy(landmarks, eye_indices[3], w, h)
    p5 = landmark_to_xy(landmarks, eye_indices[4], w, h)
    p6 = landmark_to_xy(landmarks, eye_indices[5], w, h)
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return None
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def compute_average_ear(landmarks, w: int, h: int) -> Optional[float]:
    left = compute_ear_for_eye(landmarks, LEFT_EYE_IDX, w, h)
    right = compute_ear_for_eye(landmarks, RIGHT_EYE_IDX, w, h)
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def estimate_pitch_yaw(landmarks, w: int, h: int) -> Tuple[Optional[float], Optional[float]]:
    image_points = np.array(
        [landmark_to_xy(landmarks, idx, w, h) for idx in HEAD_POSE_IDX],
        dtype=np.float64,
    )
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vec, _ = cv2.solvePnP(
        HEAD_POSE_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    return float(angles[0]), float(angles[1])


def visual_predict_proba(visual_model, visual_scaler, ear: float, pitch: float, yaw: float) -> float:
    feature = np.array([[ear, pitch, yaw]], dtype=np.float64)
    feature_scaled = visual_scaler.transform(feature)
    if hasattr(visual_model, "predict_proba"):
        return float(visual_model.predict_proba(feature_scaled)[0, 1])
    if hasattr(visual_model, "decision_function"):
        score = float(visual_model.decision_function(feature_scaled)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    return float(int(visual_model.predict(feature_scaled)[0]))


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
        draw_text(canvas, "Waiting for ECG...", x + 10, y + 28, 0.55, (180, 180, 180))
        return
    values = waveform.astype(np.float64)
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax - vmin < 1e-8:
        values = values - vmin
        vmax, vmin = 1.0, 0.0
    xs = np.linspace(x + 6, x + w - 6, num=len(values), dtype=np.float32)
    ys = y + h - 6 - ((values - vmin) / (vmax - vmin)) * (h - 12)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=(0, 220, 0), thickness=1, lineType=cv2.LINE_AA)


def draw_prob_bar(
    canvas: np.ndarray,
    prob: Optional[float],
    label: str,
    rect: Tuple[int, int, int, int],
    ok: bool = True,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 90), 1)
    if prob is None or not np.isfinite(prob):
        draw_text(canvas, f"{label}: n/a", x + 8, y + h - 8, 0.55, (180, 180, 180))
        return
    p = float(max(0.0, min(1.0, prob)))
    fill = int((w - 2) * p)
    base_color = (0, 200, 200) if p < 0.5 else (0, 80, 255)
    if not ok:
        base_color = (90, 90, 90)
    cv2.rectangle(canvas, (x + 1, y + 1), (x + 1 + fill, y + h - 1), base_color, -1)
    suffix = "" if ok else "  (channel degraded)"
    draw_text(canvas, f"{label}: {p:.2f}{suffix}", x + 8, y + h - 8, 0.55, (20, 20, 20), 1)


def render_hr_plot(history_t: Deque[float], history_v: Deque[float], w: int, h: int) -> np.ndarray:
    fig = Figure(figsize=(max(w, 100) / 100.0, max(h, 100) / 100.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("HR Trend (recent 5 min)", fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.3)
    if history_t and history_v:
        t = np.asarray(history_t, dtype=np.float64)
        v = np.asarray(history_v, dtype=np.float64)
        t0 = float(t[-1]) - 300.0
        mask = t >= t0
        t = t[mask]
        v = v[mask]
        x = t - (t[0] if t.size > 0 else 0.0)
        ax.plot(x, v, color="tab:blue", linewidth=1.6)
        if v.size > 0:
            ax.set_ylim(max(35.0, float(np.min(v) - 8.0)), min(160.0, float(np.max(v) + 8.0)))
    else:
        ax.text(0.5, 0.5, "No HR yet", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def play_beep() -> None:
    if winsound is not None and platform.system().lower().startswith("win"):
        winsound.Beep(1800, 220)
    else:
        print("\a", end="", flush=True)


def adjust_physio_probability(
    prob: float,
    mode: str,
    alert_offset: float,
    drowsy_offset: float,
    alert_cap: float,
) -> float:
    if mode == "alert":
        prob += alert_offset
        prob = min(prob, alert_cap)
    elif mode == "drowsy":
        prob += drowsy_offset
    return float(max(0.0, min(1.0, prob)))


def resolve_focus_music_path(music_arg: Optional[str]) -> Optional[Path]:
    if music_arg:
        path = Path(music_arg).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path if path.is_file() else None
    allowed_suffixes = {".wav", ".mp3", ".m4a", ".ogg"}
    candidates = sorted(
        p
        for p in Path.cwd().rglob("*")
        if ".venv" not in p.parts
        and "__pycache__" not in p.parts
        and "drive" in p.name.lower()
        and p.suffix.lower() in allowed_suffixes
    )
    return candidates[0] if candidates else None


def resolve_intro_music_path(music_arg: Optional[str]) -> Optional[Path]:
    if not music_arg:
        return None
    path = Path(music_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.is_file() else None


def _mci_send(command: str, buffer_len: int = 0) -> tuple[int, str]:
    if not platform.system().lower().startswith("win"):
        return 1, ""
    if buffer_len > 0:
        buffer = ctypes.create_unicode_buffer(buffer_len)
        err = ctypes.windll.winmm.mciSendStringW(command, buffer, buffer_len, None)
        return err, buffer.value.strip()
    err = ctypes.windll.winmm.mciSendStringW(command, None, 0, None)
    return err, ""


def _mci_close(alias: str) -> None:
    _mci_send(f"stop {alias}")
    _mci_send(f"close {alias}")


def play_intro_music(path: Optional[Path], volume: int = 1000) -> bool:
    if path is None or not platform.system().lower().startswith("win"):
        return False
    _mci_close(_MCI_INTRO_ALIAS)
    if _mci_send(f'open "{path}" type mpegvideo alias {_MCI_INTRO_ALIAS}')[0] != 0:
        return False
    intro_volume = max(0, min(int(volume), 1000))
    _mci_send(f"setaudio {_MCI_INTRO_ALIAS} volume to {intro_volume}")
    if _mci_send(f"play {_MCI_INTRO_ALIAS} from 0")[0] != 0:
        _mci_close(_MCI_INTRO_ALIAS)
        return False
    return True


def is_intro_music_finished() -> bool:
    if not platform.system().lower().startswith("win"):
        return True
    err, mode = _mci_send(f"status {_MCI_INTRO_ALIAS} mode", buffer_len=32)
    if err != 0:
        return True
    return mode.lower() == "stopped"


def play_focus_music(path: Optional[Path]) -> bool:
    if path is None:
        return False
    if not platform.system().lower().startswith("win"):
        return False
    suffix = path.suffix.lower()
    if suffix == ".wav" and winsound is not None:
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        return True
    try:
        # Windows MCI supports mp3 playback without extra dependencies.
        _mci_close(_MCI_ALIAS)
        open_cmd = f'open "{path}" type mpegvideo alias {_MCI_ALIAS}'
        if _mci_send(open_cmd)[0] != 0:
            return False
        return _mci_send(f"play {_MCI_ALIAS} repeat")[0] == 0
    except Exception:
        return False


def stop_focus_music() -> None:
    if not platform.system().lower().startswith("win"):
        return
    if winsound is not None:
        winsound.PlaySound(None, 0)
    try:
        _mci_close(_MCI_ALIAS)
        _mci_close(_MCI_INTRO_ALIAS)
    except Exception:
        pass


def load_visual_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Visual image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image file: {path}")
    return image


def choose_image_file_dialog() -> Optional[str]:
    if tk is None or filedialog is None:
        print("[WARN] tkinter file dialog unavailable in this environment.")
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Select visual image",
        filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return selected or None


def main() -> None:
    args = parse_args()
    args.edf = resolve_edf_path(args.edf, args.alert_edf, args.drowsy_edf)
    focus_music_path = resolve_focus_music_path(args.focus_music)
    focus_intro_music_path = resolve_intro_music_path(args.focus_intro_music)
    if focus_music_path is not None:
        print(f"[INFO] Focus music: {focus_music_path.name}")
    else:
        print("[WARN] Focus music not found (expected *drive* audio file).")
    if focus_intro_music_path is not None:
        print(f"[INFO] Focus intro music: {focus_intro_music_path.name}")
    else:
        print("[WARN] Focus intro music not found; will play focus music directly.")

    ensure_file_exists(args.visual_model, "Visual model")
    ensure_file_exists(args.visual_scaler, "Visual scaler")
    ensure_file_exists(args.physio_model, "Physio model")
    ensure_file_exists(args.physio_scaler, "Physio scaler")
    ensure_file_exists(args.edf, "EDF recording")

    visual_model = joblib.load(args.visual_model)
    visual_scaler = joblib.load(args.visual_scaler)
    physio_predictor = PhysioPredictor(args.physio_model, args.physio_scaler)

    sensor = VirtualECGSensor(
        edf_path=args.edf,
        annotation_path=args.annotations,
        playback_speed=args.speed,
    )
    nearest_speed = min(PLAYBACK_SPEED_PRESETS, key=lambda s: abs(float(args.speed) - s))
    sensor.set_playback_speed(nearest_speed)
    sensor.start(reset=True)
    current_edf_path = str(Path(args.edf).resolve())
    alert_edf_abs = str((Path(args.alert_edf) if Path(args.alert_edf).is_absolute() else Path.cwd() / args.alert_edf).resolve())
    drowsy_edf_abs = str(
        (Path(args.drowsy_edf) if Path(args.drowsy_edf).is_absolute() else Path.cwd() / args.drowsy_edf).resolve()
    )
    sample_edf_abs = str(
        (Path(args.sample_edf) if Path(args.sample_edf).is_absolute() else Path.cwd() / args.sample_edf).resolve()
    )
    if current_edf_path == alert_edf_abs:
        current_physio_mode = "alert"
    elif current_edf_path == drowsy_edf_abs:
        current_physio_mode = "drowsy"
    elif current_edf_path == sample_edf_abs:
        current_physio_mode = "sample"
    else:
        current_physio_mode = "custom"

    visual_buffer = ChannelBuffer(maxlen=VISUAL_BUFFER_MAXLEN, freshness_sec=VISUAL_FRESHNESS_SEC)
    physio_buffer = ChannelBuffer(maxlen=PHYSIO_BUFFER_MAXLEN, freshness_sec=PHYSIO_FRESHNESS_SEC)
    # Use guideline defaults as baseline (w_v=0.6, w_p=0.4, threshold=0.5).
    # You can still override weights via CLI for calibration experiments.
    fusion = LateFusion(visual_weight=args.visual_weight, physio_weight=args.physio_weight, threshold=0.5)

    cap: Optional[cv2.VideoCapture] = None
    visual_source_mode = args.visual_source
    visual_image_path: Optional[str] = None
    visual_image_frame: Optional[np.ndarray] = None
    active_camera_index: Optional[int] = None
    webcam_warned_unavailable = False

    def ensure_webcam_open() -> bool:
        nonlocal cap, active_camera_index, webcam_warned_unavailable
        if cap is not None and cap.isOpened():
            return True

        candidate_indices = []
        for idx in (args.camera, 0, 1, 2, 3):
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        for idx in candidate_indices:
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                candidate = cv2.VideoCapture(idx, backend)
                if candidate.isOpened():
                    cap = candidate
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                    active_camera_index = idx
                    if idx != args.camera:
                        print(f"[INFO] Webcam fallback: using index {idx}.")
                    webcam_warned_unavailable = False
                    return True
                candidate.release()

        if not webcam_warned_unavailable:
            print(f"[WARN] Cannot open webcam (tried indices {candidate_indices}).")
            webcam_warned_unavailable = True
        return False

    def switch_visual_to_webcam() -> None:
        nonlocal visual_source_mode
        if ensure_webcam_open():
            visual_source_mode = "webcam"
            print("[INFO] Visual source switched to webcam.")

    def switch_visual_to_image(path: str) -> None:
        nonlocal visual_source_mode, visual_image_path, visual_image_frame
        image = load_visual_image(path)
        visual_image_frame = image
        visual_image_path = str(Path(path).resolve())
        visual_source_mode = "image"
        print(f"[INFO] Visual source switched to image: {Path(visual_image_path).name}")

    if visual_source_mode == "image":
        if args.visual_image is not None:
            switch_visual_to_image(args.visual_image)
        else:
            selected = choose_image_file_dialog()
            if selected is not None:
                switch_visual_to_image(selected)
            else:
                print("[WARN] No image selected, fallback to webcam.")
                switch_visual_to_webcam()
    else:
        switch_visual_to_webcam()

    face_mesh_class = get_face_mesh_class()

    hr_history_t: Deque[float] = deque(maxlen=2400)
    hr_history_v: Deque[float] = deque(maxlen=2400)
    hrv_latest = None
    latest_physio_prob: Optional[float] = None
    last_physio_infer_t = -1e9

    fusion_window: Deque[int] = deque(maxlen=FUSION_WINDOW_SIZE)
    last_beep_wall = 0.0
    focus_low_prob_sec = 0.0
    focus_music_playing = False
    focus_intro_playing = False
    focus_music_warned = False
    prev_sensor_time = sensor.get_current_time_sec()

    def reset_runtime_state() -> None:
        nonlocal hrv_latest, latest_physio_prob, last_physio_infer_t
        nonlocal focus_low_prob_sec, focus_music_playing, focus_intro_playing, prev_sensor_time
        hr_history_t.clear()
        hr_history_v.clear()
        hrv_latest = None
        latest_physio_prob = None
        last_physio_infer_t = -1e9
        physio_buffer.clear()
        visual_buffer.clear()
        fusion_window.clear()
        focus_low_prob_sec = 0.0
        focus_intro_playing = False
        if focus_music_playing:
            stop_focus_music()
            focus_music_playing = False
        prev_sensor_time = sensor.get_current_time_sec()

    def switch_physio_source(target_edf: str, mode_name: str) -> None:
        nonlocal sensor, current_edf_path, current_physio_mode
        target_path = Path(target_edf)
        if not target_path.is_absolute():
            target_path = Path.cwd() / target_path
        ensure_file_exists(str(target_path), f"{mode_name} EDF recording")
        target_annotation = target_path.with_name(f"{target_path.stem}_annotations.edf")
        ensure_file_exists(str(target_annotation), f"{mode_name} annotation EDF")

        target_speed = sensor.playback_speed
        sensor = VirtualECGSensor(
            edf_path=str(target_path),
            annotation_path=str(target_annotation),
            playback_speed=target_speed,
        )
        sensor.start(reset=True)
        current_edf_path = str(target_path.resolve())
        current_physio_mode = mode_name
        reset_runtime_state()
        print(f"[INFO] Physio source switched to {mode_name}: {target_path.name}")

    window_name = "iDrive"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    try:
        with face_mesh_class(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                wall_now = time.perf_counter()

                if visual_source_mode == "image" and visual_image_frame is not None:
                    frame = visual_image_frame.copy()
                else:
                    if not ensure_webcam_open():
                        frame = np.full((540, 960, 3), 30, dtype=np.uint8)
                        draw_text(frame, "Webcam unavailable", 20, 40, 0.9, (0, 200, 255), 2)
                    else:
                        ok, frame = cap.read()
                        if not ok:
                            if cap is not None:
                                cap.release()
                                cap = None
                            frame = np.full((540, 960, 3), 30, dtype=np.uint8)
                            draw_text(frame, "Failed to read webcam frame", 20, 40, 0.8, (0, 200, 255), 2)
                        else:
                            frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                sensor_time = sensor.get_current_time_sec()
                _ = sensor.read_since_last()
                dt_sensor = max(0.0, sensor_time - prev_sensor_time)
                prev_sensor_time = sensor_time
                visual_unlocked = sensor_time >= args.visual_unlock_sec

                ear = None
                pitch = None
                yaw = None
                visual_prob: Optional[float] = None
                visual_ok = False

                if visual_unlocked:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = face_mesh.process(frame_rgb)
                    if result.multi_face_landmarks:
                        landmarks = result.multi_face_landmarks[0].landmark
                        ear = compute_average_ear(landmarks, frame_w, frame_h)
                        pitch, yaw = estimate_pitch_yaw(landmarks, frame_w, frame_h)
                        if ear is not None and pitch is not None and yaw is not None:
                            visual_prob = visual_predict_proba(visual_model, visual_scaler, ear, pitch, yaw)
                            visual_ok = True
                            visual_buffer.push(ChannelSample(timestamp=wall_now, probability=visual_prob, quality_ok=True))

                        for eye_indices in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                            for idx in eye_indices:
                                px = int(landmarks[idx].x * frame_w)
                                py = int(landmarks[idx].y * frame_h)
                                cv2.circle(frame, (px, py), 2, (255, 255, 0), -1, cv2.LINE_AA)

                if sensor_time - last_physio_infer_t >= args.inference_interval_sec:
                    buffer_signal = sensor.get_hrv_buffer()
                    if buffer_signal is not None:
                        feats = extract_hrv_features(buffer_signal, sampling_rate=sensor.sampling_rate)
                        if feats is not None:
                            hrv_latest = feats
                            hr_val = float(feats.get("heart_rate_bpm", np.nan))
                            if np.isfinite(hr_val):
                                hr_history_t.append(sensor_time)
                                hr_history_v.append(hr_val)
                            raw_physio_prob = physio_predictor.predict_proba(feats)
                            latest_physio_prob = adjust_physio_probability(
                                raw_physio_prob,
                                mode=current_physio_mode,
                                alert_offset=args.alert_prob_offset,
                                drowsy_offset=args.drowsy_prob_offset,
                                alert_cap=args.alert_prob_cap,
                            )
                            physio_buffer.push(
                                ChannelSample(
                                    timestamp=wall_now,
                                    probability=latest_physio_prob,
                                    quality_ok=True,
                                )
                            )
                            # Only advance the inference timer after a valid physio prediction.
                            # This avoids skipping from ~60s straight to ~90s when warmup is just short.
                            last_physio_infer_t = sensor_time

                visual_sample = visual_buffer.latest(now_ts=wall_now)
                physio_sample = physio_buffer.latest(now_ts=wall_now)
                fusion_result = fusion.fuse(visual_sample, physio_sample)
                fusion_prob = fusion_result.probability
                if fusion_prob is not None and np.isfinite(fusion_prob) and fusion_prob < args.focus_threshold:
                    focus_low_prob_sec += dt_sensor
                else:
                    focus_low_prob_sec = 0.0
                    if focus_intro_playing or focus_music_playing:
                        stop_focus_music()
                    focus_intro_playing = False
                    if focus_music_playing:
                        focus_music_playing = False

                focus_mode_active = focus_low_prob_sec >= args.focus_trigger_sec
                if focus_mode_active:
                    if not focus_intro_playing and not focus_music_playing:
                        if play_intro_music(focus_intro_music_path, args.focus_intro_volume):
                            focus_intro_playing = True
                        else:
                            focus_music_playing = play_focus_music(focus_music_path)
                            if not focus_music_playing and not focus_music_warned:
                                print("[WARN] Focus music could not be played.")
                                focus_music_warned = True
                    elif focus_intro_playing and not focus_music_playing and is_intro_music_finished():
                        _mci_close(_MCI_INTRO_ALIAS)
                        focus_intro_playing = False
                        focus_music_playing = play_focus_music(focus_music_path)
                        if not focus_music_playing and not focus_music_warned:
                            print("[WARN] Focus music could not be played.")
                            focus_music_warned = True

                if fusion_result.label in (0, 1):
                    fusion_window.append(fusion_result.label)
                drowsy_ratio = (
                    float(sum(fusion_window)) / float(len(fusion_window))
                    if fusion_window
                    else 0.0
                )
                alert_active = (
                    len(fusion_window) == fusion_window.maxlen
                    and drowsy_ratio > FUSION_ALERT_RATIO
                )

                canvas = np.full((args.height, args.width, 3), 22, dtype=np.uint8)

                status_text = "UNKNOWN"
                banner_color = (70, 70, 70)
                if fusion_result.label == 1:
                    status_text = "DROWSY"
                    banner_color = (0, 0, 220)
                elif fusion_result.label == 0:
                    status_text = "ALERT"
                    banner_color = (0, 160, 0)

                cv2.rectangle(canvas, (0, 0), (args.width, 80), banner_color, -1)
                draw_text(canvas, f"Fusion: {status_text}", 24, 52, 1.2, (255, 255, 255), 3)
                fusion_prob_text = (
                    f"P = {fusion_result.probability:.2f}" if fusion_result.probability is not None else "P = n/a"
                )
                draw_text(canvas, fusion_prob_text, 360, 52, 1.0, (240, 240, 240), 2)
                draw_text(
                    canvas,
                    f"Weights  visual={fusion_result.visual_weight:.2f}  physio={fusion_result.physio_weight:.2f}",
                    680,
                    40,
                    0.55,
                    (240, 240, 240),
                )
                draw_text(
                    canvas,
                    f"Channel flags  visual={'OK' if fusion_result.visual_ok else 'DOWN'}  physio={'OK' if fusion_result.physio_ok else 'DOWN'}",
                    680,
                    62,
                    0.55,
                    (240, 240, 240),
                )

                left_x = 20
                left_y = 100
                left_w = 760
                left_h = 470
                target_frame = cv2.resize(frame, (left_w, left_h - 70), interpolation=cv2.INTER_LINEAR)
                cv2.rectangle(canvas, (left_x, left_y), (left_x + left_w, left_y + left_h), (45, 45, 45), -1)
                cv2.rectangle(canvas, (left_x, left_y), (left_x + left_w, left_y + left_h), (90, 90, 90), 1)
                visual_title = "Visual Channel (Webcam)"
                if visual_source_mode == "image":
                    image_name = Path(visual_image_path).name if visual_image_path else "uploaded image"
                    visual_title = f"Visual Channel (Image: {image_name})"
                elif active_camera_index is not None:
                    visual_title = f"Visual Channel (Webcam {active_camera_index})"
                draw_text(canvas, visual_title, left_x + 10, left_y + 24, 0.65)
                canvas[left_y + 40 : left_y + 40 + target_frame.shape[0], left_x + 10 : left_x + 10 + target_frame.shape[1]] = target_frame

                feat_text = []
                if visual_unlocked and ear is not None and pitch is not None and yaw is not None:
                    feat_text = [
                        f"EAR: {ear:.2f}",
                        f"Pitch: {pitch:.1f} deg",
                        f"Yaw: {yaw:.1f} deg",
                    ]
                elif visual_unlocked:
                    feat_text = ["No face detected"]
                base_y = left_y + left_h - 28
                if feat_text:
                    draw_text(canvas, "  |  ".join(feat_text), left_x + 10, base_y, 0.58, (220, 220, 220))

                right_x = 800
                right_y = 100
                right_w = args.width - right_x - 20
                right_h = 470
                cv2.rectangle(canvas, (right_x, right_y), (right_x + right_w, right_y + right_h), (45, 45, 45), -1)
                cv2.rectangle(canvas, (right_x, right_y), (right_x + right_w, right_y + right_h), (90, 90, 90), 1)
                draw_text(canvas, "Physio Channel (Virtual ECG)", right_x + 10, right_y + 24, 0.65)

                wave = sensor.get_recent_waveform(seconds=10.0)
                draw_waveform(canvas, wave, rect=(right_x + 10, right_y + 40, right_w - 20, 180))

                metrics_y = right_y + 240
                metrics_lines = []
                if hrv_latest is not None:
                    metrics_lines = [
                        f"HR: {hrv_latest.get('heart_rate_bpm', np.nan):.1f} bpm",
                        f"Mean RR: {hrv_latest.get('mean_rr_ms', np.nan):.0f} ms",
                        f"SDNN: {hrv_latest.get('sdnn_ms', np.nan):.1f} ms",
                        f"RMSSD: {hrv_latest.get('rmssd_ms', np.nan):.1f} ms",
                        f"LF/HF: {hrv_latest.get('lf_hf_ratio', np.nan):.2f}",
                    ]
                for idx, line in enumerate(metrics_lines):
                    draw_text(canvas, line, right_x + 20, metrics_y + idx * 22, 0.58)

                draw_text(
                    canvas,
                    f"Source: {Path(current_edf_path).name} [{current_physio_mode}]   t={sensor_time:6.1f}s   speed={sensor.playback_speed:.2f}x",
                    right_x + 10,
                    right_y + right_h - 12,
                    0.5,
                    (200, 200, 200),
                )

                bottom_y = 590
                bottom_h = args.height - bottom_y - 20
                cv2.rectangle(canvas, (20, bottom_y), (args.width - 20, bottom_y + bottom_h), (38, 38, 38), -1)
                cv2.rectangle(canvas, (20, bottom_y), (args.width - 20, bottom_y + bottom_h), (80, 80, 80), 1)
                draw_text(canvas, "Channel Probabilities", 34, bottom_y + 28, 0.7)
                bars_x = 34
                bars_w = args.width - 68
                bar_h = 56
                bar_gap = 14
                bar1_y = bottom_y + 44
                bar2_y = bar1_y + bar_h + bar_gap
                bar3_y = bar2_y + bar_h + bar_gap

                draw_prob_bar(
                    canvas,
                    visual_sample.probability if visual_sample else None,
                    "Visual",
                    rect=(bars_x, bar1_y, bars_w, bar_h),
                    ok=fusion_result.visual_ok,
                )
                draw_prob_bar(
                    canvas,
                    physio_sample.probability if physio_sample else None,
                    "Physio",
                    rect=(bars_x, bar2_y, bars_w, bar_h),
                    ok=fusion_result.physio_ok,
                )
                draw_prob_bar(
                    canvas,
                    fusion_result.probability,
                    "Fusion",
                    rect=(bars_x, bar3_y, bars_w, bar_h),
                    ok=(fusion_result.visual_ok or fusion_result.physio_ok),
                )
                if alert_active:
                    draw_text(
                        canvas,
                        "!! DROWSINESS ALERT !!",
                        760,
                        bottom_y + 60,
                        0.85,
                        (0, 0, 255),
                        2,
                    )
                if focus_mode_active:
                    focus_msg = f"You are driving in focus for {focus_low_prob_sec:.1f} seconds!!!"
                    # Neon slideshow effect: pulse scale + continuously cycling colors.
                    pulse = 1.25 + 0.28 * math.sin(wall_now * 6.0)
                    scale = max(0.9, pulse)
                    r = int(127 + 128 * math.sin(wall_now * 3.8 + 0.0))
                    g = int(127 + 128 * math.sin(wall_now * 3.8 + 2.1))
                    b = int(127 + 128 * math.sin(wall_now * 3.8 + 4.2))
                    main_color = (b, g, r)
                    glow_color = (255 - b // 3, 255 - g // 3, 255 - r // 3)

                    (msg_w, msg_h), _ = cv2.getTextSize(focus_msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
                    msg_x = max(10, (args.width - msg_w) // 2)
                    msg_y = max(msg_h + 10, args.height // 2)
                    draw_text(canvas, focus_msg, msg_x + 4, msg_y + 4, scale, (0, 0, 0), 9)
                    draw_text(canvas, focus_msg, msg_x, msg_y, scale, glow_color, 6)
                    draw_text(canvas, focus_msg, msg_x, msg_y, scale, main_color, 3)

                draw_text(
                    canvas,
                    "Controls: q=quit  space=pause  +/-=speed(1x/5x/10x)  r=reset  1/2/3=physio mode  4=webcam  5=upload image",
                    34,
                    args.height - 32,
                    0.55,
                    (180, 180, 180),
                )
                draw_text(
                    canvas,
                    f"Focus streak (<{args.focus_threshold:.2f}): {focus_low_prob_sec:4.1f}s",
                    34,
                    args.height - 10,
                    0.6,
                    (200, 230, 200),
                )

                if alert_active and (wall_now - last_beep_wall) >= ALERT_BEEP_COOLDOWN_SEC:
                    play_beep()
                    last_beep_wall = wall_now

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord(" "):
                    sensor.toggle_pause()
                if key in (ord("+"), ord("=")):
                    current_idx = PLAYBACK_SPEED_PRESETS.index(
                        min(PLAYBACK_SPEED_PRESETS, key=lambda s: abs(s - sensor.playback_speed))
                    )
                    next_idx = (current_idx + 1) % len(PLAYBACK_SPEED_PRESETS)
                    sensor.set_playback_speed(PLAYBACK_SPEED_PRESETS[next_idx])
                if key == ord("-"):
                    current_idx = PLAYBACK_SPEED_PRESETS.index(
                        min(PLAYBACK_SPEED_PRESETS, key=lambda s: abs(s - sensor.playback_speed))
                    )
                    prev_idx = (current_idx - 1) % len(PLAYBACK_SPEED_PRESETS)
                    sensor.set_playback_speed(PLAYBACK_SPEED_PRESETS[prev_idx])
                if key == ord("r"):
                    sensor.reset()
                    sensor.start(reset=False)
                    reset_runtime_state()
                if key == ord("1"):
                    switch_physio_source(args.alert_edf, "alert")
                if key == ord("2"):
                    switch_physio_source(args.drowsy_edf, "drowsy")
                if key == ord("3"):
                    switch_physio_source(args.sample_edf, "sample")
                if key == ord("4"):
                    switch_visual_to_webcam()
                if key == ord("5"):
                    selected = choose_image_file_dialog()
                    if selected is not None:
                        try:
                            switch_visual_to_image(selected)
                            visual_buffer.clear()
                        except (FileNotFoundError, ValueError) as exc:
                            print(f"[WARN] {exc}")
    finally:
        stop_focus_music()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        sys.exit(1)

