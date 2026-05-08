from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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


@dataclass
class VideoItem:
    subject: str
    session: str
    video_path: Path
    stream_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract UL-DD visual time series (EAR/Pitch/Yaw).")
    parser.add_argument("--uldd-dir", default="UL-DD", help="Path to UL-DD root directory.")
    parser.add_argument(
        "--output-csv",
        default="output/uldd_visual_series.csv",
        help="Output CSV path for extracted visual features.",
    )
    parser.add_argument(
        "--video-priority",
        default="L3D,R3D,IR,Pose",
        help="Preferred stream order, comma-separated. L3D/R3D are frontal stereo cams with full face; Pose is overhead and not suitable for face mesh.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional limit for quick smoke tests (0 = process all).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log progress every N videos.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSV by skipping already processed subject/session pairs.",
    )
    return parser.parse_args()


def get_face_mesh_class():
    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "face_mesh"):
        mp_version = getattr(mp, "__version__", "unknown")
        raise ValueError(
            "Installed mediapipe does not expose mp.solutions.face_mesh. "
            f"Detected mediapipe version: {mp_version}."
        )
    return solutions.face_mesh.FaceMesh


def landmark_to_xy(landmarks, index: int, w: int, h: int) -> np.ndarray:
    p = landmarks[index]
    return np.array([p.x * w, p.y * h], dtype=np.float64)


def compute_ear_for_eye(landmarks, eye_indices: Sequence[int], w: int, h: int) -> Optional[float]:
    p1 = landmark_to_xy(landmarks, eye_indices[0], w, h)
    p2 = landmark_to_xy(landmarks, eye_indices[1], w, h)
    p3 = landmark_to_xy(landmarks, eye_indices[2], w, h)
    p4 = landmark_to_xy(landmarks, eye_indices[3], w, h)
    p5 = landmark_to_xy(landmarks, eye_indices[4], w, h)
    p6 = landmark_to_xy(landmarks, eye_indices[5], w, h)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal <= 0:
        return None
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def compute_average_ear(landmarks, w: int, h: int) -> Optional[float]:
    left = compute_ear_for_eye(landmarks, LEFT_EYE_IDX, w, h)
    right = compute_ear_for_eye(landmarks, RIGHT_EYE_IDX, w, h)
    if left is None or right is None:
        return None
    return float((left + right) / 2.0)


def estimate_pitch_yaw(landmarks, w: int, h: int) -> tuple[Optional[float], Optional[float]]:
    image_points = np.array([landmark_to_xy(landmarks, idx, w, h) for idx in HEAD_POSE_IDX], dtype=np.float64)

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


def parse_stream_name(video_name: str) -> str:
    stem = Path(video_name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[1].upper()
    return "UNKNOWN"


def choose_video(session_dir: Path, priority: Sequence[str]) -> Optional[Path]:
    if not session_dir.is_dir():
        return None
    mp4_files = sorted(session_dir.glob("*.mp4"))
    if not mp4_files:
        return None
    stream_map: dict[str, Path] = {}
    for p in mp4_files:
        stream_map[parse_stream_name(p.name)] = p
    for stream in priority:
        s = stream.strip().upper()
        if s in stream_map:
            return stream_map[s]
    return mp4_files[0]


def iter_video_items(uldd_dir: Path, priority: Sequence[str]) -> Iterator[VideoItem]:
    base = uldd_dir / "Video_Data" / "Video_Data"
    if not base.is_dir():
        raise FileNotFoundError(f"UL-DD video directory not found: {base}")

    for subject_dir in sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name):
        subject = subject_dir.name
        for session_dir in sorted([d for d in subject_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
            session = session_dir.name
            chosen = choose_video(session_dir, priority)
            if chosen is None:
                continue
            yield VideoItem(
                subject=subject,
                session=session,
                video_path=chosen,
                stream_name=parse_stream_name(chosen.name),
            )


def extract_video_series(video: VideoItem, face_mesh) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if np.isfinite(frame_count) and frame_count > 0:
        duration_sec = int(np.ceil(frame_count / fps))
    else:
        duration_sec = 0

    if duration_sec <= 0:
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return pd.DataFrame()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        duration_sec = 600

    rows: list[dict[str, object]] = []
    try:
        for sec_idx in range(duration_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec_idx * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            ear = float("nan")
            pitch = float("nan")
            yaw = float("nan")
            face_detected = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                _ear = compute_average_ear(landmarks, w, h)
                _pitch, _yaw = estimate_pitch_yaw(landmarks, w, h)
                if _ear is not None and _pitch is not None and _yaw is not None:
                    ear = float(_ear)
                    pitch = float(_pitch)
                    yaw = float(_yaw)
                    face_detected = 1

            rows.append(
                {
                    "subject": video.subject,
                    "session": video.session,
                    "video_file": video.video_path.name,
                    "stream": video.stream_name,
                    "t_sec": float(sec_idx),
                    "ear": ear,
                    "pitch": pitch,
                    "yaw": yaw,
                    "face_detected": face_detected,
                }
            )
    finally:
        cap.release()

    return pd.DataFrame(rows)


def load_processed_pairs(output_csv: Path) -> set[tuple[str, str]]:
    if not output_csv.is_file():
        return set()
    try:
        df = pd.read_csv(output_csv, usecols=["subject", "session"])  # type: ignore[arg-type]
    except Exception:
        return set()
    return set((str(r.subject), str(r.session)) for r in df.itertuples(index=False))


def append_df(output_csv: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    write_header = not output_csv.exists()
    df.to_csv(output_csv, mode="a", index=False, header=write_header)


def main() -> None:
    args = parse_args()
    uldd_dir = Path(args.uldd_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    priority = [x.strip() for x in args.video_priority.split(",") if x.strip()]

    items = list(iter_video_items(uldd_dir, priority=priority))
    if args.max_videos > 0:
        items = items[: args.max_videos]
    if not items:
        raise RuntimeError("No UL-DD videos found to process.")

    processed = load_processed_pairs(output_csv) if args.resume else set()
    if processed:
        items = [it for it in items if (it.subject, it.session) not in processed]
        print(f"[INFO] Resume mode: skip {len(processed)} completed subject/session pairs", flush=True)
    if not items:
        print("[INFO] Nothing to process. Output already complete.", flush=True)
        return

    face_mesh_class = get_face_mesh_class()

    with face_mesh_class(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for idx, item in enumerate(items, start=1):
            df_item = extract_video_series(item, face_mesh=face_mesh)
            append_df(output_csv, df_item)
            if args.log_every > 0 and (idx % args.log_every == 0 or idx == len(items)):
                valid = int(df_item["face_detected"].sum()) if not df_item.empty else 0
                total = len(df_item)
                ratio = valid / total if total else 0.0
                print(
                    f"[INFO] {idx}/{len(items)} {item.subject}-{item.session} {item.stream_name}: "
                    f"{total} seconds, face_ok={valid} ({ratio:.1%})",
                    flush=True,
                )

    out_df = pd.read_csv(output_csv)
    out_df = out_df.sort_values(["subject", "session", "t_sec"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    total = len(out_df)
    face_ok = int(out_df["face_detected"].sum())
    print(f"[INFO] Saved visual series to {output_csv} ({total} rows)", flush=True)
    print(f"[INFO] Overall face detection success: {face_ok}/{total} ({(face_ok / total if total else 0.0):.1%})", flush=True)


if __name__ == "__main__":
    main()
