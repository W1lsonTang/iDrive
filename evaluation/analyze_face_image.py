"""Analyze a single image with the visual channel pipeline.

Given an input image, this script:
  1. Runs MediaPipe Face Mesh and extracts 468 landmarks.
  2. Computes EAR (Eye Aspect Ratio), Pitch, Yaw.
  3. Optionally loads the trained Random Forest model + scaler to predict
     the drowsiness probability P_v.
  4. Saves an annotated image (face mesh + eye landmarks + head pose axes
     + a side panel with the metric values) and prints the metrics to stdout.

Usage:
    python evaluation/analyze_face_image.py --image my_face.jpg
    python evaluation/analyze_face_image.py --image my_face.jpg --no-mesh
    python evaluation/analyze_face_image.py --image my_face.jpg --output annotated.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
HEAD_POSE_IDX = [1, 152, 263, 33, 61, 291]
HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),       # nose tip
        (0.0, -63.6, -12.5),   # chin
        (-43.3, 32.7, -26.0),  # left eye corner
        (43.3, 32.7, -26.0),   # right eye corner
        (-28.9, -28.9, -24.1), # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual channel on a single image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", default=None, help="Annotated output PNG path (default: <image>_annotated.png).")
    parser.add_argument("--rf-model", default="output/rf_model.pkl")
    parser.add_argument("--scaler", default="output/scaler.pkl")
    parser.add_argument("--no-mesh", action="store_true", help="Hide the dense face mesh overlay.")
    parser.add_argument("--no-model", action="store_true", help="Skip loading RF model (only show features).")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize image if wider than this (preserves aspect).")
    parser.add_argument("--min-height", type=int, default=520, help="Upscale image if shorter than this (preserves aspect).")
    return parser.parse_args()


def landmark_to_xy(landmarks, index: int, w: int, h: int) -> np.ndarray:
    p = landmarks[index]
    return np.array([p.x * w, p.y * h], dtype=np.float64)


def compute_ear_for_eye(landmarks, eye_indices, w: int, h: int) -> Optional[float]:
    pts = [landmark_to_xy(landmarks, idx, w, h) for idx in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
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
    return (left + right) / 2.0, left, right


def estimate_pitch_yaw(landmarks, w: int, h: int) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
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
    success, rotation_vec, translation_vec = cv2.solvePnP(
        HEAD_POSE_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    return float(angles[0]), float(angles[1]), (rotation_vec, translation_vec, camera_matrix, dist_coeffs)


def predict_visual_proba(model, scaler, ear: float, pitch: float, yaw: float) -> float:
    feature = np.array([[ear, pitch, yaw]], dtype=np.float64)
    feature_scaled = scaler.transform(feature)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(feature_scaled)[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(feature_scaled)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    return float(int(model.predict(feature_scaled)[0]))


def draw_face_mesh(image: np.ndarray, face_landmarks, w: int, h: int) -> None:
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
    )


def draw_eye_points(image: np.ndarray, landmarks, eye_idx, w: int, h: int, color=(0, 0, 255)) -> None:
    for i, idx in enumerate(eye_idx):
        x, y = landmark_to_xy(landmarks, idx, w, h).astype(int)
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"p{i+1}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_pose_axes(image: np.ndarray, pose_extras, w: int, h: int) -> None:
    rotation_vec, translation_vec, camera_matrix, dist_coeffs = pose_extras
    axis_length = 80.0
    axis_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (axis_length, 0.0, 0.0),
            (0.0, axis_length, 0.0),
            (0.0, 0.0, axis_length),
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(axis_points, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)
    origin = tuple(projected[0])
    cv2.line(image, origin, tuple(projected[1]), (0, 0, 255), 3, cv2.LINE_AA)   # X red
    cv2.line(image, origin, tuple(projected[2]), (0, 255, 0), 3, cv2.LINE_AA)   # Y green
    cv2.line(image, origin, tuple(projected[3]), (255, 80, 0), 3, cv2.LINE_AA)  # Z blue


def render_panel(image: np.ndarray, metrics: dict) -> np.ndarray:
    h, w = image.shape[:2]
    panel_w = 360
    panel = np.full((h, panel_w, 3), 30, dtype=np.uint8)

    def put(text, x, y, scale=0.6, color=(230, 230, 230), thickness=1):
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    put("Visual Channel Output", 16, 32, 0.7, (255, 255, 255), 2)
    cv2.line(panel, (16, 44), (panel_w - 16, 44), (90, 90, 90), 1)

    y = 80
    if metrics["ear"] is not None:
        put(f"EAR (avg) : {metrics['ear']:.4f}", 20, y, 0.6, (180, 220, 255))
        y += 26
        put(f"  Left  : {metrics['ear_left']:.4f}", 20, y, 0.5)
        y += 22
        put(f"  Right : {metrics['ear_right']:.4f}", 20, y, 0.5)
        y += 30
    else:
        put("EAR : N/A", 20, y, 0.6, (180, 180, 180))
        y += 26

    if metrics["pitch"] is not None:
        put(f"Pitch : {metrics['pitch']:.2f} deg", 20, y, 0.6, (180, 220, 255))
        y += 26
        put(f"Yaw   : {metrics['yaw']:.2f} deg", 20, y, 0.6, (180, 220, 255))
        y += 30
    else:
        put("Pose : N/A", 20, y, 0.6, (180, 180, 180))
        y += 26

    cv2.line(panel, (16, y), (panel_w - 16, y), (90, 90, 90), 1)
    y += 26

    if metrics["p_v"] is not None:
        color = (80, 80, 255) if metrics["p_v"] >= 0.5 else (80, 200, 100)
        verdict = "DROWSY" if metrics["p_v"] >= 0.5 else "ALERT"
        put(f"P_v       : {metrics['p_v']:.4f}", 20, y, 0.7, (255, 255, 255), 2)
        y += 32
        put(f"Verdict   : {verdict}", 20, y, 0.7, color, 2)
        y += 36

        bar_x, bar_y, bar_w, bar_h = 20, y, panel_w - 40, 22
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)
        fill_w = int(bar_w * float(np.clip(metrics["p_v"], 0.0, 1.0)))
        cv2.rectangle(panel, (bar_x + 1, bar_y + 1), (bar_x + 1 + fill_w, bar_y + bar_h - 1), color, -1)
        y += bar_h + 16
    else:
        put("P_v : (model not loaded)", 20, y, 0.55, (180, 180, 180))
        y += 24

    return np.hstack([image, panel])


def main() -> None:
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"Image not found: {img_path}")

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise SystemExit(f"OpenCV failed to read: {img_path}")

    if image_bgr.shape[1] > args.max_width:
        scale = args.max_width / image_bgr.shape[1]
        new_w = args.max_width
        new_h = int(image_bgr.shape[0] * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if image_bgr.shape[0] < args.min_height:
        scale = args.min_height / image_bgr.shape[0]
        new_h = args.min_height
        new_w = int(image_bgr.shape[1] * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_mesh_class = mp.solutions.face_mesh.FaceMesh
    with face_mesh_class(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("[ERROR] No face detected in the image.")
        annotated = image_bgr.copy()
        cv2.putText(annotated, "NO FACE DETECTED", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        out_path = Path(args.output) if args.output else img_path.with_name(img_path.stem + "_annotated.png")
        cv2.imwrite(str(out_path), annotated)
        print(f"[INFO] Saved (no-face): {out_path}")
        return

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark

    ear_result = compute_average_ear(landmarks, w, h)
    if ear_result is None:
        ear, ear_left, ear_right = None, None, None
    else:
        ear, ear_left, ear_right = ear_result

    pitch, yaw, pose_extras = estimate_pitch_yaw(landmarks, w, h)

    p_v = None
    if not args.no_model and ear is not None and pitch is not None:
        rf_path = Path(args.rf_model)
        scaler_path = Path(args.scaler)
        if rf_path.is_file() and scaler_path.is_file():
            try:
                model = joblib.load(rf_path)
                scaler = joblib.load(scaler_path)
                p_v = predict_visual_proba(model, scaler, ear, pitch, yaw)
            except Exception as exc:
                print(f"[WARN] Failed to load model: {exc}")
        else:
            print(f"[WARN] Model files not found ({rf_path}, {scaler_path}); skipping P_v.")

    annotated = image_bgr.copy()
    if not args.no_mesh:
        draw_face_mesh(annotated, face_landmarks, w, h)

    draw_eye_points(annotated, landmarks, LEFT_EYE_IDX, w, h, color=(0, 100, 255))
    draw_eye_points(annotated, landmarks, RIGHT_EYE_IDX, w, h, color=(0, 200, 255))

    for idx in HEAD_POSE_IDX:
        x, y = landmark_to_xy(landmarks, idx, w, h).astype(int)
        cv2.circle(annotated, (x, y), 5, (255, 200, 0), -1, cv2.LINE_AA)

    if pose_extras is not None:
        draw_pose_axes(annotated, pose_extras, w, h)

    composed = render_panel(annotated, {
        "ear": ear,
        "ear_left": ear_left if ear_left is not None else float("nan"),
        "ear_right": ear_right if ear_right is not None else float("nan"),
        "pitch": pitch,
        "yaw": yaw,
        "p_v": p_v,
    })

    out_path = Path(args.output) if args.output else img_path.with_name(img_path.stem + "_annotated.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composed)

    print("=" * 56)
    print(f"Image          : {img_path}")
    print(f"Resized to     : {w} x {h}")
    print("-" * 56)
    if ear is not None:
        print(f"EAR (avg)      : {ear:.4f}")
        print(f"  EAR left     : {ear_left:.4f}")
        print(f"  EAR right    : {ear_right:.4f}")
    else:
        print("EAR            : N/A (eye landmarks invalid)")
    if pitch is not None:
        print(f"Pitch (deg)    : {pitch:+.2f}")
        print(f"Yaw   (deg)    : {yaw:+.2f}")
    else:
        print("Pose           : N/A (solvePnP failed)")
    if p_v is not None:
        verdict = "DROWSY" if p_v >= 0.5 else "ALERT"
        print(f"P_v (drowsy)   : {p_v:.4f}  ->  {verdict}")
    print("-" * 56)
    print(f"Annotated img  : {out_path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
