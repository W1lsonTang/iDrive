import argparse
import datetime
import os
import platform
import sys
import time
from collections import deque

try:
    import cv2
    import joblib
    import mediapipe as mp
    import numpy as np
except ModuleNotFoundError as exc:
    missing_module = exc.name
    raise SystemExit(
        "Missing required dependency: "
        f"'{missing_module}'.\n"
        "Please install dependencies first:\n"
        "pip install opencv-python mediapipe numpy joblib scikit-learn"
    ) from exc

try:
    import winsound
except Exception:
    winsound = None


LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
HEAD_POSE_IDX = [1, 152, 263, 33, 61, 291]  # nose, chin, left eye, right eye, left mouth, right mouth
HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26.0),  # Left eye corner
        (43.3, 32.7, -26.0),  # Right eye corner
        (-28.9, -28.9, -24.1),  # Left mouth corner
        (28.9, -28.9, -24.1),  # Right mouth corner
    ],
    dtype=np.float64,
)

WINDOW_SIZE = 50
ALERT_RATIO_THRESHOLD = 0.85
ALERT_BEEP_COOLDOWN_SEC = 2.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time webcam drowsiness detection prototype."
    )
    parser.add_argument(
        "--model",
        default=os.path.join("output", "rf_model.pkl"),
        help="Path to initial model file (default: output/rf_model.pkl).",
    )
    parser.add_argument(
        "--scaler",
        default=os.path.join("output", "scaler.pkl"),
        help="Path to scaler file (default: output/scaler.pkl).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture (default: 0).",
    )
    return parser.parse_args()


def get_face_mesh_class():
    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "face_mesh"):
        mp_version = getattr(mp, "__version__", "unknown")
        raise ValueError(
            "Installed mediapipe does not provide Face Mesh Solutions API "
            "(mp.solutions.face_mesh).\n"
            f"Detected mediapipe version: {mp_version}\n"
            "Recommended fix: install a mediapipe version that exposes "
            "mp.solutions.face_mesh (for example, 0.10.14 in Python 3.9)."
        )
    return solutions.face_mesh.FaceMesh


def landmark_to_xy(landmarks, index, image_w, image_h):
    point = landmarks[index]
    return np.array([point.x * image_w, point.y * image_h], dtype=np.float64)


def compute_ear_for_eye(landmarks, eye_indices, image_w, image_h):
    p1 = landmark_to_xy(landmarks, eye_indices[0], image_w, image_h)
    p2 = landmark_to_xy(landmarks, eye_indices[1], image_w, image_h)
    p3 = landmark_to_xy(landmarks, eye_indices[2], image_w, image_h)
    p4 = landmark_to_xy(landmarks, eye_indices[3], image_w, image_h)
    p5 = landmark_to_xy(landmarks, eye_indices[4], image_w, image_h)
    p6 = landmark_to_xy(landmarks, eye_indices[5], image_w, image_h)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return None
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_average_ear(landmarks, image_w, image_h):
    left_ear = compute_ear_for_eye(landmarks, LEFT_EYE_IDX, image_w, image_h)
    right_ear = compute_ear_for_eye(landmarks, RIGHT_EYE_IDX, image_w, image_h)
    if left_ear is None or right_ear is None:
        return None
    return float((left_ear + right_ear) / 2.0)


def estimate_pitch_yaw(landmarks, image_w, image_h):
    image_points = np.array(
        [landmark_to_xy(landmarks, idx, image_w, image_h) for idx in HEAD_POSE_IDX],
        dtype=np.float64,
    )

    focal_length = float(image_w)
    center = (image_w / 2.0, image_h / 2.0)
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
        return None, None

    _ = translation_vec  # Not used in this pipeline.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    pitch = float(angles[0])
    yaw = float(angles[1])
    return pitch, yaw


def ensure_file_exists(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def infer_model_name(path):
    name = os.path.basename(path).lower()
    if "svm" in name:
        return "SVM"
    return "RF"


def load_models(initial_model_path):
    models = {}
    initial_name = infer_model_name(initial_model_path)
    models[initial_name] = joblib.load(initial_model_path)

    counterpart_name = "SVM" if initial_name == "RF" else "RF"
    counterpart_file = "svm_model.pkl" if initial_name == "RF" else "rf_model.pkl"
    counterpart_path = os.path.join(os.path.dirname(initial_model_path), counterpart_file)

    if os.path.isfile(counterpart_path):
        try:
            models[counterpart_name] = joblib.load(counterpart_path)
            print(f"[INFO] Loaded backup model: {counterpart_name} ({counterpart_path})")
        except Exception as exc:
            print(
                f"[WARN] Failed to load backup model at {counterpart_path}: {exc}. "
                f"Model switching may be unavailable."
            )
    else:
        print(
            f"[WARN] Backup model not found at {counterpart_path}. "
            f"Model switching may be unavailable."
        )

    return models, initial_name


def to_pixel(point, image_w, image_h):
    x = int(point.x * image_w)
    y = int(point.y * image_h)
    return x, y


def draw_face_mesh_overlay(frame, landmarks, image_w, image_h):
    overlay = frame.copy()
    connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
    for start_idx, end_idx in connections:
        p1 = to_pixel(landmarks[start_idx], image_w, image_h)
        p2 = to_pixel(landmarks[end_idx], image_w, image_h)
        cv2.line(overlay, p1, p2, (0, 180, 0), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def draw_eye_highlights(frame, landmarks, image_w, image_h):
    eye_color = (255, 255, 0)  # Cyan in BGR

    for eye_indices in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        eye_points = [to_pixel(landmarks[idx], image_w, image_h) for idx in eye_indices]
        for point in eye_points:
            cv2.circle(frame, point, 2, eye_color, -1, cv2.LINE_AA)
        for i in range(len(eye_points)):
            p1 = eye_points[i]
            p2 = eye_points[(i + 1) % len(eye_points)]
            cv2.line(frame, p1, p2, eye_color, 2, cv2.LINE_AA)


def draw_prediction_banner(frame, status_text):
    if status_text == "DROWSY":
        bg_color = (0, 0, 255)
        text_color = (255, 255, 255)
    elif status_text == "ALERT":
        bg_color = (0, 180, 0)
        text_color = (255, 255, 255)
    else:
        bg_color = (60, 60, 60)
        text_color = (255, 255, 255)

    x1, y1, x2, y2 = 20, 20, 320, 90
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(
        frame,
        status_text,
        (35, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        text_color,
        3,
        cv2.LINE_AA,
    )


def draw_feature_panel(frame, ear, pitch, yaw):
    frame_h, frame_w = frame.shape[:2]
    panel_w = 290
    x1 = frame_w - panel_w - 20
    y1 = 20
    x2 = frame_w - 20
    y2 = 140

    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

    ear_text = f"EAR: {ear:.2f}" if ear is not None else "EAR: --"
    pitch_text = f"Pitch: {pitch:.1f} deg" if pitch is not None else "Pitch: --"
    yaw_text = f"Yaw: {yaw:.1f} deg" if yaw is not None else "Yaw: --"

    cv2.putText(frame, ear_text, (x1 + 12, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, pitch_text, (x1 + 12, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, yaw_text, (x1 + 12, y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)


def draw_footer(frame, fps, model_name):
    frame_h = frame.shape[0]
    cv2.putText(
        frame,
        f"Model: {model_name}",
        (20, frame_h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, frame_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_no_face_message(frame):
    frame_h, frame_w = frame.shape[:2]
    text = "No face detected"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x = max((frame_w - text_size[0]) // 2, 10)
    y = max(frame_h - 30, 30)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )


def draw_drowsiness_alert(frame, should_flash):
    if should_flash:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 14)

    text = "DROWSINESS ALERT"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    x = max((frame.shape[1] - text_size[0]) // 2, 10)
    y = frame.shape[0] // 2
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )


def play_beep():
    if winsound is not None and platform.system().lower().startswith("win"):
        winsound.Beep(1800, 220)
    else:
        print("\a", end="", flush=True)


def save_screenshot(frame, screenshot_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
    cv2.imwrite(path, frame)
    print(f"[INFO] Screenshot saved: {path}")


def main():
    args = parse_args()
    model_path = os.path.abspath(args.model)
    scaler_path = os.path.abspath(args.scaler)
    screenshot_dir = os.path.join(os.getcwd(), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    ensure_file_exists(model_path, "Model file")
    ensure_file_exists(scaler_path, "Scaler file")

    scaler = joblib.load(scaler_path)
    models, current_model_name = load_models(model_path)
    available_model_names = [name for name in ("RF", "SVM") if name in models]

    if not available_model_names:
        raise ValueError("No model loaded successfully.")

    print(f"[INFO] Loaded models: {', '.join(available_model_names)}")
    print(f"[INFO] Initial model: {current_model_name}")
    print("[INFO] Controls: q=quit, s=screenshot, m=toggle model")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam (index={args.camera}).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    face_mesh_class = get_face_mesh_class()

    prediction_window = deque(maxlen=WINDOW_SIZE)
    last_beep_time = 0.0
    prev_time = time.time()
    fps = 0.0

    try:
        with face_mesh_class(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] Failed to read frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                image_h, image_w = frame.shape[:2]

                now = time.time()
                frame_dt = max(now - prev_time, 1e-6)
                prev_time = now
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)

                ear = None
                pitch = None
                yaw = None
                status_text = "NO FACE"
                has_valid_prediction = False

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(frame_rgb)

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    draw_face_mesh_overlay(frame, landmarks, image_w, image_h)
                    draw_eye_highlights(frame, landmarks, image_w, image_h)

                    ear = compute_average_ear(landmarks, image_w, image_h)
                    pitch, yaw = estimate_pitch_yaw(landmarks, image_w, image_h)

                    if ear is not None and pitch is not None and yaw is not None:
                        feature = np.array([[ear, pitch, yaw]], dtype=np.float64)
                        feature_scaled = scaler.transform(feature)
                        prediction = int(models[current_model_name].predict(feature_scaled)[0])
                        prediction_window.append(prediction)
                        has_valid_prediction = True
                    else:
                        draw_no_face_message(frame)
                else:
                    draw_no_face_message(frame)

                drowsy_ratio = (
                    float(sum(prediction_window)) / float(len(prediction_window))
                    if prediction_window
                    else 0.0
                )
                if has_valid_prediction and prediction_window:
                    # Show a stable status based on rolling-window majority vote.
                    status_text = "DROWSY" if drowsy_ratio >= 0.5 else "ALERT"

                alert_active = (
                    len(prediction_window) == prediction_window.maxlen
                    and drowsy_ratio > ALERT_RATIO_THRESHOLD
                )

                if alert_active:
                    should_flash = int(time.time() * 4) % 2 == 0
                    draw_drowsiness_alert(frame, should_flash)
                    if time.time() - last_beep_time >= ALERT_BEEP_COOLDOWN_SEC:
                        play_beep()
                        last_beep_time = time.time()

                draw_prediction_banner(frame, status_text)
                draw_feature_panel(frame, ear, pitch, yaw)
                draw_footer(frame, fps, current_model_name)

                cv2.imshow("IDrive Webcam Drowsiness Prototype", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("s"):
                    save_screenshot(frame, screenshot_dir)
                if key == ord("m"):
                    if len(available_model_names) < 2:
                        print("[WARN] Cannot toggle model: only one model is available.")
                    else:
                        current_index = available_model_names.index(current_model_name)
                        next_index = (current_index + 1) % len(available_model_names)
                        current_model_name = available_model_names[next_index]
                        print(f"[INFO] Switched model to: {current_model_name}")
    finally:
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
