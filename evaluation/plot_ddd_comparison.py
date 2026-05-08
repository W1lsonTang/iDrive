"""Generate a DDD drowsy vs non-drowsy comparison figure.

Each panel shows: Face Mesh landmarks, EAR / Pitch / Yaw, and BOTH
RF and SVM confidence scores (P_v) side-by-side for direct comparison.

Usage:
    python evaluation/plot_ddd_comparison.py
    python evaluation/plot_ddd_comparison.py --drowsy-img "path" --nondrowsy-img "path"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
HEAD_POSE_IDX = [1, 152, 263, 33, 61, 291]
HEAD_POSE_MODEL_POINTS = np.array([
    (0.0,   0.0,   0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    ( 43.3, 32.7, -26.0),
    (-28.9,-28.9, -24.1),
    ( 28.9,-28.9, -24.1),
], dtype=np.float64)

MIN_DISPLAY_H = 480


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--drowsy-img",    default=None)
    p.add_argument("--nondrowsy-img", default=None)
    p.add_argument("--ddd-dir",   default="Driver Drowsiness Dataset (DDD)")
    p.add_argument("--rf-model",  default="output/rf_model.pkl")
    p.add_argument("--svm-model", default="output/svm_model.pkl")
    p.add_argument("--scaler",    default="output/scaler.pkl")
    p.add_argument("--output",    default="output/ddd_comparison.png")
    return p.parse_args()


# ── geometry helpers ────────────────────────────────────────────────────────

def lm_xy(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float64)


def ear_eye(lm, idx, w, h):
    pts = [lm_xy(lm, i, w, h) for i in idx]
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    ho = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2 * ho)) if ho else None


def compute_ear(lm, w, h):
    l = ear_eye(lm, LEFT_EYE_IDX,  w, h)
    r = ear_eye(lm, RIGHT_EYE_IDX, w, h)
    if l is None or r is None:
        return None, None, None
    return (l + r) / 2, l, r


def compute_pose(lm, w, h):
    img_pts = np.array([lm_xy(lm, i, w, h) for i in HEAD_POSE_IDX], dtype=np.float64)
    fl  = float(w)
    cam = np.array([[fl, 0, w/2], [0, fl, h/2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(HEAD_POSE_MODEL_POINTS, img_pts, cam, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    rm, _ = cv2.Rodrigues(rvec)
    ang, *_ = cv2.RQDecomp3x3(rm)
    return float(ang[0]), float(ang[1]), (rvec, tvec, cam, dist)


def predict_pv(model, scaler, ear, pitch, yaw) -> float:
    x  = np.array([[ear, pitch, yaw]], dtype=np.float64)
    xs = scaler.transform(x)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(xs)[0, 1])
    s = float(model.decision_function(xs)[0])
    return float(1 / (1 + np.exp(-s)))


# ── annotation ──────────────────────────────────────────────────────────────

def annotate(img_bgr: np.ndarray, rf_model, svm_model, scaler, face_mesh) -> Tuple[np.ndarray, dict]:
    h, w = img_bgr.shape[:2]
    if h < MIN_DISPLAY_H:
        scale   = MIN_DISPLAY_H / h
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    out = img_bgr.copy()
    metrics = {"ear": None, "ear_l": None, "ear_r": None,
               "pitch": None, "yaw": None, "pv_rf": None, "pv_svm": None}

    if not res.multi_face_landmarks:
        cv2.putText(out, "NO FACE", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        return out, metrics

    lm = res.multi_face_landmarks[0].landmark

    # draw mesh
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_fm     = mp.solutions.face_mesh
    mp_draw.draw_landmarks(out, res.multi_face_landmarks[0],
                           mp_fm.FACEMESH_TESSELATION, None,
                           mp_styles.get_default_face_mesh_tesselation_style())
    mp_draw.draw_landmarks(out, res.multi_face_landmarks[0],
                           mp_fm.FACEMESH_CONTOURS, None,
                           mp_styles.get_default_face_mesh_contours_style())

    # eye & pose key points
    for idx in LEFT_EYE_IDX:
        x, y = lm_xy(lm, idx, w, h).astype(int)
        cv2.circle(out, (x, y), 4, (0, 80, 255), -1, cv2.LINE_AA)
    for idx in RIGHT_EYE_IDX:
        x, y = lm_xy(lm, idx, w, h).astype(int)
        cv2.circle(out, (x, y), 4, (0, 180, 255), -1, cv2.LINE_AA)
    for idx in HEAD_POSE_IDX:
        x, y = lm_xy(lm, idx, w, h).astype(int)
        cv2.circle(out, (x, y), 5, (255, 200, 0), -1, cv2.LINE_AA)

    ear, ear_l, ear_r  = compute_ear(lm, w, h)
    pitch, yaw, pose_ex = compute_pose(lm, w, h)

    # head-pose axes
    if pose_ex:
        rvec, tvec, cam, dist = pose_ex
        ax_pts = np.array([(0,0,0),(80,0,0),(0,80,0),(0,0,80)], dtype=np.float64)
        proj, _ = cv2.projectPoints(ax_pts, rvec, tvec, cam, dist)
        proj     = proj.reshape(-1, 2).astype(int)
        orig     = tuple(proj[0])
        cv2.line(out, orig, tuple(proj[1]), (0,   0, 220), 3, cv2.LINE_AA)
        cv2.line(out, orig, tuple(proj[2]), (0, 220,   0), 3, cv2.LINE_AA)
        cv2.line(out, orig, tuple(proj[3]), (220,100,   0), 3, cv2.LINE_AA)

    pv_rf = pv_svm = None
    if ear is not None and pitch is not None:
        if rf_model  is not None:
            pv_rf  = predict_pv(rf_model,  scaler, ear, pitch, yaw)
        if svm_model is not None:
            pv_svm = predict_pv(svm_model, scaler, ear, pitch, yaw)

    metrics = {"ear": ear, "ear_l": ear_l, "ear_r": ear_r,
               "pitch": pitch, "yaw": yaw,
               "pv_rf": pv_rf, "pv_svm": pv_svm}

    # semi-transparent bottom overlay (taller to fit two confidence rows)
    bg_h    = 175
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bg_h), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

    def put(text, x, y, color=(255, 255, 255), scale=0.55, thick=1):
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    y0, dy = h - bg_h + 22, 26

    if ear is not None:
        put(f"EAR: {ear:.3f}  (L:{ear_l:.3f}  R:{ear_r:.3f})", 10, y0, (180, 220, 255))
    if pitch is not None:
        put(f"Pitch: {pitch:+.1f} deg     Yaw: {yaw:+.1f} deg", 10, y0 + dy, (180, 220, 255))

    # row offsets: RF starts at row 2, SVM at row 4 (extra gap between them)
    ROW_OFFSETS = {0: 2, 1: 4}

    def draw_conf_row(pv, label_str, row_idx):
        verdict = "DROWSY" if pv >= 0.5 else "ALERT"
        col_txt = (80, 80, 255) if pv >= 0.5 else (80, 220, 80)
        y_text  = y0 + ROW_OFFSETS[row_idx] * dy
        put(f"{label_str}  P_v={pv:.3f}  ->  {verdict}", 10, y_text, col_txt, 0.6, 2)
        bx = 10
        by = y_text + 6
        bw = w - 20
        bh = 11
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (70, 70, 70), 1)
        fw = int(bw * float(np.clip(pv, 0, 1)))
        cv2.rectangle(out, (bx + 1, by + 1), (bx + 1 + fw, by + bh - 1), col_txt, -1)

    if pv_rf is not None:
        draw_conf_row(pv_rf,  "[RF ]", 0)
    if pv_svm is not None:
        draw_conf_row(pv_svm, "[SVM]", 1)

    return out, metrics


# ── figure layout ────────────────────────────────────────────────────────────

def make_figure(img_drowsy, img_nd, metrics_d, metrics_nd, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.patch.set_facecolor("#1a1a1a")

    titles  = ["Drowsy Sample  (label = 1)", "Non-Drowsy Sample  (label = 0)"]
    images  = [img_drowsy, img_nd]
    metrics = [metrics_d, metrics_nd]
    colors  = ["#FF5252", "#69F0AE"]

    for ax, title, img, met, col in zip(axes, titles, images, metrics, colors):
        ax.set_facecolor("#1a1a1a")
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color=col, fontsize=13, fontweight="bold", pad=6)
        ax.axis("off")

        lines = []
        if met["ear"] is not None:
            lines.append(f"EAR = {met['ear']:.3f}   Pitch = {met['pitch']:+.1f}°   Yaw = {met['yaw']:+.1f}°")
        for model_name, key in [("RF ", "pv_rf"), ("SVM", "pv_svm")]:
            if met[key] is not None:
                v = met[key]
                verdict = "DROWSY" if v >= 0.5 else "ALERT"
                lines.append(f"[{model_name}]  P_v = {v:.3f}  →  {verdict}")
        ax.set_xlabel("\n".join(lines), color="#dddddd", fontsize=10,
                      labelpad=6, ha="center", family="monospace")

    fig.suptitle(
        "DDD Dataset — Visual Channel Feature Extraction & Model Comparison\n"
        "MediaPipe 468-point Face Mesh  →  EAR / Pitch / Yaw  →  RF vs SVM  →  P_v",
        color="white", fontsize=12, fontweight="bold"
    )

    # performance table (RF vs SVM)
    table_ax = fig.add_axes([0.12, 0.01, 0.76, 0.10])
    table_ax.set_facecolor("#111111")
    table_ax.axis("off")
    col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1 (drowsy)", "ROC-AUC", "Selected"]
    rows = [
        ["SVM (RBF)",     "0.771", "0.765", "0.820", "0.792", "0.843", ""],
        ["Random Forest", "0.853", "0.857", "0.867", "0.862", "0.929", "✓"],
    ]
    tbl = table_ax.table(cellText=rows, colLabels=col_labels,
                         loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1e1e1e" if r % 2 == 0 else "#2a2a2a")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#444444")
        if r == 0:
            cell.set_facecolor("#303030")
            cell.set_text_props(color="#aaaaaa", fontweight="bold")
        if c == 6 and r == 2:
            cell.set_text_props(color="#69F0AE", fontweight="bold", fontsize=12)

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Saved: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rf_model  = scaler = svm_model = None
    if Path(args.scaler).is_file():
        scaler = joblib.load(args.scaler)
    if Path(args.rf_model).is_file():
        rf_model = joblib.load(args.rf_model)
    else:
        print("[WARN] RF model not found; RF P_v will not be shown.")
    if Path(args.svm_model).is_file():
        svm_model = joblib.load(args.svm_model)
    else:
        print("[WARN] SVM model not found; SVM P_v will not be shown.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5)

    def load_and_annotate(path, label):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        ann, met = annotate(img, rf_model, svm_model, scaler, face_mesh)
        print(f"[INFO] {label}: {Path(path).name}  "
              f"EAR={met['ear']:.3f}  "
              f"RF P_v={met['pv_rf']}  SVM P_v={met['pv_svm']}")
        return ann, met

    def find_good(folder, label):
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            img  = cv2.imread(path)
            if img is None:
                continue
            ann, met = annotate(img, rf_model, svm_model, scaler, face_mesh)
            if met["ear"] is not None:
                print(f"[INFO] {label}: {fname}  "
                      f"EAR={met['ear']:.3f}  "
                      f"RF P_v={met['pv_rf']}  SVM P_v={met['pv_svm']}")
                return ann, met
        return None, {}

    drowsy_dir    = os.path.join(args.ddd_dir, "Drowsy")
    nondrowsy_dir = os.path.join(args.ddd_dir, "Non Drowsy")

    if args.drowsy_img:
        ann_d, met_d = load_and_annotate(args.drowsy_img, "Drowsy")
    else:
        ann_d, met_d = find_good(drowsy_dir, "Drowsy")

    if args.nondrowsy_img:
        ann_nd, met_nd = load_and_annotate(args.nondrowsy_img, "NonDrowsy")
    else:
        ann_nd, met_nd = find_good(nondrowsy_dir, "NonDrowsy")

    face_mesh.close()

    if ann_d is None or ann_nd is None:
        raise SystemExit("[ERROR] Could not find valid face images.")

    make_figure(ann_d, ann_nd, met_d, met_nd, args.output)


if __name__ == "__main__":
    main()
