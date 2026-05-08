import argparse
import os
import sys

try:
    import cv2
    import joblib
    import matplotlib.pyplot as plt
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    missing_module = exc.name
    raise SystemExit(
        "Missing required dependency: "
        f"'{missing_module}'.\n"
        "Please install dependencies first:\n"
        "pip install opencv-python mediapipe numpy pandas scikit-learn "
        "matplotlib seaborn joblib tqdm"
    ) from exc


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
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
FEATURE_COLUMNS = ["ear", "pitch", "yaw"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual channel pipeline for driver drowsiness detection."
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to DDD dataset root folder (contains Drowsy and Non Drowsy).",
    )
    return parser.parse_args()


def validate_source_dirs(src_root):
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    drowsy_dir = os.path.join(src_root, "Drowsy")
    non_drowsy_dir = os.path.join(src_root, "Non Drowsy")

    if not os.path.isdir(drowsy_dir):
        raise FileNotFoundError(f"Class folder not found: {drowsy_dir}")
    if not os.path.isdir(non_drowsy_dir):
        raise FileNotFoundError(f"Class folder not found: {non_drowsy_dir}")

    return {"Drowsy": drowsy_dir, "Non Drowsy": non_drowsy_dir}


def collect_image_entries(class_dirs):
    entries = []
    for class_name, label in (("Drowsy", 1), ("Non Drowsy", 0)):
        class_dir = class_dirs[class_name]
        for filename in sorted(os.listdir(class_dir)):
            file_path = os.path.join(class_dir, filename)
            if not os.path.isfile(file_path):
                continue
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            display_name = f"{class_name}/{filename}"
            entries.append(
                {
                    "path": file_path,
                    "display_name": display_name,
                    "label": label,
                    "class_name": class_name,
                }
            )
    if not entries:
        raise ValueError("No valid images found (.jpg/.jpeg/.png) in source dataset.")
    return entries


def get_face_mesh_class():
    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "face_mesh"):
        mp_version = getattr(mp, "__version__", "unknown")
        raise ValueError(
            "Installed mediapipe does not provide Face Mesh Solutions API "
            "(mp.solutions.face_mesh).\n"
            f"Detected mediapipe version: {mp_version}\n"
            "This usually happens with Tasks-only builds (common on Python 3.13).\n"
            "Recommended fix:\n"
            "1) Use a Python 3.11 or 3.12 virtual environment\n"
            "2) Install dependencies again:\n"
            "   python -m pip install opencv-python mediapipe numpy pandas "
            "scikit-learn matplotlib seaborn joblib tqdm"
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

    _ = translation_vec  # Translation is estimated but not needed for this pipeline.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    pitch = float(angles[0])
    yaw = float(angles[1])
    return pitch, yaw


def extract_features(src_root, output_dir):
    class_dirs = validate_source_dirs(src_root)
    image_entries = collect_image_entries(class_dirs)

    print("[Step 1/3] Starting feature extraction...")
    print(f"Total images found: {len(image_entries)}")

    features = []
    skipped = []
    success_by_class = {"Drowsy": 0, "Non Drowsy": 0}
    skipped_by_class = {"Drowsy": 0, "Non Drowsy": 0}

    face_mesh_class = get_face_mesh_class()
    with face_mesh_class(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        use_tqdm = os.name != "nt"
        if not use_tqdm:
            print("[INFO] Using safe text progress on Windows terminal.")

        iterable = image_entries
        if use_tqdm:
            iterable = tqdm(
                image_entries,
                desc="Extracting features",
                unit="image",
                file=sys.stdout,
                dynamic_ncols=False,
                ascii=True,
            )

        for idx, entry in enumerate(iterable, start=1):
            image_path = entry["path"]
            display_name = entry["display_name"]
            label = entry["label"]
            class_name = entry["class_name"]

            if not use_tqdm and (idx == 1 or idx % 500 == 0 or idx == len(image_entries)):
                print(f"[Progress] {idx}/{len(image_entries)} images processed...")

            image = cv2.imread(image_path)
            if image is None:
                reason = "OpenCV failed to read image"
                skipped.append((display_name, reason))
                skipped_by_class[class_name] += 1
                print(f"[WARN] Skipped {display_name}: {reason}")
                continue

            image_h, image_w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                result = face_mesh.process(image_rgb)
            except Exception as exc:
                reason = f"MediaPipe processing failed ({exc})"
                skipped.append((display_name, reason))
                skipped_by_class[class_name] += 1
                print(f"[WARN] Skipped {display_name}: {reason}")
                continue

            if not result.multi_face_landmarks:
                reason = "No face detected"
                skipped.append((display_name, reason))
                skipped_by_class[class_name] += 1
                print(f"[WARN] Skipped {display_name}: {reason}")
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            ear = compute_average_ear(landmarks, image_w, image_h)
            if ear is None:
                reason = "Failed to compute EAR"
                skipped.append((display_name, reason))
                skipped_by_class[class_name] += 1
                print(f"[WARN] Skipped {display_name}: {reason}")
                continue

            pitch, yaw = estimate_pitch_yaw(landmarks, image_w, image_h)
            if pitch is None or yaw is None:
                reason = "Head pose estimation failed"
                skipped.append((display_name, reason))
                skipped_by_class[class_name] += 1
                print(f"[WARN] Skipped {display_name}: {reason}")
                continue

            features.append(
                {
                    "filename": os.path.relpath(image_path, src_root).replace("\\", "/"),
                    "ear": ear,
                    "pitch": pitch,
                    "yaw": yaw,
                    "label": label,
                }
            )
            success_by_class[class_name] += 1

    if not features:
        raise ValueError("No features were extracted successfully. Cannot continue.")

    features_df = pd.DataFrame(features, columns=["filename", *FEATURE_COLUMNS, "label"])
    features_csv_path = os.path.join(output_dir, "visual_features.csv")
    features_df.to_csv(features_csv_path, index=False)

    skipped_log_path = os.path.join(output_dir, "skipped_images.log")
    with open(skipped_log_path, "w", encoding="utf-8") as file:
        for name, reason in skipped:
            file.write(f"{name}\t{reason}\n")

    print("[Step 1/3] Feature extraction completed.")
    print(f"- Successful: {len(features)}")
    print(f"- Skipped: {len(skipped)}")
    print(
        f"- Drowsy success/skipped: {success_by_class['Drowsy']}/{skipped_by_class['Drowsy']}"
    )
    print(
        f"- Non Drowsy success/skipped: "
        f"{success_by_class['Non Drowsy']}/{skipped_by_class['Non Drowsy']}"
    )
    print(f"- Saved features CSV: {features_csv_path}")
    print(f"- Saved skip log: {skipped_log_path}")

    return features_df, features_csv_path


def evaluate_classifier(model_name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["non-drowsy", "drowsy"],
        zero_division=0,
    )

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)

    return {
        "name": model_name,
        "metrics": metrics,
        "confusion_matrix": cm,
        "report": report,
        "y_score": y_score,
    }


def train_and_evaluate(features_csv_path, output_dir):
    print("[Step 2/3] Starting classifier training and evaluation...")
    data = pd.read_csv(features_csv_path)

    if data.empty:
        raise ValueError("visual_features.csv is empty. Cannot train models.")
    if data["label"].nunique() < 2:
        raise ValueError("visual_features.csv contains only one class. Need both classes to train.")

    x = data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = data["label"].to_numpy(dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    svm_model = SVC(kernel="rbf")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    svm_model.fit(x_train_scaled, y_train)
    rf_model.fit(x_train_scaled, y_train)

    svm_eval = evaluate_classifier("SVM (RBF)", svm_model, x_test_scaled, y_test)
    rf_eval = evaluate_classifier("Random Forest", rf_model, x_test_scaled, y_test)

    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("IDrive Visual Channel Results\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Total samples: {len(data)}\n")
        file.write(f"Train samples: {len(x_train)}\n")
        file.write(f"Test samples: {len(x_test)}\n\n")

        for eval_result in (svm_eval, rf_eval):
            file.write(f"{eval_result['name']}\n")
            file.write("-" * 40 + "\n")
            file.write(f"Accuracy : {eval_result['metrics']['accuracy']:.4f}\n")
            file.write(f"Precision: {eval_result['metrics']['precision']:.4f}\n")
            file.write(f"Recall   : {eval_result['metrics']['recall']:.4f}\n")
            file.write(f"F1-score : {eval_result['metrics']['f1_score']:.4f}\n")
            file.write("Confusion Matrix [rows=true, cols=pred]:\n")
            file.write(f"{eval_result['confusion_matrix']}\n\n")
            file.write("Classification Report:\n")
            file.write(f"{eval_result['report']}\n")
            file.write("\n")

    svm_model_path = os.path.join(output_dir, "svm_model.pkl")
    rf_model_path = os.path.join(output_dir, "rf_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    joblib.dump(svm_model, svm_model_path)
    joblib.dump(rf_model, rf_model_path)
    joblib.dump(scaler, scaler_path)

    print("[Step 2/3] Training and evaluation completed.")
    print(f"- Saved results: {results_path}")
    print(f"- Saved model: {svm_model_path}")
    print(f"- Saved model: {rf_model_path}")
    print(f"- Saved scaler: {scaler_path}")

    return {
        "data": data,
        "y_test": y_test,
        "svm_eval": svm_eval,
        "rf_eval": rf_eval,
    }


def plot_feature_distribution(data, output_dir):
    plot_data = data.copy()
    plot_data["label_name"] = plot_data["label"].map({0: "Non Drowsy", 1: "Drowsy"})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, feature in enumerate(FEATURE_COLUMNS):
        sns.histplot(
            data=plot_data,
            x=feature,
            hue="label_name",
            bins=30,
            kde=True,
            stat="density",
            common_norm=False,
            ax=axes[idx],
            alpha=0.45,
            element="step",
        )
        axes[idx].set_title(f"{feature.upper()} distribution")
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel("Density")

    fig.tight_layout()
    output_path = os.path.join(output_dir, "feature_distribution.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(svm_cm, rf_cm, output_dir):
    labels = ["Non Drowsy", "Drowsy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        svm_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("SVM Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("Random Forest Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.tight_layout()
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_roc_curves(y_test, svm_scores, rf_scores, output_dir):
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_scores)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_scores)
    svm_auc = auc(svm_fpr, svm_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(svm_fpr, svm_tpr, label=f"SVM (AUC={svm_auc:.3f})", linewidth=2)
    ax.plot(rf_fpr, rf_tpr, label=f"RF (AUC={rf_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_visualizations(train_eval_result, output_dir):
    print("[Step 3/3] Generating visualizations...")
    data = train_eval_result["data"]
    svm_cm = train_eval_result["svm_eval"]["confusion_matrix"]
    rf_cm = train_eval_result["rf_eval"]["confusion_matrix"]
    y_test = train_eval_result["y_test"]
    svm_scores = train_eval_result["svm_eval"]["y_score"]
    rf_scores = train_eval_result["rf_eval"]["y_score"]

    feature_plot_path = plot_feature_distribution(data, output_dir)
    cm_plot_path = plot_confusion_matrices(svm_cm, rf_cm, output_dir)
    roc_plot_path = plot_roc_curves(y_test, svm_scores, rf_scores, output_dir)

    print("[Step 3/3] Visualization completed.")
    print(f"- Saved plot: {feature_plot_path}")
    print(f"- Saved plot: {cm_plot_path}")
    print(f"- Saved plot: {roc_plot_path}")


def main():
    args = parse_args()
    src_root = os.path.abspath(args.src)
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("IDrive Visual Channel Pipeline")
    print("=" * 35)
    print(f"Source dataset: {src_root}")
    print(f"Output folder: {output_dir}")

    try:
        _, features_csv_path = extract_features(src_root, output_dir)
        train_eval_result = train_and_evaluate(features_csv_path, output_dir)
        generate_visualizations(train_eval_result, output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        sys.exit(1)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
