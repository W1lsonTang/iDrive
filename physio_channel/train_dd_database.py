"""Train physiological drowsiness models from DD-Database."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .edf_loader import generate_labeled_windows, iter_recording_pairs
from .features import FEATURE_COLUMNS, extract_hrv_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physiological drowsiness models on DD-Database.")
    parser.add_argument("--data-dir", default="DD-Database", help="Path to DD-Database folder.")
    parser.add_argument("--output-dir", default="output", help="Directory to save models and reports.")
    parser.add_argument("--window-sec", type=float, default=60.0, help="Window size in seconds.")
    parser.add_argument("--step-sec", type=float, default=30.0, help="Sliding step in seconds.")
    parser.add_argument("--alert-duration-sec", type=float, default=600.0, help="Alert period from recording start.")
    parser.add_argument("--drowsy-lead-sec", type=float, default=60.0, help="Seconds before button event marked drowsy.")
    parser.add_argument("--max-recordings", type=int, default=0, help="Optional limit for quick smoke tests.")
    parser.add_argument(
        "--max-windows-per-recording",
        type=int,
        default=0,
        help="Optional per-recording window cap for quick tests.",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        default=True,
        help="Per-subject class balancing via majority-class downsampling (default: on).",
    )
    parser.add_argument(
        "--no-balance-classes",
        dest="balance_classes",
        action="store_false",
        help="Disable per-subject class balancing.",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="Random seed for balancing downsampling.",
    )
    return parser.parse_args()


def balance_per_subject(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Downsample the majority class within each subject so alert/drowsy are 1:1."""
    balanced_frames: List[pd.DataFrame] = []
    summary_rows: List[str] = []
    for subject, sub_df in df.groupby("subject"):
        alert_df = sub_df[sub_df["label"] == 0]
        drowsy_df = sub_df[sub_df["label"] == 1]
        if alert_df.empty or drowsy_df.empty:
            summary_rows.append(
                f"  {subject}: dropped (alert={len(alert_df)}, drowsy={len(drowsy_df)}) - missing one class"
            )
            continue
        target = min(len(alert_df), len(drowsy_df))
        a = alert_df.sample(n=target, random_state=seed)
        d = drowsy_df.sample(n=target, random_state=seed)
        balanced_frames.append(a)
        balanced_frames.append(d)
        summary_rows.append(
            f"  {subject}: alert {len(alert_df)}->{target}, drowsy {len(drowsy_df)}->{target}"
        )

    if not balanced_frames:
        raise RuntimeError("Class balancing removed all samples. Check label distribution.")

    balanced = pd.concat(balanced_frames, ignore_index=True)
    print("[INFO] Class balance per subject after downsampling:")
    for line in summary_rows:
        print(line)
    print(f"[INFO] Balanced total samples: {len(balanced)} (from {len(df)})")
    return balanced


def _model_factories() -> Dict[str, object]:
    return {
        "RF": lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM": lambda: SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
    }


def build_feature_table(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pairs = list(iter_recording_pairs(data_dir))
    if args.max_recordings > 0:
        pairs = pairs[: args.max_recordings]
    if not pairs:
        raise RuntimeError("No valid recording/annotation EDF pairs found.")

    rows: List[Dict[str, object]] = []
    for recording_path, annotation_path in pairs:
        print(f"[INFO] Processing {recording_path.name}")
        windows = generate_labeled_windows(
            recording_path=recording_path,
            annotation_path=annotation_path,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            alert_duration_sec=args.alert_duration_sec,
            drowsy_lead_sec=args.drowsy_lead_sec,
            drop_gray_zone=True,
        )
        if args.max_windows_per_recording > 0:
            windows = windows[: args.max_windows_per_recording]

        for sample in windows:
            feature_dict = extract_hrv_features(
                ecg_window=np.asarray(sample["ecg"], dtype=np.float64),
                sampling_rate=float(sample["sampling_rate"]),
            )
            if feature_dict is None:
                continue

            row = {
                "subject": sample["subject"],
                "trial": sample["trial"],
                "recording_path": sample["recording_path"],
                "start_sec": sample["start_sec"],
                "end_sec": sample["end_sec"],
                "label": int(sample["label"]),
            }
            row.update(feature_dict)
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid HRV feature rows extracted.")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label", "subject"])
    df["label"] = df["label"].astype(int)
    return df


def _metric_summary(values: List[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def run_loso(df: pd.DataFrame, output_dir: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    subjects = sorted(df["subject"].unique().tolist())
    model_factories = _model_factories()
    report_lines: List[str] = []
    rf_all_true: List[int] = []
    rf_all_pred: List[int] = []

    report_lines.append(f"Total samples: {len(df)}")
    report_lines.append(f"Subjects: {subjects}")
    report_lines.append("")

    for model_name, factory in model_factories.items():
        fold_scores: Dict[str, List[float]] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
        }
        report_lines.append(f"[{model_name}]")
        for subject in subjects:
            train_df = df[df["subject"] != subject]
            test_df = df[df["subject"] == subject]
            if train_df.empty or test_df.empty:
                continue

            X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            y_train = train_df["label"].to_numpy(dtype=np.int64)
            X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            y_test = test_df["label"].to_numpy(dtype=np.int64)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = factory()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled).astype(np.int64)
            y_prob = (
                model.predict_proba(X_test_scaled)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred.astype(np.float64)
            )

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                roc = roc_auc_score(y_test, y_prob)
            except ValueError:
                roc = float("nan")

            fold_scores["accuracy"].append(float(acc))
            fold_scores["precision"].append(float(prec))
            fold_scores["recall"].append(float(rec))
            fold_scores["f1"].append(float(f1))
            if np.isfinite(roc):
                fold_scores["roc_auc"].append(float(roc))

            report_lines.append(
                f"- Subject {subject}: acc={acc:.4f}, prec={prec:.4f}, "
                f"rec={rec:.4f}, f1={f1:.4f}, roc_auc={roc:.4f}"
            )

            if model_name == "RF":
                rf_all_true.extend(y_test.tolist())
                rf_all_pred.extend(y_pred.tolist())

        report_lines.append("Summary:")
        for metric_name, values in fold_scores.items():
            report_lines.append(f"  {metric_name}: {_metric_summary(values)}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    (output_dir / "physio_loso_results.txt").write_text(report_text, encoding="utf-8")
    return report_text, np.asarray(rf_all_true, dtype=np.int64), np.asarray(rf_all_pred, dtype=np.int64)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    if y_true.size == 0:
        return
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Alert (0)", "Drowsy (1)"],
        yticklabels=["Alert (0)", "Drowsy (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Physio RF LOSO Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "physio_confusion.png", dpi=180)
    plt.close(fig)


def train_final_models(df: pd.DataFrame, output_dir: Path) -> None:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, output_dir / "physio_scaler.pkl")

    factories = _model_factories()
    rf = factories["RF"]()
    rf.fit(X_scaled, y)
    joblib.dump(rf, output_dir / "physio_rf.pkl")

    svm = factories["SVM"]()
    svm.fit(X_scaled, y)
    joblib.dump(svm, output_dir / "physio_svm.pkl")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_feature_table(args)
    df.to_csv(output_dir / "physio_features.csv", index=False)
    print(f"[INFO] Saved feature table: {output_dir / 'physio_features.csv'} ({len(df)} rows)")

    if args.balance_classes:
        df = balance_per_subject(df, seed=args.balance_seed)
        df.to_csv(output_dir / "physio_features_balanced.csv", index=False)

    report_text, y_true_rf, y_pred_rf = run_loso(df, output_dir)
    print(report_text)
    save_confusion_matrix(y_true_rf, y_pred_rf, output_dir)

    train_final_models(df, output_dir)
    print("[INFO] Saved models: physio_rf.pkl, physio_svm.pkl, physio_scaler.pkl")


if __name__ == "__main__":
    main()
