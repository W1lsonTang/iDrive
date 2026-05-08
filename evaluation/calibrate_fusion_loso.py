"""LOSO cross-validated grid search for late-fusion parameters.

For each candidate (w_v, threshold), compute F1-macro across all leave-one-subject-out
folds and take the mean. The (w_v, threshold) with highest mean F1-macro wins.
This is more robust than a single random split when subject count is moderate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO grid search for fusion params.")
    parser.add_argument("--fused-table", default="output/fused_table.csv")
    parser.add_argument("--report", default="output/fusion_calibration_loso.txt")
    parser.add_argument("--heatmap", default="output/fusion_grid_heatmap_loso.png")
    parser.add_argument("--best-params-json", default="output/fusion_best_params_loso.json")
    parser.add_argument("--weights", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    return parser.parse_args()


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def loso_score(
    df: pd.DataFrame,
    w_v: float,
    threshold: float,
) -> tuple[float, list[float]]:
    """Return mean F1-macro across LOSO folds and per-fold scores."""
    subjects = sorted(df["subject"].unique().tolist())
    fold_scores: list[float] = []
    for test_sub in subjects:
        test_df = df[df["subject"] == test_sub]
        if test_df.empty:
            continue
        y = test_df["y_binary"].to_numpy(dtype=np.int64)
        if len(np.unique(y)) < 2:
            # Skip folds where the test subject has only one class (degenerate F1).
            continue
        fused = w_v * test_df["p_v"].to_numpy(dtype=np.float64) + (1.0 - w_v) * test_df["p_p"].to_numpy(dtype=np.float64)
        pred = (fused >= threshold).astype(int)
        score = f1_score(y, pred, average="macro", zero_division=0)
        fold_scores.append(float(score))
    if not fold_scores:
        return 0.0, []
    return float(np.mean(fold_scores)), fold_scores


def loso_pooled_metrics(
    df: pd.DataFrame,
    w_v: float,
    threshold: float,
) -> dict[str, float]:
    """Pool all LOSO predictions and compute aggregate metrics."""
    fused = w_v * df["p_v"].to_numpy(dtype=np.float64) + (1.0 - w_v) * df["p_p"].to_numpy(dtype=np.float64)
    return metrics_from_probs(df["y_binary"].to_numpy(dtype=np.int64), fused, threshold)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.fused_table)
    subjects = sorted(df["subject"].unique().tolist())
    print(f"[INFO] {len(df)} samples across {len(subjects)} subjects: {subjects}")

    weights = np.array([float(x) for x in args.weights.split(",")], dtype=np.float64)
    thresholds = np.array([float(x) for x in args.thresholds.split(",")], dtype=np.float64)

    grid = np.full((len(weights), len(thresholds)), np.nan, dtype=np.float64)
    best_score = -np.inf
    best_w = float(weights[0])
    best_t = float(thresholds[0])

    for i, w in enumerate(weights):
        for j, t in enumerate(thresholds):
            mean_score, _ = loso_score(df, float(w), float(t))
            grid[i, j] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_w = float(w)
                best_t = float(t)

    print(f"[INFO] Best LOSO mean F1-macro: {best_score:.4f} at w_v={best_w:.2f}, threshold={best_t:.2f}")

    baseline_specs = [
        ("visual_only (w_v=1.0, t=0.50)", 1.0, 0.5),
        ("physio_only (w_v=0.0, t=0.50)", 0.0, 0.5),
        ("fixed_0.6_0.4 (t=0.50)", 0.6, 0.5),
        ("fixed_0.7_0.3 (t=0.35) [previous]", 0.7, 0.35),
        (f"best_loso (w_v={best_w:.2f}, t={best_t:.2f})", best_w, best_t),
    ]

    report_lines: list[str] = []
    report_lines.append(f"Total samples: {len(df)}")
    report_lines.append(f"Subjects ({len(subjects)}): {subjects}")
    report_lines.append("")
    report_lines.append("Evaluation: LOSO cross-validation (mean across folds + pooled).")
    report_lines.append("")
    report_lines.append(f"Best params (max LOSO mean F1_macro): w_v={best_w:.2f}, w_p={1.0 - best_w:.2f}, threshold={best_t:.2f}")
    report_lines.append(f"Best LOSO mean F1_macro: {best_score:.4f}")
    report_lines.append("")
    report_lines.append("[LOSO Mean F1_macro per Strategy]")
    for name, w, t in baseline_specs:
        mean_f1, fold_scores = loso_score(df, w, t)
        std_f1 = float(np.std(fold_scores)) if fold_scores else 0.0
        report_lines.append(
            f"{name}: mean_f1_macro={mean_f1:.4f}, std={std_f1:.4f}, n_folds={len(fold_scores)}"
        )

    report_lines.append("")
    report_lines.append("[Pooled Metrics across all LOSO predictions]")
    for name, w, t in baseline_specs:
        m = loso_pooled_metrics(df, w, t)
        report_lines.append(
            f"{name}: acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, "
            f"rec={m['recall']:.4f}, f1={m['f1']:.4f}, "
            f"f1_macro={m['f1_macro']:.4f}, roc_auc={m['roc_auc']:.4f}"
        )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report_lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("LOSO grid search (mean F1 Macro)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Visual weight (w_v)")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_yticks(np.arange(len(weights)))
    ax.set_yticklabels([f"{w:.2f}" for w in weights])
    for i in range(len(weights)):
        for j in range(len(thresholds)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="Mean F1 Macro (LOSO)")
    fig.tight_layout()
    Path(args.heatmap).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.heatmap, dpi=150)
    plt.close(fig)

    best_params = {
        "visual_weight": float(best_w),
        "physio_weight": float(1.0 - best_w),
        "fusion_threshold": float(best_t),
        "loso_mean_f1_macro": float(best_score),
        "subjects": subjects,
        "n_samples": int(len(df)),
        "method": "LOSO grid search",
    }
    Path(args.best_params_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.best_params_json).write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print(f"[INFO] Saved report:      {args.report}")
    print(f"[INFO] Saved heatmap:     {args.heatmap}")
    print(f"[INFO] Saved best params: {args.best_params_json}")


if __name__ == "__main__":
    main()
