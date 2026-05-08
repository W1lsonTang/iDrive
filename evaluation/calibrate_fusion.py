from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

FEATURE_COLUMNS = [
    "mean_rr_ms",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
]

VISUAL_COLUMNS = ["ear", "pitch", "yaw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate late-fusion parameters using UL-DD.")
    parser.add_argument("--uldd-dir", default="UL-DD", help="Path to UL-DD root directory.")
    parser.add_argument("--visual-series", default="output/uldd_visual_series.csv")
    parser.add_argument("--physio-series", default="output/uldd_physio_series.csv")
    parser.add_argument("--visual-model", default="output/rf_model.pkl")
    parser.add_argument("--visual-scaler", default="output/scaler.pkl")
    parser.add_argument("--physio-model", default="output/physio_rf.pkl")
    parser.add_argument("--physio-scaler", default="output/physio_scaler.pkl")
    parser.add_argument("--fused-table", default="output/fused_table.csv")
    parser.add_argument("--report", default="output/fusion_calibration.txt")
    parser.add_argument("--heatmap", default="output/fusion_grid_heatmap.png")
    parser.add_argument("--best-params-json", default="output/fusion_best_params.json")
    parser.add_argument("--kss-threshold", type=float, default=6.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--min-visual-points", type=int, default=20)
    return parser.parse_args()


def load_labels_map(uldd_dir: Path) -> dict[tuple[str, str], list[float]]:
    base = uldd_dir / "CSV_Files" / "CSV_Files"
    out: dict[tuple[str, str], list[float]] = {}
    for subject_dir in sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name):
        subject = subject_dir.name
        for session_dir in sorted([d for d in subject_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
            session = session_dir.name
            label_file = session_dir / f"{subject}_Labels_{session}.csv"
            if not label_file.is_file():
                continue
            values = pd.read_csv(label_file, header=None).iloc[0].to_numpy(dtype=np.float64).tolist()
            out[(subject, session)] = values
    if not out:
        raise RuntimeError("No UL-DD label files found.")
    return out


def model_predict_proba(model, scaler, x_row: np.ndarray) -> float:
    x_row = x_row.reshape(1, -1)
    x_scaled = scaler.transform(x_row)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x_scaled)[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(x_scaled)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    return float(int(model.predict(x_scaled)[0]))


def build_fused_table(
    visual_df: pd.DataFrame,
    physio_df: pd.DataFrame,
    labels_map: dict[tuple[str, str], list[float]],
    visual_model,
    visual_scaler,
    physio_model,
    physio_scaler,
    min_visual_points: int,
    kss_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    visual_grouped = {
        key: group.sort_values("t_sec").reset_index(drop=True)
        for key, group in visual_df.groupby(["subject", "session"], sort=False)
    }

    for _, p in physio_df.iterrows():
        key = (str(p["subject"]), str(p["session"]))
        if key not in labels_map:
            continue
        v_df = visual_grouped.get(key)
        if v_df is None:
            continue

        start = float(p["window_start_sec"])
        end = float(p["window_end_sec"])
        center = float(p["t_sec"])

        vw = v_df[(v_df["t_sec"] >= start) & (v_df["t_sec"] < end)]
        if vw.empty:
            continue

        vf = vw[VISUAL_COLUMNS].replace([np.inf, -np.inf], np.nan)
        valid_mask = np.isfinite(vf.to_numpy(dtype=np.float64)).all(axis=1)
        if int(valid_mask.sum()) < min_visual_points:
            continue

        vf_mean = vf[valid_mask].mean(axis=0).to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(vf_mean)):
            continue

        pf = p[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(pf)):
            continue

        minute_idx = int(max(0, center) // 60)
        labels = labels_map[key]
        if minute_idx >= len(labels):
            minute_idx = len(labels) - 1
        kss = float(labels[minute_idx])
        y_binary = int(kss >= kss_threshold)

        p_v = model_predict_proba(visual_model, visual_scaler, vf_mean)
        p_p = model_predict_proba(physio_model, physio_scaler, pf)

        rows.append(
            {
                "subject": key[0],
                "session": key[1],
                "window_start_sec": start,
                "window_end_sec": end,
                "t_sec": center,
                "ear": float(vf_mean[0]),
                "pitch": float(vf_mean[1]),
                "yaw": float(vf_mean[2]),
                "p_v": float(p_v),
                "p_p": float(p_p),
                "kss": kss,
                "y_binary": y_binary,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Fused table is empty. Check visual extraction and alignment rules.")
    return df.sort_values(["subject", "session", "t_sec"]).reset_index(drop=True)


def split_subjects(subjects: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    n = len(subjects)
    n_test = max(1, int(round(n * (1.0 - train_ratio))))
    test_sub = sorted(rng.choice(subjects, size=n_test, replace=False).tolist())
    train_sub = sorted([s for s in subjects if s not in set(test_sub)])
    return train_sub, test_sub


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


def run_grid_search(
    y_true: np.ndarray,
    p_v: np.ndarray,
    p_p: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    scores = np.full((len(weights), len(thresholds)), np.nan, dtype=np.float64)
    best_score = -np.inf
    best_w = 0.6
    best_t = 0.5

    for i, w in enumerate(weights):
        for j, t in enumerate(thresholds):
            fused = w * p_v + (1.0 - w) * p_p
            pred = (fused >= t).astype(int)
            score = f1_score(y_true, pred, average="macro", zero_division=0)
            scores[i, j] = score
            if score > best_score:
                best_score = float(score)
                best_w = float(w)
                best_t = float(t)

    return best_w, best_t, scores


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name}: acc={metrics['accuracy']:.4f}, prec={metrics['precision']:.4f}, "
        f"rec={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
    )


def save_heatmap(path: Path, scores: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(scores, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("Fusion grid search (Train F1 Macro)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Visual weight (w_v)")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_yticks(np.arange(len(weights)))
    ax.set_yticklabels([f"{w:.2f}" for w in weights])
    fig.colorbar(im, ax=ax, label="F1 Macro")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    visual_df = pd.read_csv(args.visual_series)
    physio_df = pd.read_csv(args.physio_series)

    labels_map = load_labels_map(Path(args.uldd_dir))

    visual_model = joblib.load(args.visual_model)
    visual_scaler = joblib.load(args.visual_scaler)
    physio_model = joblib.load(args.physio_model)
    physio_scaler = joblib.load(args.physio_scaler)

    fused_df = build_fused_table(
        visual_df=visual_df,
        physio_df=physio_df,
        labels_map=labels_map,
        visual_model=visual_model,
        visual_scaler=visual_scaler,
        physio_model=physio_model,
        physio_scaler=physio_scaler,
        min_visual_points=args.min_visual_points,
        kss_threshold=args.kss_threshold,
    )

    fused_path = Path(args.fused_table)
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(fused_path, index=False)

    subjects = sorted(fused_df["subject"].unique().tolist())
    if len(subjects) < 2:
        raise RuntimeError("Need at least 2 subjects for train/test split.")
    train_sub, test_sub = split_subjects(subjects, train_ratio=args.train_ratio, seed=args.seed)

    train_df = fused_df[fused_df["subject"].isin(train_sub)].reset_index(drop=True)
    test_df = fused_df[fused_df["subject"].isin(test_sub)].reset_index(drop=True)

    weights = np.array([float(x) for x in args.weights.split(",")], dtype=np.float64)
    thresholds = np.array([float(x) for x in args.thresholds.split(",")], dtype=np.float64)

    best_w, best_t, score_grid = run_grid_search(
        y_true=train_df["y_binary"].to_numpy(dtype=np.int64),
        p_v=train_df["p_v"].to_numpy(dtype=np.float64),
        p_p=train_df["p_p"].to_numpy(dtype=np.float64),
        weights=weights,
        thresholds=thresholds,
    )

    def fused_prob_fn(w: float) -> Callable[[pd.DataFrame], np.ndarray]:
        return lambda d: w * d["p_v"].to_numpy(dtype=np.float64) + (1.0 - w) * d["p_p"].to_numpy(dtype=np.float64)

    baseline_defs = [
        ("visual_only", lambda d: d["p_v"].to_numpy(dtype=np.float64), 0.5),
        ("physio_only", lambda d: d["p_p"].to_numpy(dtype=np.float64), 0.5),
        ("fixed_0.6_0.4", fused_prob_fn(0.6), 0.5),
        ("best_fusion", fused_prob_fn(best_w), best_t),
    ]

    report_lines: list[str] = []
    report_lines.append(f"Total fused samples: {len(fused_df)}")
    report_lines.append(f"Subjects ({len(subjects)}): {subjects}")
    report_lines.append(f"Train subjects ({len(train_sub)}): {train_sub}")
    report_lines.append(f"Test subjects ({len(test_sub)}): {test_sub}")
    report_lines.append("")
    report_lines.append(f"Best params on train: w_v={best_w:.2f}, w_p={1.0 - best_w:.2f}, threshold={best_t:.2f}")
    report_lines.append("")

    report_lines.append("[Train Metrics]")
    for name, prob_fn, thr in baseline_defs:
        probs = prob_fn(train_df)
        m = metrics_from_probs(train_df["y_binary"].to_numpy(dtype=np.int64), probs, threshold=thr)
        report_lines.append(format_metrics(name, m))

    report_lines.append("")
    report_lines.append("[Test Metrics]")
    for name, prob_fn, thr in baseline_defs:
        probs = prob_fn(test_df)
        m = metrics_from_probs(test_df["y_binary"].to_numpy(dtype=np.int64), probs, threshold=thr)
        report_lines.append(format_metrics(name, m))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    heatmap_path = Path(args.heatmap)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    save_heatmap(heatmap_path, score_grid, weights=weights, thresholds=thresholds)

    best_params = {
        "visual_weight": float(best_w),
        "physio_weight": float(1.0 - best_w),
        "fusion_threshold": float(best_t),
        "seed": int(args.seed),
        "kss_threshold": float(args.kss_threshold),
        "train_subjects": train_sub,
        "test_subjects": test_sub,
    }
    best_params_path = Path(args.best_params_json)
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print(f"[INFO] Saved fused table: {fused_path} ({len(fused_df)} rows)")
    print(f"[INFO] Best params: w_v={best_w:.2f}, w_p={1.0 - best_w:.2f}, threshold={best_t:.2f}")
    print(f"[INFO] Saved report: {report_path}")
    print(f"[INFO] Saved heatmap: {heatmap_path}")
    print(f"[INFO] Saved best params json: {best_params_path}")


if __name__ == "__main__":
    main()
