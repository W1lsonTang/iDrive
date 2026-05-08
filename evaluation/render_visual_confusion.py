"""Re-render the visual RF confusion matrix to match the physio confusion plot style.

Reproduces the original 80/20 stratified split (random_state=42) used in
``visual_channel.py``, applies the saved scaler + RF model, and writes a single
panel confusion matrix to ``output/visual_confusion.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["ear", "pitch", "yaw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render visual RF confusion matrix.")
    parser.add_argument("--features-csv", default="output/visual_features.csv")
    parser.add_argument("--rf-model", default="output/rf_model.pkl")
    parser.add_argument("--scaler", default="output/scaler.pkl")
    parser.add_argument("--output-png", default="output/visual_confusion.png")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.features_csv)
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int32)

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    scaler = joblib.load(args.scaler)
    rf = joblib.load(args.rf_model)

    x_test_scaled = scaler.transform(x_test)
    y_pred = rf.predict(x_test_scaled)

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

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
    ax.set_title("Visual RF Confusion Matrix (DDD 80/20 split)")
    fig.tight_layout()

    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    tn, fp, fn, tp = matrix.ravel()
    total = matrix.sum()
    print(f"[INFO] Saved figure: {out_path}")
    print(f"[INFO] Confusion matrix (n={total}):")
    print(f"        Pred Alert  Pred Drowsy")
    print(f"  Alert    {tn:>8}     {fp:>8}")
    print(f"  Drowsy   {fn:>8}     {tp:>8}")


if __name__ == "__main__":
    main()
