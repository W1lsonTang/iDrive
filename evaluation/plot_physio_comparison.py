"""Generate a Physio Channel comparison figure (Alert vs Drowsy window).

Each panel shows:
  - ECG signal + R-peak markers
  - 7-dimensional HRV feature bar chart
  - RF and SVM P_p confidence values with verdict

Usage:
    python evaluation/plot_physio_comparison.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neurokit2 as nk
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from physio_channel.edf_loader import load_ecg_signal, load_button_onsets
from physio_channel.features import FEATURE_COLUMNS, extract_hrv_features, feature_vector_from_dict

FEATURE_LABELS = {
    "mean_rr_ms":   "mean RR\n(ms)",
    "sdnn_ms":      "SDNN\n(ms)",
    "rmssd_ms":     "RMSSD\n(ms)",
    "pnn50":        "pNN50\n(%)",
    "lf_power":     "LF\npower",
    "hf_power":     "HF\npower",
    "lf_hf_ratio":  "LF/HF\nratio",
}

# performance table (from physio_loso_results.txt)
PERF_TABLE = [
    ["SVM (RBF)",     "0.592", "0.552", "0.666", ""],
    ["Random Forest", "0.592", "0.559", "0.639", "✓"],
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dd-dir",       default="DD-Database")
    p.add_argument("--subject",      default="01M",  help="Subject ID")
    p.add_argument("--trial",        default="1",    help="Trial number")
    p.add_argument("--alert-start",  type=float, default=60.0,
                   help="Start time (s) of the alert window")
    p.add_argument("--drowsy-btn",   type=float, default=5092.0,
                   help="Button press timestamp (s) for drowsy window end")
    p.add_argument("--rf-model",     default="output/physio_rf.pkl")
    p.add_argument("--svm-model",    default="output/physio_svm.pkl")
    p.add_argument("--scaler",       default="output/physio_scaler.pkl")
    p.add_argument("--output",       default="output/physio_comparison.png")
    p.add_argument("--fs",           type=float, default=128.0)
    p.add_argument("--window-sec",   type=float, default=60.0)
    return p.parse_args()


def predict_pp(model, scaler, feat_vec: np.ndarray) -> float:
    xs = scaler.transform(feat_vec.reshape(1, -1))
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(xs)[0, 1])
    s = float(model.decision_function(xs)[0])
    return float(1 / (1 + np.exp(-s)))


def get_rpeaks(ecg: np.ndarray, fs: float):
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    return np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64), cleaned


def draw_ecg_panel(ax, ecg, rpeaks, fs, title, label_color, label_str, time_offset=0.0):
    t = np.arange(len(ecg)) / fs + time_offset
    ax.plot(t, ecg * 1e6, color="#1565c0", lw=0.8, label="ECG signal")
    if rpeaks.size:
        ax.scatter(rpeaks / fs + time_offset, ecg[rpeaks] * 1e6,
                   color="#FF5252", s=40, zorder=5, label="R-peak")
        rr_ms = np.diff(rpeaks) / fs * 1000
        mean_rr = float(np.mean(rr_ms)) if rr_ms.size else float("nan")
        ax.text(0.98, 0.06, f"mean RR = {mean_rr:.0f} ms",
                transform=ax.transAxes, ha="right", va="bottom",
                color="#666666", fontsize=9)
    ax.set_facecolor("white")
    ax.set_title(title, color=label_color, fontsize=11, fontweight="bold", pad=4)
    ax.set_xlabel("Time (s)", color="#444444", fontsize=8)
    ax.set_ylabel("Amplitude (µV)", color="#444444", fontsize=8)
    ax.tick_params(colors="#444444", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    # label badge
    ax.text(0.02, 0.93, f"  {label_str}  ",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            fontweight="bold", color="white",
            bbox=dict(facecolor=label_color, edgecolor="none", pad=2))
    ax.legend(loc="upper right", fontsize=7,
              facecolor="white", edgecolor="#cccccc", labelcolor="#333333")


def draw_hrv_panel(ax, feat_dict, title_color):
    vals = [feat_dict[k] for k in FEATURE_COLUMNS]
    labels = [FEATURE_LABELS[k] for k in FEATURE_COLUMNS]
    x = np.arange(len(vals))

    bars = ax.bar(x, vals, color=title_color, alpha=0.80, edgecolor="#aaaaaa", width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.2f}", ha="center", va="bottom", color="#333333", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, color="#444444")
    ax.set_ylabel("Feature value", color="#444444", fontsize=8)
    ax.set_title("HRV Feature Vector (7-dim)", color="#444444", fontsize=9, pad=3)
    ax.set_facecolor("white")
    ax.tick_params(axis="y", colors="#444444", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


def draw_conf_panel(ax, pp_rf, pp_svm, label_true):
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def conf_block(pv, model_name, y_top):
        verdict  = "DROWSY" if pv >= 0.5 else "ALERT"
        col      = "#e53935" if pv >= 0.5 else "#2e7d32"
        correct  = (verdict == "DROWSY") == (label_true == 1)
        check    = "✓" if correct else "✗"

        ax.text(0.05, y_top, f"[{model_name}]", color="#555555",
                fontsize=9, fontweight="bold", va="top")
        ax.text(0.05, y_top - 0.12,
                f"P_p = {pv:.3f}  →  {verdict}  {check}",
                color=col, fontsize=10, fontweight="bold", va="top")

        # progress bar
        bx, bw, bh = 0.05, 0.90, 0.07
        by = y_top - 0.30
        rect_bg  = plt.Rectangle((bx, by), bw, bh,
                                  facecolor="#eeeeee", edgecolor="#cccccc", lw=0.8)
        rect_fill = plt.Rectangle((bx, by), bw * float(np.clip(pv, 0, 1)), bh,
                                   facecolor=col, alpha=0.80)
        ax.add_patch(rect_bg)
        ax.add_patch(rect_fill)

    conf_block(pp_rf,  "RF ", 0.95)
    conf_block(pp_svm, "SVM", 0.50)

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


def make_figure(alert_ecg, drowsy_ecg, alert_feat, drowsy_feat,
                pp_rf_alert, pp_svm_alert, pp_rf_drowsy, pp_svm_drowsy,
                fs, alert_start, drowsy_start, output_path):

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")

    # outer columns: alert (left) | drowsy (right)
    outer = gridspec.GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.10,
                              left=0.06, right=0.97, top=0.88, bottom=0.13)

    panel_colors = {"alert": "#69F0AE", "drowsy": "#FF5252"}
    panel_labels = {"alert": "ALERT",   "drowsy": "DROWSY"}
    panel_ecg    = {"alert": alert_ecg, "drowsy": drowsy_ecg}
    panel_feat   = {"alert": alert_feat,"drowsy": drowsy_feat}
    panel_pp_rf  = {"alert": pp_rf_alert,  "drowsy": pp_rf_drowsy}
    panel_pp_svm = {"alert": pp_svm_alert, "drowsy": pp_svm_drowsy}
    panel_start  = {"alert": alert_start,  "drowsy": drowsy_start}
    panel_true   = {"alert": 0, "drowsy": 1}
    panel_title  = {
        "alert":  f"Alert Window  (start={alert_start:.0f}s,  label=0)",
        "drowsy": f"Drowsy Window  (start={drowsy_start:.0f}s,  label=1)",
    }

    for col_idx, key in enumerate(["alert", "drowsy"]):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[col_idx],
            height_ratios=[2.5, 1.8, 1.2], hspace=0.55)

        ax_ecg  = fig.add_subplot(inner[0])
        ax_hrv  = fig.add_subplot(inner[1])
        ax_conf = fig.add_subplot(inner[2])

        ecg    = panel_ecg[key]
        rpeaks, _ = get_rpeaks(ecg, fs)

        draw_ecg_panel(ax_ecg, ecg, rpeaks, fs,
                       panel_title[key], panel_colors[key],
                       panel_labels[key], panel_start[key])
        draw_hrv_panel(ax_hrv, panel_feat[key], panel_colors[key])
        draw_conf_panel(ax_conf, panel_pp_rf[key], panel_pp_svm[key], panel_true[key])

    fig.suptitle(
        "DD-Database — Physiological Channel Feature Extraction & Model Comparison\n"
        "ECG  →  R-peak Detection  →  HRV (7-dim)  →  RF vs SVM  →  P_p",
        color="#111111", fontsize=13, fontweight="bold", y=0.97
    )

    # performance table
    tbl_ax = fig.add_axes([0.12, 0.01, 0.76, 0.09])
    tbl_ax.set_facecolor("white")
    tbl_ax.axis("off")
    col_labels = ["Model", "Accuracy (LOSO)", "F1-macro (LOSO)", "ROC-AUC (LOSO)", "Selected"]
    tbl = tbl_ax.table(cellText=PERF_TABLE, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#f5f5f5" if r % 2 == 0 else "white")
        cell.set_text_props(color="#222222")
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#e3f2fd")
            cell.set_text_props(color="#1565c0", fontweight="bold")
        if c == 4 and r == 2:
            cell.set_text_props(color="#2e7d32", fontweight="bold", fontsize=12)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Saved: {output_path}")


def main():
    args = parse_args()

    dd_dir = Path(args.dd_dir)
    rec_path = dd_dir / f"{args.subject}_{args.trial}.edf"
    ann_path = dd_dir / f"{args.subject}_{args.trial}_annotations.edf"

    print(f"[INFO] Loading {rec_path.name} ...")
    ecg, fs, _ = load_ecg_signal(rec_path)
    onsets      = load_button_onsets(ann_path)
    print(f"[INFO] Signal length: {len(ecg)/fs:.1f}s  Button presses: {onsets}")

    win = int(args.window_sec * fs)

    # alert window
    alert_start     = args.alert_start
    alert_start_idx = int(round(alert_start * fs))
    alert_ecg       = ecg[alert_start_idx: alert_start_idx + win]

    # drowsy window: 60s ending at specified button press
    btn            = args.drowsy_btn
    drow_end       = int(round(btn * fs))
    drow_start_idx = max(0, drow_end - win)
    drowsy_ecg     = ecg[drow_start_idx:drow_end]
    drowsy_start   = drow_start_idx / fs
    print(f"[INFO] Alert window : {alert_start:.0f} – {alert_start + args.window_sec:.0f}s")
    print(f"[INFO] Drowsy window: {drowsy_start:.1f} – {btn:.1f}s  (button press at {btn:.1f}s)")

    # HRV features
    alert_feat_dict  = extract_hrv_features(alert_ecg,  fs)
    drowsy_feat_dict = extract_hrv_features(drowsy_ecg, fs)
    if alert_feat_dict is None or drowsy_feat_dict is None:
        raise SystemExit("[ERROR] HRV extraction failed for one of the windows.")

    alert_vec  = feature_vector_from_dict(alert_feat_dict)
    drowsy_vec = feature_vector_from_dict(drowsy_feat_dict)
    print(f"[INFO] Alert  HRV: { {k: round(v,3) for k,v in alert_feat_dict.items() if k in FEATURE_COLUMNS} }")
    print(f"[INFO] Drowsy HRV: { {k: round(v,3) for k,v in drowsy_feat_dict.items() if k in FEATURE_COLUMNS} }")

    # load models
    scaler    = joblib.load(args.scaler)
    rf_model  = joblib.load(args.rf_model)
    svm_model = joblib.load(args.svm_model)

    pp_rf_alert  = predict_pp(rf_model,  scaler, alert_vec)
    pp_svm_alert = predict_pp(svm_model, scaler, alert_vec)
    pp_rf_drowsy  = predict_pp(rf_model,  scaler, drowsy_vec)
    pp_svm_drowsy = predict_pp(svm_model, scaler, drowsy_vec)

    print(f"[INFO] Alert   → RF P_p={pp_rf_alert:.3f}  SVM P_p={pp_svm_alert:.3f}")
    print(f"[INFO] Drowsy  → RF P_p={pp_rf_drowsy:.3f}  SVM P_p={pp_svm_drowsy:.3f}")

    make_figure(
        alert_ecg, drowsy_ecg,
        alert_feat_dict, drowsy_feat_dict,
        pp_rf_alert, pp_svm_alert,
        pp_rf_drowsy, pp_svm_drowsy,
        fs, alert_start, drowsy_start,
        args.output,
    )


if __name__ == "__main__":
    main()
