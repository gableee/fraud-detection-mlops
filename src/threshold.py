"""Choose a decision threshold for the tuned model, then evaluate on the test set once.

The threshold is chosen ONLY from out-of-fold (OOF) predictions on the training set,
so the test set stays untouched until the final evaluation.

Rule: catching fraud matters more than avoiding false alarms (an analyst can clear a
false alarm; a missed fraud is lost money). So we pick the highest threshold whose
OOF recall is at least TARGET_RECALL - i.e. the fewest false alarms that still
reach the recall target.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from train import load_processed_data

# 0.85 is the knee of the OOF curve: recall climbs from 0.82 (t=0.5) to 0.85 for a few
# hundred extra false alarms, but going from 0.85 to 0.90 costs thousands more.
TARGET_RECALL = 0.85
FIG_DIR = "reports/figures"

# Colors: neutral gray for the baseline, blue / orange for the two tuned settings.
C_BASE, C_TUNED, C_THRESH = "#8a8a85", "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e6e5e0"


def choose_threshold(model, X_train, y_train):
    """Return the highest threshold with OOF recall >= TARGET_RECALL."""
    params = model.get_params()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(xgb.XGBClassifier(**params), X_train, y_train,
                            cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof)
    # recall[i] / precision[i] correspond to thresholds[i] (last point has no threshold)
    ok = np.where(recall[:-1] >= TARGET_RECALL)[0]
    i = ok[-1]  # highest threshold that still meets the target
    return float(thresholds[i]), float(recall[i]), float(precision[i])


def counts(y_true, proba, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true, (proba >= threshold).astype(int)).ravel()
    return {"caught": int(tp), "missed": int(fn), "false_alarms": int(fp),
            "precision": tp / (tp + fp), "recall": tp / (tp + fn)}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr(y_test, p_base, p_tuned, threshold, rows):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for proba, name, color in ((p_base, "Baseline", C_BASE), (p_tuned, "Tuned", C_TUNED)):
        pr, rc, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rc, pr, color=color, linewidth=2, label=f"{name} (AUPRC = {ap:.3f})")
    points = [(rows[0], C_BASE, "Baseline @ 0.50"),
              (rows[1], C_TUNED, "Tuned @ 0.50"),
              (rows[2], C_THRESH, f"Tuned @ {threshold:.3f} (chosen)")]
    for r, color, label in points:
        ax.scatter(r["recall"], r["precision"], s=70, color=color,
                   edgecolor="white", linewidth=2, zorder=5, label=label)
    style(ax)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of frauds caught)", color=INK)
    ax.set_ylabel("Precision (share of alarms that are real fraud)", color=INK)
    ax.set_title("Precision-recall on the test set, with each operating point",
                 color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.savefig(f"{FIG_DIR}/pr_curve_operating_points.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(rows, labels, colors):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    total = rows[0]["caught"] + rows[0]["missed"]
    for ax, key, title in ((axes[0], "caught", f"Frauds caught (of {total})"),
                           (axes[1], "false_alarms", "False alarms (legit flagged)")):
        vals = [r[key] for r in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, color=colors, height=0.6, edgecolor="white", linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v}", va="center", color=INK, fontsize=10)
        ax.set_yticks(y, labels, color=INK)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.18)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/tradeoff_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_test, probas, thresholds, labels):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, proba, t, label in zip(axes, probas, thresholds, labels):
        cm = confusion_matrix(y_test, (proba >= t).astype(int))
        ax.imshow(cm, cmap="Blues", norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=cm.max()))
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                    color="white" if v > 1000 else INK)
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual", color=MUTED)
        ax.set_title(label, color=INK, fontsize=11)
        ax.tick_params(colors=INK, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed_data("data/processed")
    baseline = joblib.load("models/baseline_xgb_model.joblib")
    tuned = joblib.load("models/tuned_xgb_model.joblib")

    print(f"Choosing threshold from 5-fold OOF predictions (target recall {TARGET_RECALL:.0%})...")
    threshold, oof_recall, oof_precision = choose_threshold(tuned, X_train, y_train)
    print(f"Chosen threshold: {threshold:.4f} (OOF recall {oof_recall:.3f}, precision {oof_precision:.3f})")

    p_base = baseline.predict_proba(X_test)[:, 1]
    p_tuned = tuned.predict_proba(X_test)[:, 1]
    labels = ["Baseline @ 0.50", "Tuned @ 0.50", f"Tuned @ {threshold:.3f}"]
    rows = [counts(y_test, p_base, 0.5), counts(y_test, p_tuned, 0.5),
            counts(y_test, p_tuned, threshold)]

    print("\nTest set results:")
    print(f"{'Model':<20}{'Caught':>8}{'Missed':>8}{'False alarms':>14}{'Precision':>11}{'Recall':>8}")
    for label, r in zip(labels, rows):
        print(f"{label:<20}{r['caught']:>8}{r['missed']:>8}{r['false_alarms']:>14}"
              f"{r['precision']:>11.3f}{r['recall']:>8.3f}")
    print(f"AUPRC baseline {average_precision_score(y_test, p_base):.3f}, "
          f"tuned {average_precision_score(y_test, p_tuned):.3f}")

    plot_pr(y_test, p_base, p_tuned, threshold, rows)
    plot_tradeoff(rows, labels, [C_BASE, C_TUNED, C_THRESH])
    plot_confusion(y_test, [p_base, p_tuned, p_tuned], [0.5, 0.5, threshold], labels)

    with open("models/threshold.json", "w") as f:
        json.dump({"model": "tuned_xgb_model.joblib", "threshold": threshold,
                   "target_recall": TARGET_RECALL, "oof_recall": oof_recall,
                   "oof_precision": oof_precision}, f, indent=2)
    print(f"\nFigures saved to {FIG_DIR}/, threshold saved to models/threshold.json")
