"""

comparison.py

Compares three models on the held-out test set:
 
  1. Baseline Decision Tree

       Features: asymmetry_score, border_irregularity, colour_complexity

       Data:     features_train.csv / features_validation.csv / features_testing.csv
 
  2. Extended Baseline Decision Tree   (random_forest.py feature set)

       Features: the 11 colour-expanded columns present in features_train.csv

                 (frac_*, n_distinct_colors, color_entropy, off_palette_dist)

                 NOTE: diameter_px / hair_coverage were NOT saved to these CSVs,

                 so the extended-DT also uses only the 11 available columns.
 
  3. Extended Baseline Random Forest

       Loaded from:  extended_baseline_random_forest.pkl

       Scaled with:  extended_baseline_scaler.pkl

       Features:     whatever columns the saved scaler was fitted on

                     (13 columns including diameter_px + hair_coverage)

       Data:         features_extended_train / _validation / _testing .csv
 
Usage

─────

"""
 
import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import joblib

import os
 
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (

    accuracy_score,

    roc_auc_score,

    classification_report,

    confusion_matrix,

    ConfusionMatrixDisplay,

    roc_curve,

)
 
# ── 0. File paths ─────────────────────────────────────────────────────────────

BASELINE_TRAIN   = "features_train.csv"

BASELINE_VAL     = "features_validation.csv"

BASELINE_TEST    = "features_testing.csv"
 
EXT_TRAIN        = "features_extended_train.csv"

EXT_VAL          = "features_extended_validation.csv"

EXT_TEST         = "features_extended_testing.csv"
 
EXT_MODEL_PATH   = "extended_baseline_random_forest.pkl"

EXT_SCALER_PATH  = "extended_baseline_scaler.pkl"
 
TARGET = "is_cancer"
 
# ── 1. Baseline feature set (only what was saved to features_*.csv) ───────────

BASELINE_FEATURES = [

    "asymmetry_score",

    "border_irregularity",

    "colour_complexity", 
    
    "frac_white",

    "frac_red",

    "frac_light_brown",

    "frac_dark_brown",

    "frac_blue_gray",

    "frac_black",

    "n_distinct_colors",

    "color_entropy",

    "off_palette_dist",

]
 
# ── 2. Extended feature set (13 cols; used by the saved RF + scaler) ──────────

EXTENDED_FEATURES = [

    "asymmetry_score",

    "border_irregularity",

    "frac_white",

    "frac_red",

    "frac_light_brown",

    "frac_dark_brown",

    "frac_blue_gray",

    "frac_black",

    "n_distinct_colors",

    "color_entropy",

    "off_palette_dist",

    "diameter_px",

    "hair_coverage",

]
 
# ── 3. Load CSVs ──────────────────────────────────────────────────────────────

print("Loading feature CSVs …")
 
b_train = pd.read_csv(BASELINE_TRAIN)

b_val   = pd.read_csv(BASELINE_VAL)

b_test  = pd.read_csv(BASELINE_TEST)
 
# Only keep features actually present in the file

b_feats = [c for c in BASELINE_FEATURES if c in b_train.columns]

missing_b = set(BASELINE_FEATURES) - set(b_feats)

if missing_b:

    print(f"  [Baseline] Missing columns (excluded): {missing_b}")

print(f"  [Baseline] Features in use ({len(b_feats)}): {b_feats}")
 
# Extended CSVs (may not exist if the pipeline hasn't been run yet)

ext_csvs_exist = all(

    os.path.exists(p) for p in [EXT_TRAIN, EXT_VAL, EXT_TEST]

)

ext_model_exists = (

    os.path.exists(EXT_MODEL_PATH) and os.path.exists(EXT_SCALER_PATH)

)
 
if not ext_csvs_exist:

    print(

        "\n  WARNING: Extended-baseline CSVs not found "

        f"({EXT_TRAIN} / {EXT_VAL} / {EXT_TEST}).\n"

        "  The Extended Baseline Random Forest will be skipped.\n"

    )
 
if not ext_model_exists:

    print(

        f"\n  WARNING: {EXT_MODEL_PATH} or {EXT_SCALER_PATH} not found.\n"

        "  The Extended Baseline Random Forest will be skipped.\n"

    )
 
# ── 4. Build arrays for baseline models ───────────────────────────────────────

X_b_train  = b_train[b_feats].values

y_b_train  = b_train[TARGET].values

X_b_val    = b_val[b_feats].values

y_b_val    = b_val[TARGET].values

X_b_test   = b_test[b_feats].values

y_b_test   = b_test[TARGET].values
 
# Train on train+val (consistent with extended baseline pipeline)

X_b_trainval = np.vstack([X_b_train, X_b_val])

y_b_trainval = np.concatenate([y_b_train, y_b_val])
 
# ── 5. Scale baseline data ────────────────────────────────────────────────────

b_scaler      = StandardScaler()

X_b_trainval_s = b_scaler.fit_transform(X_b_trainval)

X_b_test_s    = b_scaler.transform(X_b_test)
 
# ── 6. Train baseline Decision Tree ──────────────────────────────────────────

print("\nTraining Baseline Decision Tree …")

dt_baseline = DecisionTreeClassifier(random_state=42)

dt_baseline.fit(X_b_trainval_s, y_b_trainval)
 
# ── 7. Load extended baseline RF (if available) ───────────────────────────────

rf_extended   = None

X_e_test_s    = None

y_e_test      = None
 
if ext_csvs_exist and ext_model_exists:

    print("Loading Extended Baseline Random Forest …")

    e_train = pd.read_csv(EXT_TRAIN)

    e_val   = pd.read_csv(EXT_VAL)

    e_test  = pd.read_csv(EXT_TEST)
 
    e_feats = [c for c in EXTENDED_FEATURES if c in e_test.columns]

    missing_e = set(EXTENDED_FEATURES) - set(e_feats)

    if missing_e:

        print(f"  [Extended] Missing columns (excluded): {missing_e}")

    print(f"  [Extended] Features in use ({len(e_feats)}): {e_feats}")
 
    e_scaler    = joblib.load(EXT_SCALER_PATH)

    rf_extended = joblib.load(EXT_MODEL_PATH)
 
    # The saved scaler was fitted on the specific columns present at training

    # time — use only those columns, in the same order

    try:

        X_e_test_s = e_scaler.transform(e_test[e_feats].values)

        y_e_test   = e_test[TARGET].values

    except Exception as exc:

        print(f"  [Extended] Scaler transform failed: {exc}")

        print("  Skipping Extended Baseline Random Forest.")

        rf_extended = None
 
# ── 8. Evaluation helper ──────────────────────────────────────────────────────

def evaluate(model, X, y, label):

    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)[:, 1]

    acc    = accuracy_score(y, y_pred)

    auc    = roc_auc_score(y, y_prob)

    rep    = classification_report(

        y, y_pred, target_names=["Benign", "Cancer"], output_dict=True

    )

    print(f"\n{'─' * 56}")

    print(f"  {label}")

    print(f"{'─' * 56}")

    print(f"  Accuracy          : {acc:.4f}")

    print(f"  AUC-ROC           : {auc:.4f}")

    print(f"  Precision (Cancer): {rep['Cancer']['precision']:.4f}")

    print(f"  Recall    (Cancer): {rep['Cancer']['recall']:.4f}")

    print(f"  F1        (Cancer): {rep['Cancer']['f1-score']:.4f}")

    print(classification_report(y, y_pred, target_names=["Benign", "Cancer"]))

    return {

        "model":     label,

        "accuracy":  round(acc,                           4),

        "auc":       round(auc,                           4),

        "precision": round(rep["Cancer"]["precision"],    4),

        "recall":    round(rep["Cancer"]["recall"],       4),

        "f1":        round(rep["Cancer"]["f1-score"],     4),

        "y_pred":    y_pred,

        "y_prob":    y_prob,

        "y_true":    y,

    }
 
# ── 9. Run evaluations ────────────────────────────────────────────────────────

results = {}
 
results["Baseline\nDecision Tree"] = evaluate(

    dt_baseline, X_b_test_s, y_b_test,

    "Baseline – Decision Tree  "

    f"(features: {', '.join(b_feats)})"

)
 
if rf_extended is not None:

    results["Extended Baseline\nRandom Forest"] = evaluate(

        rf_extended, X_e_test_s, y_e_test,

        "Extended Baseline – Random Forest  "

        f"(features: {', '.join(e_feats)})"

    )
 
# ── 10. Summary table ─────────────────────────────────────────────────────────

summary = pd.DataFrame([

    {k: v for k, v in r.items() if k not in ("y_pred", "y_prob", "y_true")}

    for r in results.values()

])

print("\n\n══════════════════════════════════════════════════════")

print("  COMPARISON SUMMARY — TEST SET")

print("══════════════════════════════════════════════════════")

print(summary.to_string(index=False))

summary.to_csv("comparison_results.csv", index=False)

print("\nSaved: comparison_results.csv")
 
# ── 11. Plot 1 – Metric bar chart ─────────────────────────────────────────────

metrics      = ["accuracy", "auc", "precision", "recall", "f1"]

model_names  = list(results.keys())

n_models     = len(model_names)

x            = np.arange(len(metrics))

width        = 0.35 if n_models == 2 else 0.25

colors       = ["#4C72B0", "#C44E52", "#55A868"][:n_models]
 
fig, ax = plt.subplots(figsize=(12, 6))

offsets = np.linspace(-(width * (n_models - 1) / 2),

                       width * (n_models - 1) / 2, n_models)
 
for i, (name, color) in enumerate(zip(model_names, colors)):

    vals = [results[name][m] for m in metrics]

    bars = ax.bar(x + offsets[i], vals, width,

                  label=name.replace("\n", " "),

                  color=color, alpha=0.85)

    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
 
ax.set_xticks(x)

ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11)

ax.set_ylim(0, 1.18)

ax.set_ylabel("Score", fontsize=12)

ax.set_title("Baseline vs Extended Baseline — Test Set Metrics",

             fontsize=14, fontweight="bold")

ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8,

           alpha=0.6, label="Chance level (0.5)")

ax.legend(fontsize=9)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig("comparison_metrics_bar.png", dpi=150)

print("Saved: comparison_metrics_bar.png")
 
# ── 12. Plot 2 – ROC curves ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

for (name, r), color in zip(results.items(), colors):

    fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])

    ax.plot(fpr, tpr,

            label=f"{name.replace(chr(10), ' ')}  (AUC = {r['auc']:.3f})",

            color=color, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

ax.set_xlabel("False Positive Rate", fontsize=12)

ax.set_ylabel("True Positive Rate", fontsize=12)

ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight="bold")

ax.legend(fontsize=9)

ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("comparison_roc_curves.png", dpi=150)

print("Saved: comparison_roc_curves.png")
 
# ── 13. Plot 3 – Confusion matrices ──────────────────────────────────────────

n_plots = len(results)

fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

if n_plots == 1:

    axes = [axes]   # make iterable

fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
 
for ax, (name, r) in zip(axes, results.items()):

    cm   = confusion_matrix(r["y_true"], r["y_pred"])

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm, display_labels=["Benign", "Cancer"]

    )

    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title(

        f"{name.replace(chr(10), ' ')}\n"

        f"Acc={r['accuracy']:.3f}  AUC={r['auc']:.3f}",

        fontsize=9,

    )
 
plt.tight_layout()

plt.savefig("comparison_confusion_matrices.png", dpi=150)

print("Saved: comparison_confusion_matrices.png")
 
print("\n✓  All outputs written successfully.")
 