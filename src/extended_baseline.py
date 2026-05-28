import numpy as np
import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from skimage.io import imread
from scipy.spatial.distance import cdist

# ── Data split ────────────────────────────────────────────────────────────────
from split_data_in_3sets import X_train, y_train, X_val, y_val, X_test, y_test

# ── Extended cleaner ──────────────────────────────────────────────────────────
from clean_imgs_extenB import preprocess_img, detect_hair

# ── Baseline ABC features ─────────────────────────────────────────────────────
from featureA_baseline import asymmetry
from featureB_baseline import border_irregularity
from featureC_baseline import color_complexity
from featureD import diameter
from feature_hair_shortcut import hair_coverage

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

np.random.seed(1907)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

mask_dir = r"C:\Users\Администратор\Documents\GitHub\ProjectInDataScience2026_ExamTemplate1\data\masks"

def extract_features(img_paths, labels, split_name):
    results = []
    print(f"\n── Extracting: {split_name} ({len(img_paths)} images) ──")

    for i, (img_path, label) in enumerate(zip(img_paths, labels)):
        file_id   = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{file_id}_mask.png")

        if not os.path.exists(mask_path):
            print(f"  Skipping {file_id}: mask not found.")
            continue

        # Extended preprocessing
        img_clean, _, _, _ = preprocess_img(img_path)

        # Load mask
        mask_img = imread(mask_path, as_gray=True)

        # A: Asymmetry
        asym = asymmetry(mask_img)

        # B: Border irregularity
        border = border_irregularity(mask_img)

        # C: Colour complexity (entropy summary value, index 7)
        cc = color_complexity(img_path, mask_path)
        colour_entropy = cc[7]

        # D: Diameter
        diam = diameter(mask_img)

        # E: Hair coverage
        hair_cov = hair_coverage(img_path)

        results.append({
            "img_id":              file_id,
            "asymmetry_score":     asym,
            "border_irregularity": border,
            "colour_complexity":   colour_entropy,
            "diameter_px":         diam,
            "hair_coverage":       hair_cov,
            "is_cancer":           label,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(img_paths)} done")

    df = pd.DataFrame(results)
    csv_name = f"features_extended_{split_name}.csv"
    df.to_csv(csv_name, index=False)
    print(f"  Saved {csv_name}  ({len(df)} rows)")
    return df


train_df = extract_features(X_train, y_train, "train")
val_df   = extract_features(X_val,   y_val,   "validation")
test_df  = extract_features(X_test,  y_test,  "testing")


feature_cols = [
    "asymmetry_score",
    "border_irregularity",
    "colour_complexity",
    "diameter_px",
    "hair_coverage",
]

x_train = train_df[feature_cols].values;  y_train_arr = train_df["is_cancer"].values
x_val   = val_df[feature_cols].values;    y_val_arr   = val_df["is_cancer"].values
x_test  = test_df[feature_cols].values;   y_test_arr  = test_df["is_cancer"].values


scaler    = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_val_s   = scaler.transform(x_val)
x_test_s  = scaler.transform(x_test)
joblib.dump(scaler, "extended_baseline_scaler.pkl")


decision_tree = DecisionTreeClassifier(random_state=1907)
decision_tree.fit(x_train_s, y_train_arr)
print("\nDecision Tree  — validation accuracy:",
      round(decision_tree.score(x_val_s, y_val_arr), 4))


print("\n── Hyperparameter search ──────────────────────────────────")
print(f"{'n_estimators':>12}  {'max_depth':>9}  {'val_acc':>8}  {'val_auc':>8}")

search_results = []
for n_est in [10, 50, 100, 200]:
    for depth in [1, 3, 5, None]:
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                    random_state=1907)
        rf.fit(x_train_s, y_train_arr)
        acc = rf.score(x_val_s, y_val_arr)
        auc = roc_auc_score(y_val_arr, rf.predict_proba(x_val_s)[:, 1])
        print(f"{n_est:>12}  {str(depth):>9}  {acc:>8.4f}  {auc:>8.4f}")
        search_results.append({"n_estimators": n_est, "max_depth": depth,
                                "val_acc": acc, "val_auc": auc})


random_forest = RandomForestClassifier(
    n_estimators=10,   # number of trees
    max_depth=None,    # complexity of each tree (None = fully grown)
    random_state=1907,
)
random_forest.fit(x_train_s, y_train_arr)
print("\nRandom Forest  — validation accuracy:",
      round(random_forest.score(x_val_s, y_val_arr), 4))


y_pred_dt = decision_tree.predict(x_val_s)
y_prob_dt = decision_tree.predict_proba(x_val_s)[:, 1]
y_pred_rf = random_forest.predict(x_val_s)
y_prob_rf = random_forest.predict_proba(x_val_s)[:, 1]

print("\n Decision Tree (Validation)")
print(classification_report(y_val_arr, y_pred_dt, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_val_arr, y_prob_dt):.4f}")

print("\n Random Forest (Validation)")
print(classification_report(y_val_arr, y_pred_rf, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_val_arr, y_prob_rf):.4f}")

y_pred_rf_test = random_forest.predict(x_test_s)
y_prob_rf_test = random_forest.predict_proba(x_test_s)[:, 1]

print("\n Random Forest (Test Set)")
print(classification_report(y_test_arr, y_pred_rf_test, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_test_arr, y_prob_rf_test):.4f}")

print("\n Feature Importances")
for name, imp in zip(feature_cols, random_forest.feature_importances_):
    print(f"  {name:<25} {imp:.4f}")


# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_val_arr, y_pred_dt),
                       display_labels=["Benign", "Cancer"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title(f"Decision Tree (Validation)\n"
                  f"Acc={decision_tree.score(x_val_s, y_val_arr):.2f}  "
                  f"AUC={roc_auc_score(y_val_arr, y_prob_dt):.2f}")
ConfusionMatrixDisplay(confusion_matrix(y_val_arr, y_pred_rf),
                       display_labels=["Benign", "Cancer"]).plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"Random Forest (Validation)\n"
                  f"Acc={random_forest.score(x_val_s, y_val_arr):.2f}  "
                  f"AUC={roc_auc_score(y_val_arr, y_prob_rf):.2f}")
plt.suptitle("Extended Baseline — Validation Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("extended_baseline_confusion_matrices.png", dpi=150)
print("\nSaved: extended_baseline_confusion_matrices.png")

# Feature importances
plt.figure(figsize=(7, 4))
colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
bars = plt.barh(feature_cols, random_forest.feature_importances_, color=colours)
plt.bar_label(bars, fmt="%.3f", padding=4)
plt.xlabel("Importance")
plt.title("Random Forest — Feature Importances (Extended Baseline)")
plt.xlim(0, max(random_forest.feature_importances_) * 1.25)
plt.tight_layout()
plt.savefig("extended_baseline_feature_importances.png", dpi=150)
print("Saved: extended_baseline_feature_importances.png")

# Hyperparameter heatmaps
results_df    = pd.DataFrame(search_results)
depths_labels = ["1", "3", "5", "None"]
n_ests        = [10, 50, 100, 200]
acc_grid      = results_df["val_acc"].values.reshape(4, 4)
auc_grid      = results_df["val_auc"].values.reshape(4, 4)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, grid, title in zip(axes, [acc_grid, auc_grid],
                            ["Validation Accuracy", "Validation AUC"]):
    im = ax.imshow(grid, cmap="YlGn", vmin=grid.min()-0.02, vmax=grid.max()+0.02)
    ax.set_xticks(range(4)); ax.set_xticklabels(depths_labels)
    ax.set_yticks(range(4)); ax.set_yticklabels(n_ests)
    ax.set_xlabel("max_depth"); ax.set_ylabel("n_estimators")
    ax.set_title(title); plt.colorbar(im, ax=ax)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                    fontsize=9,
                    color="black" if grid[i,j] < grid.max()-0.03 else "white")
plt.suptitle("Hyperparameter Search — Extended Baseline", fontweight="bold")
plt.tight_layout()
plt.savefig("extended_baseline_hyperparam_search.png", dpi=150)
print("Saved: extended_baseline_hyperparam_search.png")

joblib.dump(random_forest, "extended_baseline_random_forest.pkl")
print("Saved: extended_baseline_random_forest.pkl")
