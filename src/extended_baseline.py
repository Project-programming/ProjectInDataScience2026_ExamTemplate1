import numpy as np
import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy.spatial.distance import cdist
import cv2

#importing functions from other files 
from split_data_in_3sets import X_train, y_train, X_val, y_val, X_test, y_test
from clean_imgs_baseline import preprocess_img
from featureA_baseline import asymmetry
from featureB_baseline import border_irregularity
from featureC_baseline import color_complexity
from featureD import diameter
from feature_hair_shortcut import hair_coverage

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

mask_dir = r"/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data/masks"

train_results = []
print(f"the length of the training data is: {len(X_train)}")

for i in range(len(X_train)):
    img_path = X_train[i]
    label    = y_train[i]
    file_id  = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f"{file_id}_mask.png")

    if not os.path.exists(mask_path):
        print(f"Skipping: {file_id}_mask.png not found.")
        continue

    mask_img = imread(mask_path, as_gray=True)
    cc = color_complexity(img_path, mask_path)

    train_results.append({
        'img_id':              file_id,
        'asymmetry_score':     asymmetry(mask_img),
        'border_irregularity': border_irregularity(mask_img),
        'frac_white':          cc[0],
        'frac_red':            cc[1],
        'frac_light_brown':    cc[2],
        'frac_dark_brown':     cc[3],
        'frac_blue_gray':      cc[4],
        'frac_black':          cc[5],
        'n_distinct_colors':   cc[6],
        'color_entropy':       cc[7],
        'off_palette_dist':    cc[8],
        'diameter_px':         diameter(mask_img),
        'hair_coverage':       hair_coverage(img_path)[0],
        'is_cancer':           label,
    })

    if (i + 1) % 10 == 0:
        print(f"  train: {i + 1}/{len(X_train)} done")

train_df = pd.DataFrame(train_results)
train_df.to_csv("features_extended_train.csv", index=False)
print(f"Saved features_extended_train.csv ({len(train_df)} rows)")

validation_results = []

for i in range(len(X_val)):
    img_path = X_val[i]
    label    = y_val[i]
    file_id  = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f"{file_id}_mask.png")

    if not os.path.exists(mask_path):
        continue

    mask_img = imread(mask_path, as_gray=True)
    cc = color_complexity(img_path, mask_path)

    validation_results.append({
        'img_id':              file_id,
        'asymmetry_score':     asymmetry(mask_img),
        'border_irregularity': border_irregularity(mask_img),
        'frac_white':          cc[0],
        'frac_red':            cc[1],
        'frac_light_brown':    cc[2],
        'frac_dark_brown':     cc[3],
        'frac_blue_gray':      cc[4],
        'frac_black':          cc[5],
        'n_distinct_colors':   cc[6],
        'color_entropy':       cc[7],
        'off_palette_dist':    cc[8],
        'diameter_px':         diameter(mask_img),
        'hair_coverage':       hair_coverage(img_path)[0],
        'is_cancer':           label,
    })

    if (i + 1) % 10 == 0:
        print(f"  val: {i + 1}/{len(X_val)} done")

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv("features_extended_validation.csv", index=False)
print(f"Saved features_extended_validation.csv ({len(validation_df)} rows)")

testing_results = []

for i in range(len(X_test)):
    img_path = X_test[i]
    label    = y_test[i]
    file_id  = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f"{file_id}_mask.png")

    if not os.path.exists(mask_path):
        continue

    mask_img = imread(mask_path, as_gray=True)
    cc = color_complexity(img_path, mask_path)

    testing_results.append({
        'img_id':              file_id,
        'asymmetry_score':     asymmetry(mask_img),
        'border_irregularity': border_irregularity(mask_img),
        'frac_white':          cc[0],
        'frac_red':            cc[1],
        'frac_light_brown':    cc[2],
        'frac_dark_brown':     cc[3],
        'frac_blue_gray':      cc[4],
        'frac_black':          cc[5],
        'n_distinct_colors':   cc[6],
        'color_entropy':       cc[7],
        'off_palette_dist':    cc[8],
        'diameter_px':         diameter(mask_img),
        'hair_coverage':       hair_coverage(img_path)[0],
        'is_cancer':           label,
    })

    if (i + 1) % 10 == 0:
        print(f"  test: {i + 1}/{len(X_test)} done")

testing_df = pd.DataFrame(testing_results)
testing_df.to_csv("features_extended_testing.csv", index=False)
print(f"Saved features_extended_testing.csv ({len(testing_df)} rows)")

feature_cols = [
    'asymmetry_score', 'border_irregularity',
    'frac_white', 'frac_red', 'frac_light_brown', 'frac_dark_brown',
    'frac_blue_gray', 'frac_black', 'n_distinct_colors',
    'color_entropy', 'off_palette_dist',
    'diameter_px', 'hair_coverage',
]

x_train = train_df[feature_cols].values;      y_train_arr = train_df['is_cancer'].values
x_val   = validation_df[feature_cols].values; y_val_arr   = validation_df['is_cancer'].values
x_test  = testing_df[feature_cols].values;    y_test_arr  = testing_df['is_cancer'].values

# Add this right before your scaler!
for col in feature_cols:
    print(f"{col}: {type(train_df[col].iloc[0])} - Value: {train_df[col].iloc[0]}")
scaler    = StandardScaler()


x_train_s = scaler.fit_transform(x_train)
x_val_s   = scaler.transform(x_val)
x_test_s  = scaler.transform(x_test)
joblib.dump(scaler, "extended_baseline_scaler.pkl")

decision_tree = DecisionTreeClassifier(random_state=1907)
decision_tree.fit(x_train_s, y_train_arr)
print("\nDecision Tree  — validation accuracy:",
      round(decision_tree.score(x_val_s, y_val_arr), 4))

print("\n Hyperparameter search")
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
    n_estimators=10,
    max_depth=None,
    random_state=1907,
)
random_forest.fit(x_train_s, y_train_arr)
print("\nRandom Forest  — validation accuracy:",
      round(random_forest.score(x_val_s, y_val_arr), 4))

y_pred_dt = decision_tree.predict(x_val_s)
y_prob_dt = decision_tree.predict_proba(x_val_s)[:, 1]
y_pred_rf = random_forest.predict(x_val_s)
y_prob_rf = random_forest.predict_proba(x_val_s)[:, 1]

print("\n--- Decision Tree (Validation) ---")
print(classification_report(y_val_arr, y_pred_dt, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_val_arr, y_prob_dt):.4f}")

print("\n--- Random Forest (Validation) ---")
print(classification_report(y_val_arr, y_pred_rf, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_val_arr, y_prob_rf):.4f}")

y_pred_rf_test = random_forest.predict(x_test_s)
y_prob_rf_test = random_forest.predict_proba(x_test_s)[:, 1]

print("\n--- Random Forest (Test Set) ---")
print(classification_report(y_test_arr, y_pred_rf_test, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_test_arr, y_prob_rf_test):.4f}")

print("\n--- Feature Importances ---")
for name, imp in zip(feature_cols, random_forest.feature_importances_):
    print(f"  {name:<25} {imp:.4f}")


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

plt.figure(figsize=(8, 6))
bars = plt.barh(feature_cols, random_forest.feature_importances_)
plt.bar_label(bars, fmt="%.3f", padding=4)
plt.xlabel("Importance")
plt.title("Random Forest — Feature Importances (Extended Baseline)")
plt.xlim(0, max(random_forest.feature_importances_) * 1.3)
plt.tight_layout()
plt.savefig("extended_baseline_feature_importances.png", dpi=150)
print("Saved: extended_baseline_feature_importances.png")

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
            ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="black" if grid[i,j] < grid.max()-0.03 else "white")
plt.suptitle("Hyperparameter Search — Extended Baseline", fontweight="bold")
plt.tight_layout()

#saving the files
plt.savefig("extended_baseline_hyperparam_search.png", dpi=150)
print("Saved: extended_baseline_hyperparam_search.png")

joblib.dump(random_forest, "extended_baseline_random_forest.pkl")
print("Saved: extended_baseline_random_forest.pkl")