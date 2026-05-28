import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
 
np.random.seed(1907)
 
train_df = pd.read_csv("features_train.csv")
val_df   = pd.read_csv("features_validation.csv")
test_df  = pd.read_csv("features_testing.csv")
 
feature_cols = ["asymmetry_score", "border_irregularity", "colour_complexity"]
 
x_train = train_df[feature_cols].values
y_train = train_df["is_cancer"].values
x_val   = val_df[feature_cols].values
y_val   = val_df["is_cancer"].values
x_test  = test_df[feature_cols].values
y_test  = test_df["is_cancer"].values
 
scaler    = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_val_s   = scaler.transform(x_val)
x_test_s  = scaler.transform(x_test)
joblib.dump(scaler, "extended_scaler.pkl")
 
decision_tree = DecisionTreeClassifier(random_state=1907)
decision_tree.fit(x_train_s, y_train)
print("Decision Tree  — validation accuracy:", round(decision_tree.score(x_val_s, y_val), 4))
 
print("\n Hyperparameter search")
print(f"{'n_estimators':>12}  {'max_depth':>9}  {'val_acc':>8}  {'val_auc':>8}")
 
results = []
for n_est in [10, 50, 100, 200]:
    for depth in [1, 3, 5, None]:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            random_state=1907,
        )
        rf.fit(x_train_s, y_train)
        acc = rf.score(x_val_s, y_val)
        auc = roc_auc_score(y_val, rf.predict_proba(x_val_s)[:, 1])
        print(f"{n_est:>12}  {str(depth):>9}  {acc:>8.4f}  {auc:>8.4f}")
        results.append({"n_estimators": n_est, "max_depth": depth,
                         "val_acc": acc, "val_auc": auc})
 
print("\n Best model selected: n_estimators=10, max_depth=None")
 
random_forest = RandomForestClassifier(
    n_estimators=10,   # Change to adjust the number of trees
    max_depth=None,    # Change to adjust the complexity of the trees
    random_state=1907,
)
random_forest.fit(x_train_s, y_train)
print("Random Forest — validation accuracy:", round(random_forest.score(x_val_s, y_val), 4))
 
y_pred_dt  = decision_tree.predict(x_val_s)
y_prob_dt  = decision_tree.predict_proba(x_val_s)[:, 1]
y_pred_rf  = random_forest.predict(x_val_s)
y_prob_rf  = random_forest.predict_proba(x_val_s)[:, 1]
 
print("\n Decision Tree (Validation)")
print(classification_report(y_val, y_pred_dt, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_val, y_prob_dt):.4f}")
 
print("\n Random Forest (Validation)")
print(classification_report(y_val, y_pred_rf, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_val, y_prob_rf):.4f}")
 
y_pred_rf_test = random_forest.predict(x_test_s)
y_prob_rf_test = random_forest.predict_proba(x_test_s)[:, 1]
 
print("\n Random Forest (Test Set)")
print(classification_report(y_test, y_pred_rf_test, target_names=["Benign", "Cancer"]))
print(f"AUC: {roc_auc_score(y_test, y_prob_rf_test):.4f}")
 
importances = random_forest.feature_importances_
print("\n Feature Importances")
for name, imp in zip(feature_cols, importances):
    print(f"  {name:<25} {imp:.4f}")
 
#Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 
cm_dt = confusion_matrix(y_val, y_pred_dt)
ConfusionMatrixDisplay(cm_dt, display_labels=["Benign", "Cancer"]).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title(
    f"Decision Tree (Validation)\n"
    f"Acc={decision_tree.score(x_val_s, y_val):.2f}  "
    f"AUC={roc_auc_score(y_val, y_prob_dt):.2f}")
 
cm_rf = confusion_matrix(y_val, y_pred_rf)
ConfusionMatrixDisplay(cm_rf, display_labels=["Benign", "Cancer"]).plot(
    ax=axes[1], colorbar=False)
axes[1].set_title(
    f"Random Forest  (Validation)\n"
    f"Acc={random_forest.score(x_val_s, y_val):.2f}  "
    f"AUC={roc_auc_score(y_val, y_prob_rf):.2f}")
 
plt.suptitle("Extended Model — Validation Set Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("extended_confusion_matrices.png", dpi=150)
print("\nSaved: extended_confusion_matrices.png")
 
#Feature importance bar chart
plt.figure(figsize=(6, 3.5))
colours = ["#4C72B0", "#DD8452", "#55A868"]
bars = plt.barh(feature_cols, importances, color=colours)
plt.bar_label(bars, fmt="%.3f", padding=4)
plt.xlabel("Importance")
plt.title("Random Forest — Feature Importances")
plt.xlim(0, max(importances) * 1.25)
plt.tight_layout()
plt.savefig("extended_feature_importances.png", dpi=150)
print("Saved: extended_feature_importances.png")
 
#Hyperparameter search heatmaps
results_df = pd.DataFrame(results)
depths_labels = ["1", "3", "5", "None"]
n_ests = [10, 50, 100, 200]
 
acc_grid = results_df["val_acc"].values.reshape(4, 4)
auc_grid = results_df["val_auc"].values.reshape(4, 4)
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, grid, title, fmt in zip(
    axes,
    [acc_grid, auc_grid],
    ["Validation Accuracy", "Validation AUC"],
    [".3f", ".3f"],
):
    im = ax.imshow(grid, cmap="YlGn", vmin=grid.min() - 0.02, vmax=grid.max() + 0.02)
    ax.set_xticks(range(4)); ax.set_xticklabels(depths_labels)
    ax.set_yticks(range(4)); ax.set_yticklabels(n_ests)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("n_estimators")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, format(grid[i, j], fmt),
                    ha="center", va="center", fontsize=9,
                    color="black" if grid[i, j] < grid.max() - 0.03 else "white")
 
plt.suptitle("Hyperparameter Search — n_estimators vs max_depth", fontweight="bold")
plt.tight_layout()
plt.savefig("extended_hyperparam_search.png", dpi=150)
print("Saved: extended_hyperparam_search.png")
 
joblib.dump(random_forest, "extended_random_forest.pkl")
print("Saved: extended_random_forest.pkl")
 