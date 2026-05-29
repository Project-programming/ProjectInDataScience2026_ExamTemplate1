"""
main_extended.py

the full pipeline for the extended Random Forest model.

Three modes
-----------
Script 1 — Extract all 13 features from raw images, train, and evaluate.
Script 2 — Train and evaluate from existing feature CSVs.
Script 3 — Evaluate a saved trained model on completely unseen data
           (all 13 extended features).

Usage
-----
    python main_extended.py --script 1
    python main_extended.py --script 2 --train_path data/features_extended_train.csv ...
    python main_extended.py --script 3 --model_path results/models/extended_random_forest.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# imports
sys.path.append(str(Path(__file__).parent / "src"))
from featureA_baseline import asymmetry
from featureB_baseline import border_irregularity
from featureC_baseline import color_complexity
from featureD import diameter
from feature_hair_shortcut import hair_coverage

# constants
EXTENDED_FEATURE_COLS = [
    'asymmetry_score',
    'border_irregularity',
    'frac_white', 'frac_red', 'frac_light_brown',
    'frac_dark_brown', 'frac_blue_gray', 'frac_black',
    'n_distinct_colors', 'color_entropy', 'off_palette_dist',
    'diameter',
    'hair_coverage',
]  # 13 features

MASK_DIR = r"/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data/masks"

# Best hyperparameters from grid search
BEST_N_ESTIMATORS = 50
BEST_MAX_DEPTH    = None   # fully grown trees


# ── Feature extraction helper ─────────────────────────────────────────────────
def extract_features_from_images(img_paths, labels, mask_dir):
    """
    Extract all 13 extended features from a list of image paths.

    Parameters
    ----------
    img_paths : list of str
    labels    : list / array of int  (0=benign, 1=cancer)
    mask_dir  : str  path to mask directory

    Returns
    -------
    pd.DataFrame  with 13 feature columns + img_id + is_cancer
    """
    results = []
    for img_path, label in zip(img_paths, labels):
        file_id   = Path(img_path).stem
        mask_path = Path(mask_dir) / f"{file_id}_mask.png"

        if not mask_path.exists():
            print(f"  Skipping {file_id}: mask not found.")
            continue

        mask_img = imread(str(mask_path), as_gray=True)
        cc       = color_complexity(str(img_path), str(mask_path))
        diam     = diameter(mask_img)
        h_cov    = hair_coverage(str(img_path))

        results.append({
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
            'diameter':            diam,
            'hair_coverage':       h_cov[0],
            'is_cancer':           label,
        })
    return pd.DataFrame(results)


# ── Train helper ──────────────────────────────────────────────────────────────
def train_model(train_df, val_df, feature_cols, result_dir,
                n_estimators=BEST_N_ESTIMATORS, max_depth=BEST_MAX_DEPTH):
    """
    Fit StandardScaler + RandomForestClassifier on train_df,
    evaluate on val_df, save scaler and model.

    Returns
    -------
    scaler, model
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_cols].values
    y_train = train_df['is_cancer'].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df['is_cancer'].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("\n── Validation Results ──────────────────────────────")
    print(f"Accuracy : {accuracy_score(y_val, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_val, y_prob):.4f}")
    print(classification_report(y_val, y_pred, target_names=['Benign', 'Cancer']))

    # Feature importances
    importances = model.feature_importances_
    print("\n── Feature Importances ─────────────────────────────")
    for name, imp in sorted(zip(feature_cols, importances),
                            key=lambda x: x[1], reverse=True):
        print(f"  {name:<25} {imp:.4f}")

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred,
        display_labels=["Benign", "Cancer"],
        cmap="Greens", values_format="d"
    )
    plt.title("Extended Random Forest — Validation Confusion Matrix")
    plt.grid(False)
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "extended_rf_confusion_matrix.png", dpi=150)
    plt.close()

    # Save model + scaler
    models_dir = result_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "extended_scaler.pkl")
    joblib.dump(model,  models_dir / "extended_random_forest.pkl")
    print(f"\nModel and scaler saved to {models_dir}")

    return scaler, model


# ── Evaluate on any dataframe ─────────────────────────────────────────────────
def evaluate(df, scaler, model, feature_cols, result_dir, split_name="test"):
    """
    Evaluate a fitted model on df, print metrics, save confusion matrix
    and predictions CSV.
    """
    result_dir = Path(result_dir)
    X = scaler.transform(df[feature_cols].values)
    y = df['is_cancer'].values

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n── {split_name} Results ──────────────────────────────")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y, y_prob):.4f}")
    print(f"Recall   : {recall_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"F1       : {f1_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred, target_names=['Benign', 'Cancer']))

    ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=["Benign", "Cancer"],
        cmap="Greens", values_format="d"
    )
    plt.title(f"Extended Random Forest — {split_name} Confusion Matrix")
    plt.grid(False)
    fig_path = result_dir / "figures" / f"extended_rf_cm_{split_name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    preds_df = pd.DataFrame({
        'img_id':      df['img_id'] if 'img_id' in df.columns else range(len(df)),
        'label':       y,
        'prediction':  y_pred,
        'probability': y_prob,
    })
    preds_path = result_dir / "predictions" / f"extended_predictions_{split_name}.csv"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Script 1 — extract features, train, evaluate
# ══════════════════════════════════════════════════════════════════════════════
def script1_extract_train_evaluate(mask_dir, result_dir):
    from split_data_in_3sets import X_train, y_train, X_val, y_val, X_test, y_test

    print("Extracting training features …")
    train_df = extract_features_from_images(X_train, y_train, mask_dir)

    print("Extracting validation features …")
    val_df = extract_features_from_images(X_val, y_val, mask_dir)

    print("Extracting test features …")
    test_df = extract_features_from_images(X_test, y_test, mask_dir)

    data_dir = Path(result_dir).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "features_extended_train.csv", index=False)
    val_df.to_csv(data_dir   / "features_extended_validation.csv", index=False)
    test_df.to_csv(data_dir  / "features_extended_testing.csv", index=False)

    scaler, model = train_model(train_df, val_df, EXTENDED_FEATURE_COLS, result_dir)
    evaluate(test_df, scaler, model, EXTENDED_FEATURE_COLS, result_dir, "test")


# ══════════════════════════════════════════════════════════════════════════════
# Script 2 — train + evaluate from existing CSVs
# ══════════════════════════════════════════════════════════════════════════════
def script2_train_from_csv(train_path, val_path, test_path, result_dir):
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    feature_cols = [c for c in EXTENDED_FEATURE_COLS if c in train_df.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    scaler, model = train_model(train_df, val_df, feature_cols, result_dir)
    evaluate(test_df, scaler, model, feature_cols, result_dir, "test")


# ══════════════════════════════════════════════════════════════════════════════
# Script 3 — evaluate saved model on completely unseen data
#            works with all 13 extended features
# ══════════════════════════════════════════════════════════════════════════════
def script3_evaluate_unseen(image_dir, mask_dir, metadata_path, model_path, scaler_path, result_dir):
    image_dir  = Path(image_dir)
    mask_dir   = Path(mask_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    scaler   = joblib.load(scaler_path)
    model    = joblib.load(model_path)

    # Quick check — warn if scaler feature count doesn't match
    expected = scaler.n_features_in_
    print(f"Scaler expects {expected} features — using EXTENDED_FEATURE_COLS ({len(EXTENDED_FEATURE_COLS)})")
    if expected != len(EXTENDED_FEATURE_COLS):
        print("WARNING: scaler was saved with a different number of features.")
        print("Re-run Script 2 to retrain and resave the scaler with 13 features.")
        return

    img_paths, labels = [], []
    for _, row in metadata.iterrows():
        img_file = image_dir / row['img_id']
        if not img_file.exists():
            img_file = image_dir / (str(row['img_id']).replace('.png', '') + '.png')
        img_paths.append(str(img_file))
        labels.append(int(row['is_cancer']))

    print(f"Extracting 13 features for {len(img_paths)} unseen images ...")
    df = extract_features_from_images(img_paths, labels, str(mask_dir))

    if df.empty:
        print("No features extracted — check image/mask paths.")
        return

    feature_cols = [c for c in EXTENDED_FEATURE_COLS if c in df.columns]
    print(f"Using {len(feature_cols)}/13 features.")

    X      = scaler.transform(df[feature_cols].values)
    y      = df['is_cancer'].values
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Print metrics
    print(f"\n── Unseen Data Results ─────────────────────────────")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y, y_prob):.4f}")
    print(f"Recall   : {recall_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"F1       : {f1_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred, target_names=['Benign', 'Cancer']))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=["Benign", "Cancer"],
        cmap="Greens", values_format="d"
    )
    plt.title("Extended Random Forest — Unseen Data Confusion Matrix")
    plt.grid(False)
    fig_path = result_dir / "figures" / "extended_rf_cm_unseen.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {fig_path}")

    # Save predictions CSV (matching the format from the example)
    preds_df = pd.DataFrame({
        'img_id':      df['img_id'],
        'patient_id':  df['img_id'].str.extract(r'PAT_(\d+)_').astype(str).squeeze(),
        'label':       y,
        'prediction':  y_pred,
        'probability': y_prob,
    })
    preds_path = result_dir / "predictions" / "extended_predictions_unseen.csv"
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")
# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    base_dir   = Path(__file__).parent.resolve()
    result_dir = base_dir / "results"

    parser = argparse.ArgumentParser(description="Extended model pipeline.")
    parser.add_argument('--script', type=int, choices=[1, 2, 3],
                        help="1=extract+train+eval, 2=train from CSV, 3=eval unseen")
    parser.add_argument('--train_path',  type=str, default=str(base_dir / "data" / "features_extended_train.csv"))
    parser.add_argument('--val_path',    type=str, default=str(base_dir / "data" / "features_extended_validation.csv"))
    parser.add_argument('--test_path',   type=str, default=str(base_dir / "data" / "features_extended_testing.csv"))
    parser.add_argument('--model_path',  type=str, default=str(result_dir / "models" / "extended_random_forest.pkl"))
    parser.add_argument('--scaler_path', type=str, default=str(result_dir / "models" / "extended_scaler.pkl"))
    parser.add_argument('--image_dir',   type=str, default=str(base_dir / "data" / "imgs"))
    parser.add_argument('--mask_dir',    type=str, default=MASK_DIR)
    parser.add_argument('--metadata',    type=str, default=str(base_dir / "results" / "features.csv"))
    args, _ = parser.parse_known_args()

    if args.script is None and sys.stdin.isatty():
        print("Which script do you want to run?")
        print("  1: Extract features from images, train, and evaluate")
        print("  2: Train and evaluate from existing feature CSVs")
        print("  3: Evaluate saved model on unseen data")
        while True:
            try:
                args.script = int(input("Enter 1, 2, or 3: "))
                if args.script in [1, 2, 3]:
                    break
            except ValueError:
                pass
            print("Please enter 1, 2, or 3.")
    elif args.script is None:
        print("No --script provided. Exiting.")
        sys.exit(1)

    if args.script == 1:
        script1_extract_train_evaluate(args.mask_dir, str(result_dir))

    elif args.script == 2:
        script2_train_from_csv(
            args.train_path, args.val_path, args.test_path, str(result_dir)
        )

    elif args.script == 3:
        train_df = pd.read_csv(Path(base_dir) / "results" / "features_extended_train.csv")
        val_df   = pd.read_csv(Path(base_dir) / "results" / "features_extended_validation.csv")
        test_df  = pd.read_csv(Path(base_dir) / "results" / "features_extended_testing.csv")
        
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        metadata_path = Path(base_dir) / "data" / "features_all.csv"
        combined[['img_id', 'is_cancer']].to_csv(metadata_path, index=False)

        image_dir   = Path(base_dir) / "data" / "imgs"
        mask_dir    = Path(base_dir) / "data" / "masks"
        result_dir  = Path(base_dir) / "results"
        model_path  = Path(base_dir) / "results" / "models" / "extended_random_forest.pkl"  # ← fixed
        scaler_path = Path(base_dir) / "results" / "models" / "extended_scaler.pkl"          # ← fixed
        result_dir.mkdir(parents=True, exist_ok=True)

        script3_evaluate_unseen(
            image_dir=image_dir,
            mask_dir=mask_dir,
            metadata_path=metadata_path,
            model_path=model_path,
            scaler_path=scaler_path,
            result_dir=str(result_dir)
        )  # ← fixed indentation
