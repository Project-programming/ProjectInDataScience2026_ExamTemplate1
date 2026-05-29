"""
main_baseline.py

Full pipeline for the baseline Decision Tree model.

Three modes
-----------
Script 1 — Extract features from raw images, train, and evaluate.
Script 2 — Train and evaluate from an existing features CSV.
Script 3 — Evaluate a saved trained model on completely unseen data
           (all 11 baseline features).

Usage
-----
    python main_baseline.py --script 1
    python main_baseline.py --script 2 --feature_path data/features_train.csv
    python main_baseline.py --script 3 --model_path results/models/baseline_decision_tree.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
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
from sklearn.tree import DecisionTreeClassifier

# ── Feature imports ───────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent / "src"))
from featureA_baseline import asymmetry
from featureB_baseline import border_irregularity
from featureC_baseline import color_complexity

# ── Constants ─────────────────────────────────────────────────────────────────
BASELINE_FEATURE_COLS = [
    'asymmetry_score',
    'border_irregularity',
    'frac_white', 'frac_red', 'frac_light_brown',
    'frac_dark_brown', 'frac_blue_gray', 'frac_black',
    'n_distinct_colors', 'color_entropy', 'off_palette_dist',
]  # 11 features


#paths, change if needed 
MASK_DIR = r"/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data/masks"



# ── Feature extraction helper ─────────────────────────────────────────────────
def extract_features_from_images(img_paths, labels, mask_dir):
    """
    Extract all 11 baseline features from a list of image paths.

    Parameters
    ----------
    img_paths : list of str
    labels    : list / array of int  (0=benign, 1=cancer)
    mask_dir  : str  path to mask directory

    Returns
    -------
    pd.DataFrame  with 11 feature columns + img_id + is_cancer
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
            'is_cancer':           label,
        })
    return pd.DataFrame(results)


# ── Train helper ──────────────────────────────────────────────────────────────
def train_model(train_df, val_df, feature_cols, result_dir):
    """
    Fit a StandardScaler + DecisionTreeClassifier on train_df,
    evaluate on val_df, save the scaler and model.

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

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("\n── Validation Results ──────────────────────────────")
    print(f"Accuracy : {accuracy_score(y_val, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_val, y_prob):.4f}")
    print(classification_report(y_val, y_pred, target_names=['Benign', 'Cancer']))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred,
        display_labels=["Benign", "Cancer"],
        cmap="Blues", values_format="d"
    )
    plt.title("Baseline Decision Tree — Validation Confusion Matrix")
    plt.grid(False)
    cm_path = result_dir / "figures" / "baseline_dt_confusion_matrix.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save model + scaler
    models_dir = result_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "baseline_scaler.pkl")
    joblib.dump(model,  models_dir / "baseline_decision_tree.pkl")
    print(f"Model and scaler saved to {models_dir}")

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
        cmap="Blues", values_format="d"
    )
    plt.title(f"Baseline Decision Tree — {split_name} Confusion Matrix")
    plt.grid(False)
    fig_path = result_dir / "figures" / f"baseline_dt_cm_{split_name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    preds_df = pd.DataFrame({
        'img_id':      df['img_id'] if 'img_id' in df.columns else range(len(df)),
        'label':       y,
        'prediction':  y_pred,
        'probability': y_prob,
    })
    preds_path = result_dir / "predictions" / f"baseline_predictions_{split_name}.csv"
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

    # Save CSVs
    data_dir = Path(result_dir).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "features_train.csv", index=False)
    val_df.to_csv(data_dir  / "features_validation.csv", index=False)
    test_df.to_csv(data_dir / "features_testing.csv", index=False)

    scaler, model = train_model(train_df, val_df, BASELINE_FEATURE_COLS, result_dir)
    evaluate(test_df, scaler, model, BASELINE_FEATURE_COLS, result_dir, "test")


# ══════════════════════════════════════════════════════════════════════════════
# Script 2 — train + evaluate from existing CSVs
# ══════════════════════════════════════════════════════════════════════════════
def script2_train_from_csv(train_path, val_path, test_path, result_dir):
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # Keep only feature cols that are actually present (safety check)
    feature_cols = [c for c in BASELINE_FEATURE_COLS if c in train_df.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    scaler, model = train_model(train_df, val_df, feature_cols, result_dir)
    evaluate(test_df, scaler, model, feature_cols, result_dir, "test")


# ══════════════════════════════════════════════════════════════════════════════
# Script 3 — evaluate saved model on completely unseen data
#            works with all 11 baseline features
# ══════════════════════════════════════════════════════════════════════════════
def script3_evaluate_unseen(image_dir, mask_dir, metadata_path,
                             model_path, scaler_path, result_dir):
    """
    Evaluate a saved baseline model on an unseen dataset.

    Extracts all 11 baseline features from images that have matching masks,
    then scores them with the saved scaler + model.

    Parameters
    ----------
    image_dir     : str   directory of unseen images
    mask_dir      : str   directory of corresponding masks
    metadata_path : str   CSV with at least columns img_id and is_cancer
    model_path    : str   path to saved .pkl model
    scaler_path   : str   path to saved .pkl scaler
    result_dir    : str   where to write outputs
    """
    image_dir  = Path(image_dir)
    mask_dir   = Path(mask_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    scaler   = joblib.load(scaler_path)
    model    = joblib.load(model_path)

    # Build img_path + label lists from metadata
    img_paths, labels = [], []
    for _, row in metadata.iterrows():
        img_file = image_dir / row['img_id']
        if not img_file.exists():
            # try adding .png if not present
            img_file = image_dir / (str(row['img_id']).replace('.png', '') + '.png')
        img_paths.append(str(img_file))
        labels.append(int(row['is_cancer']))

    print(f"Extracting features for {len(img_paths)} unseen images …")
    df = extract_features_from_images(img_paths, labels, str(mask_dir))

    if df.empty:
        print("No features extracted — check image/mask paths.")
        return

    feature_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    print(f"Using {len(feature_cols)}/11 features.")

    evaluate(df, scaler, model, feature_cols, result_dir, "unseen")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    base_dir   = Path(__file__).parent.resolve()
    result_dir = base_dir / "results"

    parser = argparse.ArgumentParser(description="Baseline model pipeline.")
    parser.add_argument('--script', type=int, choices=[1, 2, 3],
                        help="1=extract+train+eval, 2=train from CSV, 3=eval unseen")
    parser.add_argument('--train_path',  type=str, default=str(base_dir / "data" / "features_train.csv"))
    parser.add_argument('--val_path',    type=str, default=str(base_dir / "data" / "features_validation.csv"))
    parser.add_argument('--test_path',   type=str, default=str(base_dir / "data" / "features_testing.csv"))
    parser.add_argument('--model_path',  type=str, default=str(result_dir / "models" / "baseline_decision_tree.pkl"))
    parser.add_argument('--scaler_path', type=str, default=str(result_dir / "models" / "baseline_scaler.pkl"))
    parser.add_argument('--image_dir',   type=str, default=str(base_dir / "data" / "imgs"))
    parser.add_argument('--mask_dir',    type=str, default=MASK_DIR)
    parser.add_argument('--metadata',    type=str, default=str(base_dir / "data" / "features.csv"))
    args, _ = parser.parse_known_args()

    # Interactive fallback
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
        # Combine all three splits into one metadata file for evaluation
        train_df = pd.read_csv(Path(base_dir) / "data" / "features_train.csv")
        val_df   = pd.read_csv(Path(base_dir) / "data" / "features_validation.csv")
        test_df  = pd.read_csv(Path(base_dir) / "data" / "features_testing.csv")
        
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Save the combined file — this becomes your metadata
        metadata_path = Path(base_dir) / "data" / "features_all.csv"
        combined[['img_id', 'is_cancer']].to_csv(metadata_path, index=False)

        image_dir   = Path(base_dir) / "data" / "imgs"
        mask_dir    = Path(base_dir) / "data" / "masks"
        result_dir  = Path(base_dir) / "results"
        model_path  = Path(base_dir) / "results" / "models" / "baseline_decision_tree.pkl"
        scaler_path = Path(base_dir) / "results" / "models" / "baseline_scaler.pkl"
        result_dir.mkdir(parents=True, exist_ok=True)

        script3_evaluate_unseen(
            image_dir=image_dir,
            mask_dir=mask_dir,
            metadata_path=metadata_path,
            model_path=model_path,
            scaler_path=scaler_path,
            result_dir=str(result_dir)
    )