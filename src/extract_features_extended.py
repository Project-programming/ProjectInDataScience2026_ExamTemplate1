"""
extract_features_extended.py

Extracts all 13 extended features:
  - 11 baseline features (asymmetry, border irregularity, 9 colour sub-features)
  - diameter
  - hair_coverage

For train / validation / test splits and saves them as CSV files.

Usage:
    python extract_features_extended.py
"""

import os
import numpy as np
import pandas as pd
from skimage.io import imread

#importing functions from other files 
from split_data_in_3sets import X_train, y_train, X_val, y_val, X_test, y_test
from featureA_baseline import asymmetry
from featureB_baseline import border_irregularity
from featureC_baseline import color_complexity
from featureD import diameter     # diameter feature
from feature_hair_shortcut import hair_coverage         # hair coverage feature

#paths, change as needed 
MASK_DIR = r"/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data/masks"
DATA_DIR = r"/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data"

FEATURE_COLS = [
    'img_id',
    'asymmetry_score',
    'border_irregularity',
    'frac_white', 'frac_red', 'frac_light_brown',
    'frac_dark_brown', 'frac_blue_gray', 'frac_black',
    'n_distinct_colors', 'color_entropy', 'off_palette_dist',
    'diameter',
    'hair_coverage',
    'is_cancer',
]


# helper 
def extract_features_for_split(img_paths, labels, mask_dir, split_name="split"):
    """
    Extract all 13 extended features for one data split.

    Parameters
    ----------
    img_paths : list of str
        Absolute paths to the skin-lesion images.
    labels : list / array of int
        Ground-truth labels (0 = benign, 1 = cancer).
    mask_dir : str
        Directory containing binary lesion masks named
        ``<file_id>_mask.png``.
    split_name : str
        Name used in progress messages (e.g. 'train').

    Returns
    -------
    pd.DataFrame
        One row per image with all 13 features plus img_id and is_cancer.
    """
    results = []
    missing = 0

    for i, (img_path, label) in enumerate(zip(img_paths, labels)):
        file_id   = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{file_id}_mask.png")

        if not os.path.exists(mask_path):
            print(f"  [{split_name}] Skipping {file_id}: mask not found.")
            missing += 1
            continue

        mask_img = imread(mask_path, as_gray=True)

        # Shape features (from mask)
        asym  = asymmetry(mask_img)
        bord  = border_irregularity(mask_img)
        diam  = diameter(mask_img)

        # Colour features (from raw image + mask)
        cc = color_complexity(img_path, mask_path)

        # Hair coverage (from raw uncleaned image)
        h_cov = hair_coverage(img_path)

        results.append({
            'img_id':              file_id,
            'asymmetry_score':     asym,
            'border_irregularity': bord,
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
            'hair_coverage':       h_cov,
            'is_cancer':           label,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{split_name}] Processed {i + 1}/{len(img_paths)} images …")

    print(f"  [{split_name}] Done — {len(results)} extracted, {missing} skipped.")
    return pd.DataFrame(results, columns=FEATURE_COLS)


# main
def main():
    print(f"Training images  : {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Test images      : {len(X_test)}")



    # Train
    train_df = extract_features_for_split(X_train, y_train, MASK_DIR, "train")
    train_df.to_csv(os.path.join(DATA_DIR, "features_extended_train.csv"), index=False)
    print(f"Saved features_extended_train.csv  ({len(train_df)} rows)\n")

    # Validation
    val_df = extract_features_for_split(X_val, y_val, MASK_DIR, "validation")
    val_df.to_csv(os.path.join(DATA_DIR, "features_extended_validation.csv"), index=False)
    print(f"Saved features_extended_validation.csv  ({len(val_df)} rows)\n")

    # Test
    test_df = extract_features_for_split(X_test, y_test, MASK_DIR, "test")
    test_df.to_csv(os.path.join(DATA_DIR, "features_extended_testing.csv"), index=False)
    print(f"Saved features_extended_testing.csv  ({len(test_df)} rows)\n")

    print("Extended feature extraction complete.")


if __name__ == "__main__":
    main()
