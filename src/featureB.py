import numpy as np
import os
import pandas as pd
from skimage.io import imread
from skimage import measure
# Import training data specifically
from split_data_in_3sets import X_train, y_train


def border_irregularity(mask):
    """
    Computes border irregularity using the Compactness Index (CI).

    CI = perimeter² / (4π × area)
    - A perfect circle gives CI = 1.0 (most regular)
    - Higher values mean a more irregular, jagged border

    Returns a score >= 1.0, rounded to 4 decimal places.
    """
    mask = (mask > 0).astype(np.uint8)

    # Find contours using skimage
    contours = measure.find_contours(mask, level=0.5)
    if len(contours) == 0:
        return 0.0

    # Use the longest contour (main lesion border)
    contour = max(contours, key=len)

    # Perimeter: sum of Euclidean distances between consecutive contour points
    diffs = np.diff(contour, axis=0)
    perimeter = np.sum(np.sqrt((diffs ** 2).sum(axis=1)))

    # Area: number of foreground pixels
    area = np.sum(mask)

    if area == 0 or perimeter == 0:
        return 0.0

    # Compactness Index
    ci = (perimeter ** 2) / (4 * np.pi * area)

    return round(float(ci), 4)


# Getting masks
mask_dir = "/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data/masks/"
train_results = []

print(f"Starting calculation for {len(X_train)} training samples...")

# To make sure dataset is correct length
print(f"The length of the training data is: {len(X_train)}")

# Looping through all training data
for i in range(len(X_train)):
    img_path = X_train[i]
    label = y_train[i]

    # Extract ID and find mask
    base_name = os.path.basename(img_path)
    file_id = os.path.splitext(base_name)[0]
    mask_name = f"{file_id}_mask.png"
    full_mask_path = os.path.join(mask_dir, mask_name)

    if os.path.exists(full_mask_path):
        mask_img = imread(full_mask_path, as_gray=True)
        score = border_irregularity(mask_img)

        train_results.append({
            'img_id': file_id,
            'border_irregularity_score': score,
            'is_cancer': label
        })
    else:
        print(f"Skipping: {mask_name} not found.")

train_df = pd.DataFrame(train_results)
print("\n--- Training Set Border Irregularity Complete ---")
print(train_df.head(10))

