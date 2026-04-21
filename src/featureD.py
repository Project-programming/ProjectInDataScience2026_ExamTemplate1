import numpy as np
import os
import pandas as pd
from skimage.io import imread
from scipy.spatial.distance import cdist
from split_data_in_3sets import X_train, y_train

def diameter(mask, mm_per_pixel=None):
    mask = (mask > 0).astype(np.uint8)
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return 0.0

    # subsample if lesion is large
    if len(coords) > 500:
        step = max(1, len(coords) // 500)
        coords_sample = coords[::step]
    else:
        coords_sample = coords

    # farthest pair of lesion pixels = true diameter
    dists = cdist(coords_sample, coords_sample, metric='euclidean')
    max_diameter_px = float(np.max(dists))

    if mm_per_pixel is not None:
        return round(max_diameter_px * mm_per_pixel, 4)

    return round(max_diameter_px, 4)


# getting masks
mask_dir = r"data/masks"
train_results = []

print(f"Starting calculation for {len(X_train)} training samples...")

# double check dataset length
print(f"the length of the training data is: {len(X_train)}")

# loop through all training data
for i in range(len(X_train)):
    img_path = X_train[i]
    label = y_train[i]

    # extract ID and find mask
    base_name = os.path.basename(img_path)
    file_id = os.path.splitext(base_name)[0]
    mask_name = f"{file_id}_mask.png"
    full_mask_path = os.path.join(mask_dir, mask_name)

    if os.path.exists(full_mask_path):
        mask_img = imread(full_mask_path, as_gray=True)
        score = diameter(mask_img)

        train_results.append({
            'img_id': file_id,
            'diameter_px': score,
            'is_cancer': label
        })
    else:
        print(f"Skipping: {mask_name} not found.")

# final
train_df = pd.DataFrame(train_results)
print("\n--- Training Set Diameter Complete ---")
# sanity check
print(train_df.head(10))

# save to CSV so you don't have to run it again
# train_df.to_csv("train_diameter_features.csv", index=False)