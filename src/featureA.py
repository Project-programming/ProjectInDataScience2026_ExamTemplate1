import numpy as np
import os
import pandas as pd
from skimage.io import imread

from split_data_in_3sets import X_train, y_train 

def asymmetry(mask):
    # make sure binary mask is not empty
    mask = (mask > 0).astype(np.uint8)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0.0

    row_mid = int(np.mean(coords[:, 0]))
    col_mid = int(np.mean(coords[:, 1]))

    # Horizontal
    upper = mask[:row_mid, :]
    lower = mask[row_mid:, :]
    lower_flipped = np.flip(lower, axis=0)
    min_rows = min(upper.shape[0], lower_flipped.shape[0])
    hori_xor = np.logical_xor(upper[-min_rows:, :], lower_flipped[:min_rows, :])

    # Vertical
    left = mask[:, :col_mid]
    right = mask[:, col_mid:]


    right_flipped = np.flip(right, axis=1)
    min_cols = min(left.shape[1], right_flipped.shape[1])
    vert_xor = np.logical_xor(left[:, -min_cols:], right_flipped[:, :min_cols])

    total_pixels = np.sum(mask)
    score = (np.sum(hori_xor) + np.sum(vert_xor)) / (2 * total_pixels)
    return round(float(score), 4)

# getting masks
mask_dir = "data/masks/"
train_results = []


# looping through all training data
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
        score = asymmetry(mask_img)
        
        train_results.append({
            'img_id': file_id,
            'asymmetry_score': score,
            'is_cancer': label
        })
    else:
        # It's helpful to know if a mask is missing
        print(f"Skipping: {mask_name} not found.")

# final
train_df = pd.DataFrame(train_results)
print("\n--- Training Set Asymmetry Complete ---")
#to see it worked
print(train_df.head(10))
