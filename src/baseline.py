import numpy as np
import os
import pandas as pd
from skimage.io import imread
from skimage import measure
# Import training data specifically
from split_data_in_3sets import X_train, y_train, X_val, y_val
from clean_the_imgs import preprocess_img
from sklearn.preprocessing import StandardScaler


#asymmetry partt

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
mask_dir = "/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data/masks/"
train_results = []

#to make sure dataset is correct length 
print(f"the length of the training data is: {len(X_train)}")


#borderrr irregularity

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


#colour complexityyy


#color complexity feature (Measures color variation: Uniform → benign, Many colors → melanoma)
def color_complexity_B(path):
   
    img, _ = preprocess_img(path)  
    pixels = img.reshape(-1, 3)  #makes list, Now each row = one pixel
    std_per_channel = np.std(pixels, axis=0)  #[std_R, std_G, std_B] Measures how much colors vary
    color_complexity = np.mean(std_per_channel)   #one nr (if big then melanoma likely)
    return color_complexity
 
img, _ = preprocess_img(X_train[66])
pixels = img.reshape(-1, 3)
value = color_complexity_B(X_train[66])
print(value)

 
"""
getting the plots
plt.figure(figsize=(10,4))
 
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Cleaned Image")
plt.axis('off')
 
plt.subplot(1,2,2)
plt.hist(pixels[:,0], bins=50, alpha=0.5, label='R')
plt.hist(pixels[:,1], bins=50, alpha=0.5, label='G')
plt.hist(pixels[:,2], bins=50, alpha=0.5, label='B')
plt.title("Color Distribution")
plt.legend()
plt.show()
 
"""


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
        bvalue = border_irregularity(mask_img)
        value = color_complexity_B(img_path)

        train_results.append({
            'img_id': file_id,
            'asymmetry_score': score,
            'border_irregularity': bvalue,
            'colour_complexity': value,
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

# Save to CSV so you don't have to run it again
train_df.to_csv("features_train.csv", index=False)

# looping through all validation data!!!!


validation_results = []

for i in range(len(X_val)):
    img_path = X_val[i]
    label = y_val[i]

    file_id = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f"{file_id}_mask.png")

    if not os.path.exists(mask_path):
        continue

    mask_img = imread(mask_path, as_gray=True)

    validation_results.append({
        "img_id": file_id,
        "asymmetry_score": asymmetry(mask_img),
        "border_irregularity": border_irregularity(mask_img),
        "colour_complexity": color_complexity_B(img_path),
        "is_cancer": label
    })

print("almost there")

validation_df = pd.DataFrame(validation_results)
print("\n--- Training Set Asymmetry Complete ---")
#to see it worked
print(validation_df.head(10))

# Save to CSV so you don't have to run it again
validation_df.to_csv("features_validation.csv", index=False)
print("done")
