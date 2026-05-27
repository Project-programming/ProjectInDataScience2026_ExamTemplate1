#for feature_hair:

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.io import imread
from clean_imgs_extenB import detect_hair

def hair_coverage(path):
    img = imread(path)
    img_u8 = img.copy()
    if img_u8.dtype in (np.float32, np.float64):
        img_u8 = (np.clip(img_u8, 0, 1) * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    elif img_u8.shape[2] == 4:
        img_u8 = img_u8[:, :, :3]

    hair_mask = detect_hair(img_u8)
    coverage = np.sum(hair_mask > 0) / hair_mask.size
    return float(coverage)


if __name__ == "__main__":
    img_path = X_train[20]
    print("Image:", os.path.basename(img_path))

    val = hair_coverage(img_path)
    print(f"Hair coverage: {val:.4f}")

    # visual check — show original + hair mask side by side
    img = imread(img_path)
    img_u8 = img.copy()
    if img_u8.dtype in (np.float32, np.float64):
        img_u8 = (np.clip(img_u8, 0, 1) * 255).astype(np.uint8)

    hair_mask = detect_hair(img_u8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_u8)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(hair_mask, cmap="gray")
    axes[1].set_title(f"Hair Mask (coverage = {val:.4f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()