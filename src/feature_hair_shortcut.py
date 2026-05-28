#for feature_hair

from split_data_in_3sets import X_train
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
import os
import numpy as np
from skimage.io import imread


def hair_coverage(path):
    img = imread(path)
    img_u8 = img.copy()
    if img_u8.dtype in (np.float32, np.float64):
        img_u8 = (np.clip(img_u8, 0, 1) * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    elif img_u8.shape[2] == 4:
        img_u8 = img_u8[:, :, :3]

    
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 26, 255, cv2.THRESH_BINARY)

    disk_brush = morphology.disk(3)
    hair_mask = morphology.binary_opening(hair_mask, disk_brush)
    hair_mask = morphology.binary_closing(hair_mask, disk_brush)
    hair_mask = morphology.binary_dilation(hair_mask, disk_brush)
    hair_mask = (hair_mask * 255).astype(np.uint8)

    kernel_clean = np.ones((3, 3), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel_clean)
    

    coverage = np.sum(hair_mask > 0) / hair_mask.size
    return float(coverage), hair_mask   





if __name__ == "__main__":
    img_path = X_train[14]
    print("Image:", os.path.basename(img_path))

    val, hair_mask = hair_coverage(img_path)   # unpack both values now
    print(f"Hair coverage: {val:.4f}")

    img = imread(img_path)
    img_u8 = img.copy()
    if img_u8.dtype in (np.float32, np.float64):
        img_u8 = (np.clip(img_u8, 0, 1) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_u8)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(hair_mask, cmap="gray")
    axes[1].set_title(f"Hair Mask (coverage = {val:.4f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


