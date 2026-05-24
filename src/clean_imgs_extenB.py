from skimage.io import imread
from skimage.transform import resize
from skimage import morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt



def detect_hair(img_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    gray_smooth = cv2.bilateralFilter(gray, 5, 20, 20)

    def blackhat_thresh(gray_img, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        bh = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
        otsu_val, _ = cv2.threshold(bh, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        thr = max(5, int(otsu_val * 0.4))
        _, mask = cv2.threshold(bh, thr, 255, cv2.THRESH_BINARY)
        return mask

    
    mask_thin = blackhat_thresh(gray_smooth,  9)
    mask_medium = blackhat_thresh(gray_smooth, 15)
    mask_thick = blackhat_thresh(gray_smooth, 23)
    hair_mask = cv2.bitwise_or(mask_thin,
                  cv2.bitwise_or(mask_medium, mask_thick))

    # Morphological cleanup
    d2, d3 = morphology.disk(2), morphology.disk(3)
    hair_mask = morphology.binary_opening(hair_mask > 0, d2)
    hair_mask = morphology.binary_closing(hair_mask, d3)
    hair_mask = (hair_mask * 255).astype(np.uint8)


    img_area = hair_mask.size
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask, 8)
    filtered = np.zeros_like(hair_mask)
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)
        if area < img_area * 0.02 and aspect > 2.0:
            filtered[labels == lbl] = 255

    filtered = morphology.binary_dilation(filtered > 0, d2)
    return (filtered * 255).astype(np.uint8)



def detect_pen_marks(img_uint8: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

    pen_hsv = cv2.inRange(
        hsv,
        np.array([ 80,   0,  15]),   
        np.array([185, 255, 220]),
    )
    b_star = lab[:, :, 2].astype(np.int16)
    a_star = lab[:, :, 1].astype(np.int16)
    lab_confirm = ((b_star < 130) & (a_star < 145)).astype(np.uint8) * 255
    path_a = cv2.bitwise_and(pen_hsv, lab_confirm)

    
    val = hsv[:, :, 2]
    path_b = (
        (b_star < 122) &
        (a_star < 135) &
        (val.astype(np.int16) > 15) &
        (val.astype(np.int16) < 230)
    ).astype(np.uint8) * 255

    pen_mask = cv2.bitwise_or(path_a, path_b)
    pen_mask[val < 15] = 0  

    # Morphological cleanup
    d1, d3, d5 = morphology.disk(1), morphology.disk(3), morphology.disk(5)
    pen_mask = morphology.binary_opening(pen_mask > 0, d1)
    pen_mask = morphology.binary_closing(pen_mask, d5)
    pen_mask = morphology.binary_dilation(pen_mask, d3)
    pen_mask = (pen_mask * 255).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(pen_mask, 8)
    filtered = np.zeros_like(pen_mask)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= 8:
            filtered[labels == lbl] = 255

    return filtered



def inpaint_artifacts(img_uint8: np.ndarray,
                      hair_mask: np.ndarray,
                      pen_mask:  np.ndarray) -> np.ndarray:
    combined = cv2.bitwise_or(hair_mask, pen_mask)

    combined = morphology.binary_dilation(combined > 0, morphology.disk(3))
    combined = (combined * 255).astype(np.uint8)
    radius = max(5, int(min(img_uint8.shape[:2]) * 0.015))

    cleaned = cv2.inpaint(img_uint8, combined, radius, cv2.INPAINT_TELEA)
    return cleaned.astype(np.float32) / 255.0, combined



def preprocess_img(path: str, size: tuple = (224, 224)):
    img = imread(path)
    img_u8 = img.copy()
    if img_u8.dtype in (np.float32, np.float64):
        img_u8 = (np.clip(img_u8, 0, 1) * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    elif img_u8.shape[2] == 4:
        img_u8 = img_u8[:, :, :3]

    hair_mask = detect_hair(img_u8)
    pen_mask = detect_pen_marks(img_u8)
    img_clean, combined = inpaint_artifacts(img_u8, hair_mask, pen_mask)
    img_resized = resize(img_clean, size, anti_aliasing=True)

    return img_resized, hair_mask, pen_mask, combined

#this function was used for debugging
def show_preprocessing_debug(path: str):
    original                              = imread(path)
    img_clean, hair_mask, pen_mask, comb = preprocess_img(path)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, img, title, cmap in zip(
        axes,
        [original, hair_mask, pen_mask, comb, img_clean],
        ["Original", "Hair mask", "Pen mark mask", "Combined mask", "Cleaned image"],
        [None, "gray", "gray", "gray", None],
    ):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
