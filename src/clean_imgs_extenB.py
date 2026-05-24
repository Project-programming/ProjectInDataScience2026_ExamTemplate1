from skimage.io import imread
from skimage.transform import resize
from skimage import morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt

#1 Load dataset and split into train, validation, test

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
import matplotlib.pyplot as plt


#first we add a new colloumn
ff = pd.read_csv(r"E:\Documents\ProjectInDataScience2026_Exercises\data\metadata_with_group.csv")
df = ff[ (ff["group_id"]== "G") | (ff["group_id"]=="K") | (ff["group_id"]=="E") ].copy()
cancerous_diagnostics = ['BCC', 'MEL', 'SCC'] 
df['cancer'] = df['diagnostic'].isin(cancerous_diagnostics).astype(int) #1 if the diagnostic is in the list, and 0 otherwise
df.head(5)


#now we split into sets
#to create image paths (adjust folder if needed)
df["path"] = "E:\projects\ProjectInDataScience2026_ExamTemplate1\data\imgs\\" + df["img_id"] 

# Extract X and y
X = df["path"].values
y = df["cancer"].values

# Split
X_train, X_teva, y_train, y_teva = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_teva, y_teva,
    test_size=0.50,
    stratify=y_teva,
    random_state=42
)


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — HAIR REMOVAL  (revised)
#
#  Key changes vs previous version:
#  • Lighter bilateral blur (d=5 instead of 9) — preserves fine hair edges
#  • Three kernel sizes (9, 15, 23) instead of two — catches very thin hairs
#  • Otsu scale reduced to 0.4 (was 0.6) — more sensitive on flat images
#  • Minimum Otsu floor reduced to 5 (was 8) — fires on nearly-uniform images
#  • Aspect ratio threshold lowered to 2.0 (was 2.5) — catches shorter strands
# ════════════════════════════════════════════════════════════════════════════

def detect_hair(img_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Lighter blur — preserves fine, low-contrast hair edges
    gray_smooth = cv2.bilateralFilter(gray, 5, 20, 20)

    def blackhat_thresh(gray_img, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        bh = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
        otsu_val, _ = cv2.threshold(bh, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Lower scale (0.4) and floor (5) → fires on nearly-uniform images
        thr = max(5, int(otsu_val * 0.4))
        _, mask = cv2.threshold(bh, thr, 255, cv2.THRESH_BINARY)
        return mask

    # Three scales: thin/light, medium, thick/dark hairs
    mask_thin   = blackhat_thresh(gray_smooth,  9)
    mask_medium = blackhat_thresh(gray_smooth, 15)
    mask_thick  = blackhat_thresh(gray_smooth, 23)
    hair_mask   = cv2.bitwise_or(mask_thin,
                  cv2.bitwise_or(mask_medium, mask_thick))

    # Morphological cleanup
    d2, d3 = morphology.disk(2), morphology.disk(3)
    hair_mask = morphology.binary_opening(hair_mask > 0, d2)
    hair_mask = morphology.binary_closing(hair_mask, d3)
    hair_mask = (hair_mask * 255).astype(np.uint8)

    # Keep elongated blobs only — lower aspect threshold catches shorter strands
    img_area = hair_mask.size
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask, 8)
    filtered = np.zeros_like(hair_mask)
    for lbl in range(1, n):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        w      = stats[lbl, cv2.CC_STAT_WIDTH]
        h      = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)
        if area < img_area * 0.02 and aspect > 2.0:
            filtered[labels == lbl] = 255

    filtered = morphology.binary_dilation(filtered > 0, d2)
    return (filtered * 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — PEN MARK REMOVAL  (revised)
#
#  The pen circle in the image is a very desaturated purple-blue — it barely
#  registers in HSV saturation and is nearly neutral in LAB.
#
#  Key changes vs previous version:
#  • Saturation floor dropped to 0 — catches near-grey desaturated ink
#  • Hue range widened to 80–185 — covers violet and teal-blue shifts
#  • LAB b* threshold raised to 130 and a* to 145 — less strict on neutral ink
#  • Added a second detection path using raw LAB blue-shift alone,
#    for ink so desaturated that HSV hue is unreliable
#  • Both HSV-confirmed and LAB-only paths are OR-ed together
# ════════════════════════════════════════════════════════════════════════════

def detect_pen_marks(img_uint8: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

    # Path A — HSV hue gate (very wide) + LAB blue confirmation
    pen_hsv = cv2.inRange(
        hsv,
        np.array([ 80,   0,  15]),   # sat floor = 0: catches desaturated ink
        np.array([185, 255, 220]),
    )
    b_star = lab[:, :, 2].astype(np.int16)
    a_star = lab[:, :, 1].astype(np.int16)
    lab_confirm = ((b_star < 130) & (a_star < 145)).astype(np.uint8) * 255
    path_a = cv2.bitwise_and(pen_hsv, lab_confirm)

    # Path B — LAB blue-shift alone, for ink too desaturated for HSV hue
    # b* < 122 is solidly blue-shifted; a* < 135 excludes warm tones
    # Value gate (15–230) excludes black shadows and white specular highlights
    val = hsv[:, :, 2]
    path_b = (
        (b_star < 122) &
        (a_star < 135) &
        (val.astype(np.int16) > 15) &
        (val.astype(np.int16) < 230)
    ).astype(np.uint8) * 255

    pen_mask = cv2.bitwise_or(path_a, path_b)
    pen_mask[val < 15] = 0  # final shadow exclusion

    # Morphological cleanup
    d1, d3, d5 = morphology.disk(1), morphology.disk(3), morphology.disk(5)
    pen_mask = morphology.binary_opening(pen_mask > 0, d1)
    pen_mask = morphology.binary_closing(pen_mask, d5)
    pen_mask = morphology.binary_dilation(pen_mask, d3)
    pen_mask = (pen_mask * 255).astype(np.uint8)

    # Remove specks < 8 px
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pen_mask, 8)
    filtered = np.zeros_like(pen_mask)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= 8:
            filtered[labels == lbl] = 255

    return filtered


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 3, PUBLIC API, AND DEBUG PLOT — unchanged from previous version
# ════════════════════════════════════════════════════════════════════════════
def inpaint_artifacts(img_uint8: np.ndarray,
                      hair_mask: np.ndarray,
                      pen_mask:  np.ndarray) -> np.ndarray:
    combined = cv2.bitwise_or(hair_mask, pen_mask)

    # ── Fatten the mask so thin hairs are fully covered ──────────────────────
    # Fine hairs are often only 1-2px wide; the raw mask clips their edges.
    # Dilating by disk(3) ensures the full strand width is masked before
    # inpainting, otherwise the inpainter sees the hair edge as valid context
    # and just copies it back in.
    combined = morphology.binary_dilation(combined > 0, morphology.disk(3))
    combined = (combined * 255).astype(np.uint8)

    # ── Inpaint radius: larger = smoother fill over thin strands ─────────────
    # Original used 1% of image size which can be as small as 2-3px.
    # Fine hairs need at least 5px radius to sample far enough from the strand.
    radius = max(5, int(min(img_uint8.shape[:2]) * 0.015))

    cleaned = cv2.inpaint(img_uint8, combined, radius, cv2.INPAINT_TELEA)
    return cleaned.astype(np.float32) / 255.0, combined

#def inpaint_artifacts(img_uint8, hair_mask, pen_mask):
    #combined = cv2.bitwise_or(hair_mask, pen_mask)
    #radius   = max(3, int(min(img_uint8.shape[:2]) * 0.01))
    #cleaned  = cv2.inpaint(img_uint8, combined, radius, cv2.INPAINT_TELEA)
    #return cleaned.astype(np.float32) / 255.0, combined



def preprocess_img(path: str, size: tuple = (224, 224)):
    img = imread(path)
    img_u8 = img.copy()
    if img_u8.dtype in (np.float32, np.float64):
        img_u8 = (np.clip(img_u8, 0, 1) * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    elif img_u8.shape[2] == 4:
        img_u8 = img_u8[:, :, :3]

    hair_mask              = detect_hair(img_u8)
    pen_mask               = detect_pen_marks(img_u8)
    img_clean, combined    = inpaint_artifacts(img_u8, hair_mask, pen_mask)
    img_resized            = resize(img_clean, size, anti_aliasing=True)

    return img_resized, hair_mask, pen_mask, combined


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


show_preprocessing_debug(X_train[20])

# ── Diagnostic: sample pixel values from the pen circle region ───────────────
# Manually pick a pixel (row, col) that sits ON the pen line in your image
# and print its HSV / LAB values so we can tune thresholds precisely.

sample_path = X_train[20]
img_u8 = imread(sample_path)
if img_u8.dtype != np.uint8:
    img_u8 = (np.clip(img_u8, 0, 1) * 255).astype(np.uint8)

hsv_img = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
lab_img = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)

# Adjust these row/col to point at a pixel on the blue pen circle
for (r, c, label) in [(135, 120, "pen circle"), (100, 100, "skin")]:
    print(f"{label:12s}  RGB={img_u8[r,c]}  "
          f"HSV={hsv_img[r,c]}  LAB={lab_img[r,c]}")