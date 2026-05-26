#color complexity feature (Measures color variation: Uniform → benign, Many colors → melanoma)
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from scipy.spatial.distance import cdist
from skimage.io import imread
from skimage.transform import resize
from skimage import morphology
from clean_imgs_baseline import preprocess_img
from split_data_in_3sets import X_train

# Predefined skin lesion colors from Kasmi et al.
SKIN_COLORS = {
    "white":       np.array([197, 188, 217]),
    "red":         np.array([118,  21,  17]),
    "light_brown": np.array([163,  82,  16]),
    "dark_brown":  np.array([135,  44,   5]),
    "blue_gray":   np.array([113, 108, 139]),
    "black":       np.array([ 41,  31,  30]),
}

COLOR_NAMES  = list(SKIN_COLORS.keys())
COLOR_MATRIX = np.array(list(SKIN_COLORS.values()), dtype=float)

def color_complexity(img_path, mask_path, n_segments=200, compactness=10, visualize=False):
    """
    Parameters
    ----------
    img_path  : path to the lesion image
    mask_path : path to the binary mask (white = lesion, black = background)
    """
    img, _ , = preprocess_img(img_path)

    
    img_f = img.astype(float)
    if img_f.max() <= 1.0:
        img_f = img_f * 255.0
    img_u8 = img_f.astype(np.uint8)

    
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # take first channel if RGB
    mask = mask > 127          # boolean: True = lesion pixel
    from skimage.transform import resize as sk_resize
    mask = sk_resize(mask.astype(float), img_u8.shape[:2],
                     anti_aliasing=False) > 0.5

    #SLIC superpixels on full image
    segments = slic(img_u8, n_segments=n_segments, compactness=compactness, start_label=0)
    n_sp = segments.max() + 1

    
    sp_means   = []
    valid_sids = []
    for sid in range(n_sp):
        sp_pixels_mask = (segments == sid) & mask   
        if sp_pixels_mask.sum() < 5:                
            continue
        sp_means.append(img_f[sp_pixels_mask].mean(axis=0))
        valid_sids.append(sid)

    sp_means = np.array(sp_means)
    n_valid  = len(sp_means)

    
    dists = cdist(sp_means, COLOR_MATRIX)
    nearest = dists.argmin(axis=1)
    min_dists = dists.min(axis=1)

    fracs = np.bincount(nearest, minlength=6) / n_valid
    n_colors = float((fracs > 0).sum())
    p = fracs[fracs > 0]
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    off_palette = float(min_dists.mean())

    features = [float(x) for x in fracs] + [n_colors, entropy, off_palette]

    if visualize:
        img_masked = img_f.copy()
        img_masked[~mask] = 255   

        overlay = label2rgb(segments, img_u8, kind='avg', bg_label=-1)
        overlay_masked = overlay.copy()
        overlay_masked[~mask] = 255

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(img_masked.astype(np.uint8))
        axes[0].set_title("Lesion Only (masked)")
        axes[0].axis('off')

        axes[1].imshow(overlay_masked.astype(np.uint8))
        axes[1].set_title(f"SLIC Superpixels — lesion only")
        axes[1].axis('off')

        axes[2].bar(COLOR_NAMES, fracs,
                    color=["#C5BCF9","#761511","#A35210",
                           "#872C05","#716C8B","#291F1E"])
        axes[2].set_ylabel("Fraction of superpixels")
        axes[2].set_title("Skin-color distribution (lesion only)")
        axes[2].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

        labels = (
            [f"frac_{c}" for c in COLOR_NAMES]
            + ["n_distinct_colors", "color_entropy", "off_palette_dist"]
        )
        print("\nFeature vector:")
        for name, val in zip(labels, features):
            print(f"  {name:<22} = {val:.4f}")

    return features


def get_mask_path(img_path):
    filename = os.path.basename(img_path)          
    name, ext = os.path.splitext(filename)         
    mask_filename = f"{name}_mask{ext}"            
    return os.path.join("data", "masks", mask_filename)


if __name__ == "__main__":
    img_path  = X_train[10]
    mask_path = get_mask_path(img_path)
    print("Image:", os.path.basename(img_path))
    print("Mask: ", mask_path)
    
    features = color_complexity(img_path, mask_path, n_segments=200, compactness=10, visualize=True)
    print("\nFull output list:", features)
    print("Number of features:", len(features))  
