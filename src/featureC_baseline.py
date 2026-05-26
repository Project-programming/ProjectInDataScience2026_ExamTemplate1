#color complexity feature (Measures color variation: Uniform → benign, Many colors → melanoma)
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from scipy.spatial.distance import cdist
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


def color_complexity(path, n_segments=200, compactness=10, visualize=False):

    img, _ = preprocess_img(path)
    
    # Ensure image is in 0-255 range for comparison with Kasmi palette
    img_f = img.astype(float)
    if img_f.max() <= 1.0:
        img_f = img_f * 255.0

    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
    n_sp = segments.max() + 1

    sp_means = np.zeros((n_sp, 3))
    for sid in range(n_sp):
        sp_means[sid] = img_f[segments == sid].mean(axis=0)


    dists     = cdist(sp_means, COLOR_MATRIX)
    nearest   = dists.argmin(axis=1)
    min_dists = dists.min(axis=1)

    fracs       = np.bincount(nearest, minlength=6) / n_sp
    n_colors    = float((fracs > 0).sum())
    p           = fracs[fracs > 0]
    entropy     = float(-np.sum(p * np.log(p + 1e-12)))
    off_palette = float(min_dists.mean())

    features = list(fracs) + [n_colors, entropy, off_palette]
    print("Full output list:", [round(float(x), 4) for x in features])


    if visualize:
        #this part only runs in this file 
        # Panel 1: original cleaned image
        # Panel 2: each superpixel filled with its average color
        # Panel 3: bar chart showing fraction of each skin color
        overlay = label2rgb(segments, img, kind='avg', bg_label=-1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(img)
        axes[0].set_title("Cleaned Image")
        axes[0].axis('off')

        axes[1].imshow(overlay)
        axes[1].set_title(f"SLIC Superpixels (n≈{n_segments})")
        axes[1].axis('off')

        axes[2].bar(COLOR_NAMES, fracs,
                    color=["#C5BCF9","#761511","#A35210",
                           "#872C05","#716C8B","#291F1E"])
        axes[2].set_ylabel("Fraction of superpixels")
        axes[2].set_title("Skin-color distribution")
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


# This part only runs in this file, not when you import the function somewhere
if __name__ == "__main__":
    features = color_complexity(X_train[66], visualize=True)
    print("\nFull output list:", features)
    print("Number of features:", len(features))