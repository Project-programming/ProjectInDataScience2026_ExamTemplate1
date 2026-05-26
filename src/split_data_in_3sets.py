## will load dataset and split into train, validation, test

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
import matplotlib.pyplot as plt


ff = pd.read_csv(r"/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/metadata-merged(in).csv")
df = ff[ (ff["group_id"]== "G") | (ff["group_id"]=="K") | (ff["group_id"]=="E") ].copy()
cancerous_diagnostics = ['BCC', 'MEL', 'SCC']
df['cancer'] = df['diagnostic'].isin(cancerous_diagnostics).astype(int)
df.head(5)


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df["path"] = os.path.join(base_dir, "data", "imgs", "") + df["img_id"]


X = df["path"].values
y = df["cancer"].values

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

if __name__ == "__main__":
    img = imread(df["path"].iloc[0])
    plt.imshow(img)
    plt.title("Test image")
    plt.show()