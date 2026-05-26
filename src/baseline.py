import numpy as np
import os
import pandas as pd
from skimage.io import imread
from skimage import measure
# Import training data specifically
from split_data_in_3sets import X_train, y_train, X_val, y_val, X_test, y_test
from clean_imgs_baseline import preprocess_img
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from featureC_baseline import color_complexity
from featureA_baseline import asymmetry
from featureB_baseline import border_irregularity




# getting masks
mask_dir = r"/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/data/masks"
train_results = []

#to make sure dataset is correct length 
print(f"the length of the training data is: {len(X_train)}")


#borderrr irregularity



#color complexity feature (Measures color variation: Uniform → benign, Many colors → melanoma)

 
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
        value = color_complexity(img_path)

        
        cc = color_complexity(img_path)   # returns list of 9 values

        train_results.append({
            'img_id':            file_id,
            'asymmetry_score':   asymmetry(mask_img),
            'border_irregularity': border_irregularity(mask_img),
            # 6 colour fractions
            'frac_white':        cc[0],
            'frac_red':          cc[1],
            'frac_light_brown':  cc[2],
            'frac_dark_brown':   cc[3],
            'frac_blue_gray':    cc[4],
            'frac_black':        cc[5],
            # 3 summary colour stats
            'n_distinct_colors': cc[6],
            'color_entropy':     cc[7],
            'off_palette_dist':  cc[8],
            'is_cancer':         label
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
        "colour_complexity": color_complexity(img_path),
        "is_cancer": label
    })

#print("almost there")

validation_df = pd.DataFrame(validation_results)
print("\n--- Training Set Asymmetry Complete ---")
#to see it worked
print(validation_df.head(10))

# Save to CSV so you don't have to run it again
validation_df.to_csv("features_validation.csv", index=False)
print("done")


#BASELINE MODEL TRAINING
feature_cols = ['asymmetry_score', 'border_irregularity', 'colour_complexity']
X_train_feat = train_df[feature_cols].values
y_train_feat = train_df['is_cancer'].values
X_val_feat = validation_df[feature_cols].values
y_val_feat = validation_df['is_cancer'].values

#Scaling
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat = scaler.transform(X_val_feat)

#Logistic Regression
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_feat, y_train_feat)

#Evaluation on validation set
y_pred = clf.predict(X_val_feat)
print(f"\nValidation Accuracy: {accuracy_score(y_val_feat, y_pred):.4f}")
print(classification_report(y_val_feat, y_pred, target_names=['Benign', 'Cancer']))


#testing data



testing_results = []

for i in range(len(X_test)):
    img_path = X_test[i]
    label = y_test[i]

    file_id = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f"{file_id}_mask.png")

    if not os.path.exists(mask_path):
        continue

    mask_img = imread(mask_path, as_gray=True)

    train_results.append({
            'img_id':            file_id,
            'asymmetry_score':   asymmetry(mask_img),
            'border_irregularity': border_irregularity(mask_img),
            # 6 colour fractions
            'frac_white':        cc[0],
            'frac_red':          cc[1],
            'frac_light_brown':  cc[2],
            'frac_dark_brown':   cc[3],
            'frac_blue_gray':    cc[4],
            'frac_black':        cc[5],
            # 3 summary colour stats
            'n_distinct_colors': cc[6],
            'color_entropy':     cc[7],
            'off_palette_dist':  cc[8],
            'is_cancer':         label
        })
        


testing_df = pd.DataFrame(testing_results)
print("\n--- testing Set Asymmetry Complete ---")
#to see it worked
print(testing_df.head(10))

# Save to CSV so you don't have to run it again
testing_df.to_csv("features_testing.csv", index=False)
print("done")
