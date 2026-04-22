#Function that cleans the images (removes hair strands, pen marks, then resizes and normalizes)

from skimage.io import imread
from skimage.transform import resize
from skimage import morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt
from split_data_in_3sets import X_test, X_train


def preprocess_img(path, size=(224, 224)):
    
    def load_img(p):
        return imread(p)
    

    def remove_artifacts(img):
        img_uint8 = img.copy()

        if img_uint8.dtype == np.float32 or img_uint8.dtype == np.float64:
          img_uint8 = (img_uint8 * 255).astype(np.uint8)


        if img_uint8.shape[2] == 4:
          img_uint8 = img_uint8[:, :, :3]
        
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, 8, 255, cv2.THRESH_BINARY)   #Why: gray hair = low contrast → needs lower threshold
        
        disk_brush = morphology.disk(3)
        
        hair_mask = morphology.binary_opening(hair_mask, disk_brush)
        hair_mask = morphology.binary_closing(hair_mask, disk_brush)
        hair_mask = morphology.binary_dilation(hair_mask, disk_brush)
        
        hair_mask = (hair_mask * 255).astype(np.uint8)
        kernel_clean = np.ones((3,3), np.uint8)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel_clean)
        
        
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        
        blue_mask = morphology.binary_opening(blue_mask > 0, disk_brush)
        blue_mask = morphology.binary_closing(blue_mask, disk_brush)
        blue_mask = (blue_mask * 255).astype(np.uint8)
        
        
        combined_mask = cv2.bitwise_or(hair_mask, blue_mask)
        disk_brush = morphology.disk(2)
        combined_mask = morphology.binary_dilation(combined_mask > 0, disk_brush)
        combined_mask = (combined_mask * 255).astype(np.uint8)
        combined_mask[combined_mask > 0] = 255

        print("IMG:", img_uint8.dtype, img_uint8.shape)
        print("MASK:", combined_mask.dtype, combined_mask.shape)
        
        cleaned = cv2.inpaint(img_uint8, combined_mask, 3, cv2.INPAINT_TELEA)
        cleaned = cleaned.astype(np.float32) / 255.0
        return cleaned, combined_mask
    
    def resize_image(img):
        return resize(img, size)
    
    def normalize_image(img):
        return img.astype(np.float32) / 255.0
    
    
    img = load_img(path)
    img_cleaned_float, mask = remove_artifacts(img)
    img = resize_image(img_cleaned_float)
    
    
    return img, mask
    


#to see if it worked############
img_clean, combined_mask = preprocess_img(X_train[55])
original = imread(X_train[55])

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(combined_mask, cmap='gray')
plt.title("Detected Artifacts")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img_clean)
plt.title("Cleaned Image")
plt.axis('off')
plt.show()