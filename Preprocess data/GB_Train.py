import os
import cv2
import glob as glob

# Create new folder for storing processed train images if it does not exist    
test_images = "Enter the path where you want to store processed train data"
if not os.path.exists(test_images):
    os.makedirs(test_images)
    
# Load data into workspace
filepath = "Enter the path where the train data is stored"
orig_files = [file for file in glob.glob(filepath+"/*.png")]
new_files = [os.path.join(test_images, os.path.basename(f)) for f in orig_files]

for orig_f,new_f in zip(orig_files,new_files):
    img = cv2.imread(orig_f)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 40),-4, 128)
    img = cv2.resize(img, (700,700), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(new_f, img)