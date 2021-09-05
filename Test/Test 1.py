# Importing the necessary libraries
import os
import cv2
import glob as glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime

# Create new folder for processed images if it does not exist    
test_images = "Enter the path where you want to store processed test data"
if not os.path.exists(test_images):
    os.makedirs(test_images)
    
# Load data
filepath = "Enter the path where the test data is stored"
orig_files = [file for file in glob.glob(filepath+"/*.png")]
new_files = [os.path.join(test_images, os.path.basename(f)) for f in orig_files]

# Performing Gaussian blur pre-processing on the input dataset
for orig_f,new_f in zip(orig_files,new_files):
    img = cv2.imread(orig_f)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 40),-4, 128)
    img = cv2.resize(img, (700,700), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(new_f, img)

# Loading the test images into the workspace
train_folder = 'Enter the path of the pre-processed test data/'
files_in_train = sorted(os.listdir(train_folder))

# Creating a CSV file of the test images
images = [i for i in files_in_train]
img = pd.DataFrame({'id_code':images})
img['id_code'] = img['id_code'].map(lambda x: str(x)[:-4])
img.to_csv('Enter the path to store the CSV file/test.csv', index=False)



# Mapping images in the test folder to the CSV file
test = pd.read_csv('Enter the path where CSV file is stored/test.csv')
test["id_code"] = test["id_code"].apply(lambda x: x + ".png")

# Image dimensions
HEIGHT = 700
WIDTH = 700

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test,
                                                  directory = "Enter the path of the pre-processed test data/",
                                                  x_col="id_code",
                                                  target_size=(HEIGHT, WIDTH),
                                                  batch_size=1,
                                                  shuffle=False,
                                                  class_mode=None)

# Loading the trained model into the workspace
model = load_model('Enter the path where trained model is stored/ResNet101.h5') 

# Performing predictions on the test data
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
preds = model.predict_generator(test_generator, steps=STEP_SIZE_TEST)
predictions = [np.argmax(pred) for pred in preds]

# Outputting the predictions to a CSV file
filenames = test_generator.filenames
results = pd.DataFrame({'id_code':filenames, 'diagnosis':predictions})
results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])
results.to_csv(datetime.now().strftime('Enter path where you want to store the Results CSV file/Results-%Y-%m-%d-%H-%M.csv'),index=False)

# Plotting the data distribution of the predictions
f, ax = plt.subplots(figsize=(8, 8))
ax = sns.countplot(x="diagnosis", data=results)
ax.bar_label(ax.containers[0])
sns.despine()
plt.show()
