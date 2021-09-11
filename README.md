# Automatic-Detection-and-Grade-Classification-of-Diabetic-Retinopathy-using-Deep-CNN

Diabetic Retinopathy is a complication of diabetes, caused by high blood sugar levels which results in damage to the back of the eye or the retina. Diabetic Retinopathy is the leading cause of blindness in the working age population of the world. An estimated 93 million people are affected by this disease. As per the NHS, an estimated 1280 new cases of diabetic retinopathy are recorded each year in England alone. Currently, the methods employed for the detection of diabetic retinopathy are manual and time consuming. The manual method involves a trained clinician evaluating retina fundus images. Although this method is effective, this process is time consuming and delays can be caused due to late submission of retina fundus images, miscommunication resulting in delayed treatment of the patients. Also, the lack of expertise and employing this method in highly populated areas is another major problem with this approach. 

This approach proposes a multi-label classification system using deep CNN architectures which will make predictions using a trained CNN model with a high degree of accuracy. 
Now, I have proposed two different testing approaches in this repository.

1) Approach 1:
- Load the test data stored in the test folder into the workspace.
- Perform Gaussian blur preprocessing on the input data.
- Create a CSV file out of these test images for our flow_from_dataframe approach.
- Load the trained CNN model into the workspace.
- Carry out predictions using the trained model.
- Store the predicted image labels in a CSV file.
- Plot the predicted label distribution using seaborn and matplotlib.

2) Approach 2:
- Load the test data stored in the test folder into the workspace.
- Perform Gaussian Blur preprocessing on the input data.
- Load the Benchmark.csv file into the workspace(this file will be used as reference to test our model prediction power).
- Load the trained model into the workspace.
- Carry out predictions using the trained model.
- Store the predicted image labels in the Benchmark file.
- Plot the predicted and benchmark label distribution using seaborn and matplotlib.


The dataset used for this experiment is sourced from Kaggle(https://www.kaggle.com/c/aptos2019-blindness-detection/overview/description).
It contains 3662 images for training and 1928 images for testing.
The 3662 images have been split up into 2930 train images and 732 validation images in a 80:20 train validation split.
Also, I would like to extend a Big Thank You to dimitreOliveira for his amazing Kaggle kernel(https://www.kaggle.com/dimitreoliveira/aptos-blindness-detection-eda-and-keras-resnet50).

Further, to make the project End to End(E2E), using Flask, I have created a web application where the user simply has to upload the retina fundus image to obtain the predicted label. To run the Flask application, navigate to the directory where the app.py file has been stored through the command prompt and execute 'python app.py' command. This will create a localhost URL to run the web application.

Since the localhost URL can only run on a single machine, I have utilized Ngrok to create a secure tunnel to the localhost. 
Installing and running Ngrok is pretty straightforward and can be download from here: https://ngrok.com/download

Once installed, simply run the command: 'ngrok http 5000' (The port number is 5000 as the web app is running on that port).


