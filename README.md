# Image_Classification_using_CNN

This project involves building and training deep learning models to classify images as either a dog or a cat. It utilizes the TensorFlow and Keras libraries to construct Convolutional Neural Networks (CNN) for the task. The dataset is obtained from the Kaggle Dogs vs. Cats competition, showcasing a practical approach to binary image classification problems.

# Installation
## Prerequisites
Ensure you have the following installed:

Python 3.6, 
TensorFlow, 
Keras, 
Pandas, 
Matplotlib, 
Jupyter Notebook or JupyterLab (for running the notebook).

# Setup
Clone this repository or download the code to your local machine.
Install the required Python packages:

pip install tensorflow keras pandas matplotlib jupyter

Set up Kaggle API credentials to download the dataset. Ensure you have a Kaggle account, then create an API token from the Account section of your Kaggle profile. Place the kaggle.json file in your home directory under ~/.kaggle/ (for Unix-based systems) or C:\Users\<Windows-username>\.kaggle\ (for Windows).

# Usage

1. Open the Dogs_vs_Cats_Classification.ipynb notebook in Jupyter Notebook or JupyterLab.

2.Follow the instructions within the notebook to download the dataset using Kaggle API commands.

3.Run the cells in sequence to preprocess the data, build the CNN models, and train them on the dataset.

4.Evaluate the model performance using the provided metrics and visualize the results.

# Project Structure

Data loading and preprocessing
Model building:
Basic CNN model for baseline comparison
Improved CNN model with regularization and data augmentation
Model training and validation
Evaluation and prediction on test data
Visualization of training history and prediction results

# Models
Two CNN models are implemented:

1. Basic CNN Model: A simple architecture to serve as a baseline.
2. Improved CNN Model: Includes additional convolutional layers, regularization techniques like L1/L2 regularization and dropout, and data augmentation to enhance model performance.
   
# Dataset
The dataset consists of 25,000 images of dogs and cats (12,500 from each class) for training and 12,500 unlabeled images for testing. The objective is to predict the labels for the test set images.

# Results
The notebook includes code to visualize training/validation loss and accuracy, as well as predictions on test images. The final section generates predictions for the test dataset and prepares a CSV file for submission to the Kaggle competition.

# Acknowledgements
This project uses the Dogs vs. Cats dataset from Kaggle.

