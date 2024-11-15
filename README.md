# Skin Cancer Detection

## Overview

The **Skin Cancer Detection** project is a machine learning-based application designed to assist in the early detection of skin cancer by analyzing images of skin lesions. This system uses convolutional neural networks (CNNs) to classify skin lesions as either benign or malignant.

### Key Features:
- **Image Preprocessing**: Enhances the quality of images for better classification.
- **Deep Learning Model**: Uses CNN to detect skin cancer with high accuracy.
- **User Interface**: A simple web or terminal interface that allows users to upload images and receive predictions.
- **Evaluation Metrics**: Provides performance metrics such as accuracy, precision, recall, and F1-score for model evaluation.

### Technologies:
- **Python**: Primary language for the implementation.
- **TensorFlow/Keras**: Framework for building and training the machine learning model.
- **OpenCV**: Used for image processing.
- **Scikit-learn**: For model evaluation and performance metrics.
- **Flask/Django** *(Optional)*: Web framework for creating a user-friendly interface.

## Dataset

This project uses publicly available skin lesion datasets such as the **ISIC (International Skin Imaging Collaboration)** dataset. The dataset contains thousands of labeled skin lesion images, categorized as benign (non-cancerous) or malignant (cancerous).

## How It Works

1. **Image Upload**: Users upload an image of a skin lesion through the user interface.
2. **Image Preprocessing**: The image undergoes preprocessing to normalize its size and quality for better feature extraction.
3. **Model Prediction**: The image is passed through a trained convolutional neural network (CNN) to predict if the lesion is malignant or benign.
4. **Result**: The model outputs a prediction, including a confidence score indicating the likelihood of malignancy.

## Installation


Python Installation: Download and install Python if you haven't already. During installation, check the box to "Add Python to PATH."

pip install numpy pandas matplotlib scikit-learn tensorflow keras

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000


