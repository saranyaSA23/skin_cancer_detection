# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Load the data and metadata
metadata_path = r'E:\skin_cancer_detection\HAM10000_metadata.csv'
image_folder = r'E:\skin_cancer_detection\HAM10000_images_part_1'

data = pd.read_csv(metadata_path)  # Load the metadata CSV
images = []  # List to store image data
labels = []  # List to store labels

# Loop through each image file and preprocess it
for _, row in data.iterrows():
    img_path = os.path.join(image_folder, f"{row['image_id']}.jpg")  # Adjust to your image path
    if os.path.exists(img_path):  # Check if image exists
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))  # Resize image
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize image
        images.append(img)
        labels.append(row['dx'])  # 'dx' is the label column (disease type)

# Convert lists to numpy arrays
images = np.array(images)
labels = pd.get_dummies(labels).values  # One-hot encode the labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels[0]), activation='softmax')  # Output layer based on number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, 
                             width_shift_range=0.2, height_shift_range=0.2, 
                             horizontal_flip=True)
datagen.fit(X_train)

# Train the model and capture training history
try:
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=10, validation_data=(X_test, y_test))
except Exception as e:
    print(f"An error occurred during model training: {e}")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()
