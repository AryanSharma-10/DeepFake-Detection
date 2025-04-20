import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)  # Resize the image to the target size
    image = image.astype(np.float32) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess using ResNet's preprocessing function
    return image