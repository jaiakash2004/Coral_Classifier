# prompt: using trained model of h5 test with custom images

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('DVC.h5')

# Function to preprocess and predict the image
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64,64))
    img_array = np.array(img)
    img_array = img_array.reshape(1,64,64, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(img, interpolation='nearest')
    plt.show()
    prediction = model.predict(img_array)
    indices = prediction.argmax()
    print(f"Predicted class index: {indices}")
    print(f"Prediction probabilities: {prediction}")
    return indices

# Example usage with a custom image
image_path = "C:/Users/lenovo/Downloads/Coral-Bleach-2_Sequence-01.00_01_24_05.Still009-scaled.jpg" # Replace with the actual path to your image
predicted_class = predict_image(image_path)