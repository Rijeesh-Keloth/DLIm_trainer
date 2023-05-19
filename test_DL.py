import cv2
import numpy as np
import tensorflow as tf

# Load and preprocess the test image
image_path = '../../Desktop/04.jpg'
image = cv2.imread(image_path)
# Define the dimensions of your input images
image_height = 128
image_width = 128

# Define the number of channels in your images (e.g., 3 for RGB)
num_channels = 3

# Define the number of classes in your dataset
num_classes = 10
model = tf.keras.models.load_model('face_classifier_model.h5')
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed
image = cv2.resize(image, (image_height, image_width))  # Resize the image if needed
#image = image / 255.0  # Normalize pixel values between 0 and 1

# Reshape the image to match the model's input shape
test_image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make predictions using the loaded model
predictions = model.predict(test_image)

# Interpret the predictions (e.g., get the predicted class)
predicted_class = np.argmax(predictions)

# Print the predicted class
print("Predicted class:", predicted_class)

