import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# Obtain the trained scores

# Define the dimensions of your input images
image_height = 128
image_width = 128

# Define the number of channels in your images (e.g., 3 for RGB)
num_channels = 3

# Define the number of classes in your dataset
num_classes = 10

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

X_train = np.load('X_train.npy')
scores = model.predict(X_train)  # Replace X_train with your training data
print(scores.shape)

num_scores = scores.shape[1]  # Get the number of scores

for score_index in range(num_scores):
    plt.plot(scores[:, score_index], label=f'Score {score_index+1}')

# Plot the scores
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(scores[:, 0], label='Score 1')
    plt.plot(scores[:, 1], label='Score 2')
# Add more lines for additional scores

    plt.title('Trained Scores')
    plt.xlabel('Sample')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
