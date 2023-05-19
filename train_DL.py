import numpy as np
import tensorflow as tf
import datetime
# Load and preprocess the data
# Assuming X_train contains the input images and y_train contains the corresponding labels
# Define the file paths for loading X_train and y_train
x_train_file = 'X_train.npy'
y_train_file = 'Y_train.npy'

# Load the saved numpy arrays
X_train = np.load(x_train_file)
Y_train = np.load(y_train_file)

image_height = 128
image_width = 128

# Define the number of channels in your images (e.g., 3 for RGB)
num_channels = 3

# Define the number of classes in your dataset
num_classes = 10
num_epochs = 10
# Normalize pixel values between 0 and 1
X_train = X_train / 255.0

batch_size = 1
# Convert labels to one-hot encoding if necessary
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the deep learning model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=X_train, 
          y=Y_train, 
          epochs=num_epochs, 
          validation_data=(X_train, Y_train), 
          callbacks=[tensorboard_callback], batch_size=batch_size)
# Train the model
#model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

# Save the trained model
model.save('face_classifier_model.h5')

