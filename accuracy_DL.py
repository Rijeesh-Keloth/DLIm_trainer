import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# Train the model and obtain the history object
num_epochs = 2
batch_size = 2
model = tf.keras.models.load_model('face_classifier_model.h5')
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size)

# Plot the accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

