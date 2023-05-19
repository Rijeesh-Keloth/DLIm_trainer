import cv2
import numpy as np

# Define the path to your dataset
dataset_path = '/*/*/*/'

# Define the dimensions of your input images
image_height = 128
image_width = 128

# Define the number of channels in your images (e.g., 3 for RGB)
num_channels = 3

# Define the number of classes in your dataset
num_classes = 10
num_images_per_class = 200
# Initialize empty lists to store the data
X_train = []
y_train = []

# Loop through the dataset and load the images
for class_index in range(num_classes):
#    class_path = dataset_path + str(class_index) + '/'
    class_path = dataset_path 
    # Loop through the images in the current class
    for image_index in range(num_images_per_class):
#        image_path = class_path + str(image_index) + '.jpg'
        print(class_path)
        image_path = "../../Desktop/04.jpg"   
        # Read the image and resize it to the desired dimensions
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_height, image_width))
        
        # Append the image to X_train
        X_train.append(image)
        
        # Append the label to y_train
        y_train.append(class_index)

# Convert X_train and y_train to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Define the file paths for saving X_train and y_train
x_train_file = 'X_train.npy'
y_train_file = 'Y_train.npy'

# Save X_train and y_train as numpy arrays
np.save(x_train_file, X_train)
np.save(y_train_file, y_train)
