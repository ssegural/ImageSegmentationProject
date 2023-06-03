# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:30:34 2023

@author: ssegu
"""

import tensorflow as tf
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, MaxPool2D, Concatenate


# Set the image and mask directories
image_directory = Path("your image path")
mask_directory = Path("your mask path")

SIZE = 128
num_images = 570

# Load images and masks in order so they match
image_names = sorted(list(image_directory.glob("*.tiff")))
image_names_subset = image_names[0:num_images]
images = [cv2.imread(str(img), 0) for img in image_names_subset]
image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis=3)

mask_names = sorted(list(mask_directory.glob("*.tiff")))
mask_names_subset = mask_names[0:num_images]
masks = [cv2.imread(str(mask), 0) for mask in mask_names_subset]
mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset, axis=3)

# Normalize images and masks
image_dataset = image_dataset / 255.0
mask_dataset = mask_dataset / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.20, random_state=42)

# Import necessary Keras libraries
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, MaxPool2D, Concatenate
from keras.regularizers import l2
# Define the necessary blocks for the U-Net architecture
def conv_block(input, num_filters):
    # ...
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)
    return x

def encoder_block(input, num_filters):
    # ...
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    # ...
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, n_classes):
    # ...
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512) #Bridge

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape, n_classes=1)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Define image and mask data generators with the desired augmentations
data_gen_args = dict(rotation_range=20,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

def fit_generators(image_datagen, mask_datagen, images, masks, batch_size):
    seed = 1
    image_generator = image_datagen.flow(images, seed=seed, batch_size=batch_size)
    mask_generator = mask_datagen.flow(masks, seed=seed, batch_size=batch_size)
    return zip(image_generator, mask_generator)
def train_and_visualize(model, images, masks, epochs, batch_size, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0
    history_list = []

    for train_index, val_index in kfold.split(images, masks):
        fold += 1
        print(f"Training fold {fold}")

        X_train, y_train = images[train_index], masks[train_index]
        X_val, y_val = images[val_index], masks[val_index]

        # Fit generators to the data
        train_generator = fit_generators(image_datagen, mask_datagen, X_train, y_train, batch_size)
        val_generator = fit_generators(image_datagen, mask_datagen, X_val, y_val, batch_size)

        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size

        history = model.fit(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=val_generator,
                            validation_steps=validation_steps,
                            shuffle=True)

        history_list.append(history)

        # Plot the training and validation loss for each fold
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(loss) + 1)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, 'y', label='Training loss')
        plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
        plt.title(f'Training and validation loss for fold {fold}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot the training and validation accuracy for each fold
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, acc, 'y', label='Training acc')
        plt.plot(epochs_range, val_acc, 'r', label='Validation acc')
        plt.title(f'Training and validation accuracy for fold {fold}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    model_save_path = Path("C:/Users/ssegu/Documents/Uni/BachelorProject/Models/480DataSets_100epochs_augmented_crossval.hdf5")
    model.save(model_save_path)

    return history_list


# Call the modified train_and_visualize function with the entire dataset
train_and_visualize(model, image_dataset, mask_dataset, epochs=100, batch_size=16, n_splits=5)

    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from tensorflow.keras.models import load_model, Model
model = load_model("C:/Users/ssegu/Documents/Uni/BachelorProject/Models/480DataSets_100epochs_augmented_crossval.hdf5")

# Function to compute the intersection over union (IoU) metric
def compute_iou(y_true, y_pred):
    return jaccard_score(y_true.ravel(), y_pred.ravel())

# Function to visualize the original image, ground truth, and predictions
def visualize_results(image, ground_truth, prediction):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original Image")
    
    axs[1].imshow(ground_truth, cmap='gray')
    axs[1].set_title("Ground Truth")
    
    axs[2].imshow(prediction, cmap='gray')
    axs[2].set_title("Prediction")
    
    plt.show()

# Function to visualize discrepancies between ground truth and predictions
def visualize_discrepancies(ground_truth, prediction):
    discrepancies = ground_truth - prediction
    plt.imshow(discrepancies, cmap='gray')
    plt.title("Discrepancies (Ground Truth - Prediction)")
    plt.show()

# Function to visualize the uncertainty of the model
def visualize_uncertainty(uncertainty):
    plt.imshow(uncertainty, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Uncertainty")
    plt.show()

# Function to visualize histograms of the prediction probabilities
def visualize_histogram(predictions):
    plt.hist(predictions.ravel(), bins=50, range=(0, 1))
    plt.xlabel("Prediction Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Probabilities")
    plt.show()

# Visualize the results and metrics
n_visualizations = 3
mean_predictions = []
std_predictions = []

for i in range(n_visualizations):
    image = X_test[i, :, :, 0]
    ground_truth = y_test[i, :, :, 0]
    
    prediction = model.predict(X_test[i:i+1])
    mean_predictions.append(np.mean(prediction))
    std_predictions.append(np.std(prediction))
    
    binary_prediction = np.squeeze(prediction > 0.5)
    
    # Calculate the uncertainty
    uncertainty = 1 - np.abs(prediction * 2 - 1)
    uncertainty = np.squeeze(uncertainty)
    
    iou = compute_iou(ground_truth, binary_prediction)
    
    print(f"Sample {i + 1} - IoU: {iou:.4f}")
    visualize_results(image, ground_truth, binary_prediction)
    #visualize_discrepancies(ground_truth, binary_prediction)
    #visualize_uncertainty(uncertainty)
    #visualize_histogram(model.predict(X_test[i:i+1]))
    
    # Calculate the discrepancies between ground truth and prediction
    discrepancies = ground_truth - binary_prediction
    
    # Visualize the discrepancies
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.title('Testing Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.subplot(133)
    plt.title('Discrepancies (Ground Truth - Prediction)')
    plt.imshow(discrepancies, cmap='jet')
    plt.show()


print("Mean prediction probability: ", np.mean(mean_predictions))
print("Standard deviation of prediction probabilities: ", np.mean(std_predictions))

prediction = model.predict(X_test[5:5+1])

prediction = np.squeeze(prediction)
plt.imshow(prediction, cmap='tab20b')
plt.colorbar(label='Predicted Probability')
plt.title('Predicted Probabilities')
plt.show()



from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns

# Flatten ground truth and predictions
y_true_flat = y_test.ravel()
y_pred_flat = (model.predict(X_test) > 0.5).ravel()

# Calculate precision, recall, and thresholds
precision, recall, thresholds_pr = precision_recall_curve(y_true_flat, y_pred_flat)

# Plot Precision-Recall curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds_roc = roc_curve(y_true_flat, y_pred_flat)
roc_auc = roc_auc_score(y_true_flat, y_pred_flat)

# Plot ROC curve
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(y_true_flat, y_pred_flat)

# Plot confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

iou_scores = [compute_iou(y_test[i, :, :, 0],(model.predict(X_test[i:i+1]))>0.5)]
print(iou_scores)


# Calculate the uncertainty
uncertainty = 1 - np.abs(prediction * 2 - 1)
uncertainty = np.squeeze(uncertainty)

plt.figure(figsize=(6, 6))
plt.imshow(uncertainty, cmap='hot')
plt.colorbar(label='Uncertainty')
plt.title('Uncertainty Map')
plt.show()
#SALIENCY MAP

import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_saliency_map(model, input_image):
    
    #This function computes the saliency map of a given model for a given input image.
    
    # Ensure that the image is float
    input_image = tf.cast(input_image, tf.float32)
    print(input_image.shape)
    # Create a GradientTape to monitor the input image
    with tf.GradientTape() as tape:
        # Set the input image as the variable to be watched by the GradientTape
        tape.watch(input_image)
        # Get the prediction of the model for the input image
        prediction = model(input_image[None, ...])[0]
        print(prediction.shape)
    
    # Get the gradients of the output with respect to the input image
    gradients = tape.gradient(prediction, input_image)
    print(gradients.shape)
    # Compute the saliency map as the maximum absolute value of gradients for each color channel
    saliency_map = tf.math.abs(gradients)
    saliency_map = (saliency_map - tf.math.reduce_min(saliency_map)) / (tf.math.reduce_max(saliency_map) - tf.math.reduce_min(saliency_map))
    print(saliency_map.shape)
    return saliency_map


# Choose an image from the test set
image_index = 60
image = X_test[image_index, :, :, 0]



# Compute the saliency map
saliency_map = get_saliency_map(model, image)


# Plot the original image and the saliency map side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='hot')
plt.title("Saliency Map")
plt.axis('off')

plt.show()

#GRAD-CAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

# Let's assume you already have a preprocessed image ready for the model
# e.g., img_tensor = preprocess_image(your_test_image)

# The name of the last convolutional layer
# In the U-Net model you posted, this is 'conv2d_113'
layer_name = 'conv2d_98'

# Create a Model that will output the feature maps and the output
grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

# Get the score for the maximum activated pixel
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array([image]))
    loss = tf.reduce_max(conv_outputs)

# Extract gradients
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

# Global average pooling
weights = tf.reduce_mean(grads, axis=(0, 1))

# Build a map of filters according to their importance
cam = np.zeros(output.shape[0: 2], dtype=np.float32)
for index, w in enumerate(weights):
    cam += w * output[:, :, index]

# Add dimensions for tf.image.resize
cam = np.expand_dims(cam, axis=0)  # Add the batch dimension
cam = np.expand_dims(cam, axis=-1)  # Add the channel dimension

# Resize cam to original image shape
cam = tf.image.resize(cam, [image.shape[0], image.shape[1]])

# Normalize the CAM
cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam))

# Display Grad-CAM
plt.imshow(cam[0, :, :, 0], cmap='hot')





from keras.models import Model
import matplotlib.pyplot as plt

# Specify the layers you want to visualize
# 95 98 100
layer_names = ['conv2d_95', 'conv2d_98', 'conv2d_100']  # Modify this list to match your model's layer names

outputs = [model.get_layer(name).output for name in layer_names]

# Create a new model that will return these outputs given the model input
feature_map_model = Model(inputs=model.inputs, outputs=outputs)

# Check the shape of the image
print(image.shape)

# If the image is already three-dimensional (height, width, channels), skip this line
if len(image.shape) < 3:
    image = np.expand_dims(image, axis=-1)  # Add the channel dimension

# Add the batch dimension
#image = np.expand_dims(image, axis=0)  

feature_maps = feature_map_model.predict(image)

# Plotting
for layer_name, feature_map in zip(layer_names, feature_maps):
    print(f"Visualizing {layer_name}")
    num_features = feature_map.shape[-1]  # Number of features in the feature map
    size = feature_map.shape[1]  # The feature map has shape (1, size, size, num_features).
    num_cols = int(np.ceil(num_features ** 0.5))  # Round up to the nearest integer
    fig = plt.figure(figsize=(20, 20))
    for i in range(num_features):
        ax = fig.add_subplot(num_cols, num_cols, i+1)
        ax.matshow(feature_map[0, :, :, i], cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()

from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# Specify the layers you want to visualize
# 95 98 100
layer_names = ['conv2d_107', 'conv2d_111', 'conv2d_112']  # Modify this list to match your model's layer names

outputs = [model.get_layer(name).output for name in layer_names]

# Create a new model that will return these outputs given the model input
feature_map_model = Model(inputs=model.inputs, outputs=outputs)

# Check the shape of the image
print(image.shape)

# If the image is already three-dimensional (height, width, channels), skip this line
if len(image.shape) < 3:
    image = np.expand_dims(image, axis=-1)  # Add the channel dimension

# Add the batch dimension
image = np.expand_dims(image, axis=0)  

feature_maps = feature_map_model.predict(image)

# Plotting
for layer_name, feature_map in zip(layer_names, feature_maps):
    print(f"Visualizing {layer_name}")
    num_features = feature_map.shape[-1]  # Number of features in the feature map
    size = feature_map.shape[1]  # The feature map has shape (1, size, size, num_features).
    num_cols = int(np.ceil(num_features ** 0.5))  # Round up to the nearest integer

    # Plot all feature maps
    fig = plt.figure(figsize=(20, 20))
    for i in range(num_features):
        ax = fig.add_subplot(num_cols, num_cols, i+1)
        ax.matshow(feature_map[0, :, :, i], cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()

    # Plot a randomly selected feature map
    fig = plt.figure(figsize=(4, 4))
    random_feature = np.random.choice(num_features)
    plt.matshow(feature_map[0, :, :, random_feature], cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Random feature map from {layer_name}")
    plt.show()



#GRAIN METRICS CALCULATION

def generate_lines(binary_image, num_lines, min_length_ratio=0.5):
    h, w = binary_image.shape[:2]
    min_length = min(h, w) * min_length_ratio

    line_images = []
    line_lengths = []
    for i in range(num_lines):
        # Randomly select a start point, angle, and length for the line
        x0 = np.random.randint(0, w)
        y0 = np.random.randint(0, h)
        angle = np.random.uniform(0, 180)  # Angle in degrees
        length = np.random.uniform(min_length, np.sqrt(h**2 + w**2))

        # Calculate the end point of the line
        x1 = x0 + int(length * np.cos(np.deg2rad(angle)))
        y1 = y0 + int(length * np.sin(np.deg2rad(angle)))

        # Draw the line on the image
        # Ensure the image is of type uint8 for cv2.line
        line_image = cv2.line(np.zeros_like(binary_image, dtype=np.uint8), (x0, y0), (x1, y1), (255), 1)

        line_images.append(line_image)
        line_lengths.append(length)

    return line_images, line_lengths
def calculate_grain_size(binary_image, num_lines=10):
    # Convert binary_image to uint8
    binary_image = (binary_image * 255).astype(np.uint8)

    # Generate multiple lines across the image
    line_images, line_lengths = generate_lines(binary_image, num_lines)

    # Create an image to store all the lines
    combined_lines = np.zeros_like(binary_image)

    grain_sizes = []
    for line_image, line_length in zip(line_images, line_lengths):
        # Find where the line intersects the grains
        intersections = cv2.bitwise_and(binary_image, binary_image, mask=line_image)

        # Count the number of intersections
        num_intersections = np.count_nonzero(intersections)

        # Calculate the average grain size according to the linear intercept method
        grain_size = line_length / num_intersections if num_intersections else np.nan

        grain_sizes.append(grain_size)

        # Add the current line to the combined image
        combined_lines = cv2.bitwise_or(combined_lines, line_image)

    # Return the average grain size across all lines and the combined image
    return np.nanmean(grain_sizes), combined_lines

def visualize_grain_size(binary_image, combined_lines):
    # Plot the original binary image and the image with intersections
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(binary_image, cmap='gray')
    ax[0].set_title('Original Binary Image')

    # Overlay the binary image with combined lines
    ax[1].imshow(binary_image, cmap='gray')
    ax[1].imshow(combined_lines, cmap='jet', alpha=0.5)  # Use a different colormap for lines and set transparency
    ax[1].set_title('Intersections with Lines')

    plt.show()
from skimage import measure
from skimage.measure import label, regionprops

def calculate_area(binary_image):
    labeled_image, num_grains = label(binary_image, return_num=True)
    
    areas = []
    for region in regionprops(labeled_image):
        areas.append(region.area)
    
    average_area = np.mean(areas) if areas else np.nan
    return average_area, areas

average_grain_size, combined_lines = calculate_grain_size(binary_prediction)

print(f"Average Grain Size: {average_grain_size}")

visualize_grain_size(binary_prediction, combined_lines)

grain_volumes = calculate_area(binary_prediction)

print(f"Average Area distribution: {grain_volumes}")

from skimage.measure import label
import matplotlib.pyplot as plt

def count_grains(binary_image):
    # Label connected regions of the binary image
    labeled_image, num_grains = label(binary_image, return_num=True)
    return num_grains, labeled_image

def plot_area_distribution(areas, bins=10):
    plt.hist(areas, bins=bins)
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.title('Area Distribution')
    plt.show()

# Use the functions
num_grains, labeled_image = count_grains(binary_prediction)
print(f"Number of grains: {num_grains}")

average_area, areas = calculate_area(binary_prediction)
plot_area_distribution(areas)



