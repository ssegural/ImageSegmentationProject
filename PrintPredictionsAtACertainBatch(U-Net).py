# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:54:02 2023

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
image_directory = Path("C:/Users/ssegu/Documents/Uni/BachelorProject/trainImages")
mask_directory = Path("C:/Users/ssegu/Documents/Uni/BachelorProject/maskTrainImages")

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

# Define the necessary blocks for the U-Net architecture
def conv_block(input, num_filters):
    # ...
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters,3, padding="same")(x)
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
import matplotlib.pyplot as plt

def visualize_results(model, images, masks):
    predictions = model.predict(images)
    num_images = len(images)

    for i in range(num_images):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(images[i])
        axes[0].set_title("Input Image")
        
        axes[1].imshow(masks[i].squeeze(), cmap='gray')
        axes[1].set_title("Ground Truth Mask")

        axes[2].imshow(predictions[i].squeeze(), cmap='gray')
        axes[2].set_title("Predicted Mask")

        plt.show()


from keras.callbacks import Callback

class SaveModelAtSpecificBatch(tf.keras.callbacks.Callback):
    def __init__(self, target_epoch, batch, model_name):
        super(SaveModelAtSpecificBatch, self).__init__()
        self.target_epoch = target_epoch
        self.target_batch = batch
        self.model_name = model_name

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if self.current_epoch == self.target_epoch - 1 and batch == self.target_batch - 1:
            model_path = f"temp_model_{self.model_name}_epoch{self.target_epoch}_batch{self.target_batch}.hdf5"
            self.model.save(model_path)
            print(f"Model saved at {model_path}")
def train_and_visualize(model, image_dataset, mask_dataset, epochs, batch_size, n_splits, save_epoch, save_batch, model_name="model"):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    
    for train_idx, val_idx in kf.split(image_dataset, mask_dataset):
        print(f"Training fold {fold}")
        train_images, val_images = image_dataset[train_idx], image_dataset[val_idx]
        train_masks, val_masks = mask_dataset[train_idx], mask_dataset[val_idx]

        train_generator = fit_generators(image_datagen, mask_datagen, train_images, train_masks, batch_size)
        val_generator = fit_generators(image_datagen, mask_datagen, val_images, val_masks, batch_size)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        save_model_callback = SaveModelAtSpecificBatch(target_epoch=save_epoch, batch=save_batch, model_name=model_name)

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=len(train_images) // batch_size,
            validation_steps=len(val_images) // batch_size,
            callbacks=[save_model_callback]
        )

        # Loading the model saved at a specific batch and epoch
        temp_model_path = Path(f"temp_model_{model_name}_epoch{save_epoch}_batch{save_batch}.hdf5")
        model_at_specific_batch = tf.keras.models.load_model(temp_model_path)

        # Visualize the segmentation results
        visualize_results(model_at_specific_batch, val_images, val_masks)

        # Plot the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        fold += 1

# Call the modified train_and_visualize function with the entire dataset
train_and_visualize(model, image_dataset, mask_dataset, epochs=2, batch_size=16, n_splits=5, save_epoch=1, save_batch=10, model_name="model")

    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

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
n_visualizations = 5
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
    visualize_discrepancies(ground_truth, binary_prediction)
    visualize_uncertainty(uncertainty)
    visualize_histogram(model.predict(X_test[i:i+1]))

print("Mean prediction probability: ", np.mean(mean_predictions))
print("Standard deviation of prediction probabilities: ", np.mean(std_predictions))

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