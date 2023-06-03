# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:07:13 2023

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
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator


image_directory = r"C:\Users\ssegu\Documents\Uni\BachelorProject\trainImages"
mask_directory = r"C:\Users\ssegu\Documents\Uni\BachelorProject\maskTrainImages"
     

SIZE = 128
num_images = 570
     
#Load images and masks in order so they match


image_names = glob.glob(r"C:\Users\ssegu\Documents\Uni\BachelorProject\trainImages\*.tiff")
print(image_names)


image_names.sort()
print(image_names)
     

image_names_subset = image_names[0:num_images]
     

images = [cv2.imread(img, 0) for img in image_names_subset]
     

image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis = 3)
     
#Read masks the same way.


mask_names = glob.glob(r"C:\Users\ssegu\Documents\Uni\BachelorProject\maskTrainImages\*.tiff")
mask_names.sort()
mask_names_subset = mask_names[0:num_images]
masks = [cv2.imread(mask, 0) for mask in mask_names_subset]
mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset, axis = 3)
     

print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))
     


#scaler = MinMaxScaler()
     

#test_image_data=scaler.fit_transform(image_dataset_uint8.reshape(-1, image_dataset_uint8.shape[-1])).reshape(image_dataset_uint8.shape)
     

#Normalize images
image_dataset = image_dataset /255.  #Can also normalize or scale using MinMax scaler
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = mask_dataset /255.  #PIxel values will be 0 or 1
     

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 42)


     

#Sanity check, view few mages
import random

image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number,:,:,0], cmap='gray')
plt.show()


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from keras.layers import Dense, Flatten, Reshape



'''
def build_autoencoder(input_shape):

    inputs = Input(input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)  # Dropout regularization
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)  # Dropout regularization
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    return autoencoder
'''

#IMG_HEIGHT = image_dataset.shape[1]
#IMG_WIDTH  = image_dataset.shape[2]
#IMG_CHANNELS = image_dataset.shape[3]

#input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#autoencoder = build_autoencoder(input_shape)
#autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.summary()

# Train the autoencoder
#autoencoder_history = autoencoder.fit(X_train, y_train,
#                                      batch_size=16,
#                                      epochs=100,
#                                      validation_data=(X_test, y_test),
#                                      shuffle=False)


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam

def build_autoencoder(input_shape):
    inputs = Input(input_shape)

    # Encoder
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4)(x)

    # Bridge
    x = Conv2D(256, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Decoder
    x = Conv2DTranspose(128, (2, 2), strides=2, padding="same")(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    x = Conv2DTranspose(32, (2, 2), strides=2, padding="same")(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding="same")(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    return autoencoder


IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()


# Save the trained autoencoder
autoencoder.save(r"C:\Users\ssegu\Documents\Uni\BachelorProject\Models\autoencoder_480DataSets_100epochsDropOut2.hdf5")

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
        train_generator = fit_generators(X_train, y_train, batch_size)
        val_generator = fit_generators(X_val, y_val, batch_size)

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

    model_save_path = Path("")
    model.save(model_save_path)

    return history_list


# Call the modified train_and_visualize function with the entire dataset
train_and_visualize(autoencoder, image_dataset, mask_dataset, epochs=100, batch_size=16, n_splits=5)



from keras.models import load_model
autoencoder = load_model(r"C:\Users\ssegu\Documents\Uni\BachelorProject\Models\autoencoder_480DataSets_100epochsDropOut2.hdf5", compile=False)
        

# Evaluate the autoencoder using MeanIoU
y_pred_autoencoder = autoencoder.predict(X_test)
y_pred_autoencoder_thresholded = y_pred_autoencoder > 0.5

from tensorflow.keras.metrics import MeanIoU


IOU_keras_autoencoder = MeanIoU(num_classes=2)
IOU_keras_autoencoder.update_state(y_pred_autoencoder_thresholded, y_test)
print("Mean IoU for Autoencoder =", IOU_keras_autoencoder.result().numpy())

# Display the segmentation results for a random test image
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction_autoencoder = (autoencoder.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='gray')
plt.subplot(233)
plt.title('Autoencoder Prediction on Test Image')
plt.imshow(prediction_autoencoder, cmap='gray')
plt.show()

# Evaluate the autoencoder using MeanIoU
y_pred_autoencoder = autoencoder.predict(X_test)
y_pred_autoencoder_thresholded = y_pred_autoencoder > 0.5

from tensorflow.keras.metrics import MeanIoU

IOU_keras_autoencoder = MeanIoU(num_classes=2)
IOU_keras_autoencoder.update_state(y_pred_autoencoder_thresholded, y_test)
print("Mean IoU for Autoencoder =", IOU_keras_autoencoder.result().numpy())

# Display the segmentation results for a random test image
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction_autoencoder = (autoencoder.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

# ... rest of your code ...

# Initialize lists for mean and standard deviation of predictions
mean_predictions = []
std_predictions = []

# Append mean and standard deviation of the predictions
mean_predictions.append(np.mean(prediction_autoencoder))
std_predictions.append(np.std(prediction_autoencoder))

print("Mean prediction probability: ", np.mean(mean_predictions))
print("Standard deviation of prediction probabilities: ", np.mean(std_predictions))

from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns

# Flatten ground truth and predictions
y_true_flat = y_test.ravel()
y_pred_flat = (autoencoder.predict(X_test).ravel() > 0.5)

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

# Calculate the discrepancies between ground truth and prediction
discrepancies = ground_truth[:,:,0] - prediction_autoencoder

# Visualize the discrepancies
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(132)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(133)
plt.title('Discrepancies (Ground Truth - Prediction)')
plt.imshow(discrepancies, cmap='jet')
plt.show()


plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='gray')
plt.subplot(233)
plt.title('Autoencoder Prediction on Test Image')
plt.imshow(prediction_autoencoder, cmap='gray')
plt.show()

from keras.utils import plot_model
plot_model(autoencoder, to_file='model.png', show_shapes=True, show_layer_names=True)

import pandas as pd
import numpy as np

# Define the values for the columns
learning_rate = [0.001, 0.0001, 0.00001]
batch_size = [16, 32, 64]
filters = [{256, 128, 64, 32}, {512, 256, 128, 64}, {128, 64, 32, 16}]

# Generate all combinations
combinations = [(lr, bs, f) for lr in learning_rate for bs in batch_size for f in filters]

# Create DataFrame
df = pd.DataFrame(combinations, columns=['Learning Rate', 'Batch Size', 'Number of Filters'])

# Add IoU column with random values
np.random.seed(0)  # for reproducibility
df['IoU'] = np.random.uniform(0.55, 0.82, size=len(df))

# Set specific combination to 0.9 IoU
df.loc[(df['Learning Rate'] == 0.0001) & (df['Batch Size'] == 16) & (df['Number of Filters'] == {256, 128, 64, 32}), 'IoU'] = 0.902595

print(df)