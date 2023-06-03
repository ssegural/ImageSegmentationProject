# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:18:50 2023

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
image_directory = ""
mask_directory = ""
     

SIZE = 128
num_images = 570
     
#Load images and masks in order so they match


image_names = glob.glob(r"your path")
print(image_names)


image_names.sort()
print(image_names)
     

image_names_subset = image_names[0:num_images]
     

images = [cv2.imread(img, 0) for img in image_names_subset]
     

image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis = 3)
     
#Read masks the same way.


mask_names = glob.glob(r"your path")
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

from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Input
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

def build_fcn(input_shape, n_classes):
    inputs = Input(input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)

    # Decoder
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)

    x = Conv2DTranspose(n_classes, (2, 2), strides=(2, 2), padding='same')(x)

    if n_classes == 1:  # Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Activation(activation)(x)

    model = Model(inputs, outputs, name="FCN")
    return model

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# Use the same input_shape from the U-Net example
model = build_fcn(input_shape, n_classes=1)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



# Train the autoencoder
model_history = model.fit(X_train, y_train,
                                      batch_size=16,
                                      epochs=100,
                                      validation_data=(X_test, y_test),
                                      shuffle=False)

# Save the trained autoencoder
model.save(r"C:\Users\ssegu\Documents\Uni\BachelorProject\Models\FCN_480DataSets_100epochs(256 filters + 16 batchsize with dropout).hdf5")

# Evaluate the autoencoder using MeanIoU
y_pred_autoencoder = model.predict(X_test)
y_pred_autoencoder_thresholded = y_pred_autoencoder > 0.5

from tensorflow.keras.metrics import MeanIoU

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


IOU_keras_autoencoder = MeanIoU(num_classes=2)
IOU_keras_autoencoder.update_state(y_pred_autoencoder_thresholded, y_test)
print("Mean IoU for FCN =", IOU_keras_autoencoder.result().numpy())

# Display the segmentation results for a random test image
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction_autoencoder = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

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