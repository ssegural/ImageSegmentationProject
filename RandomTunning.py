# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:15:13 2023

@author: ssegu
"""
import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose,
                          BatchNormalization, Dropout, Activation, MaxPool2D, Concatenate)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping


image_directory = r"C:\Users\ssegu\Documents\Uni\BachelorProject\trainImages"
mask_directory = r"C:\Users\ssegu\Documents\Uni\BachelorProject\maskTrainImages"

num_images=480
SIZE = 128

image_names = glob.glob(r"C:\Users\ssegu\Documents\Uni\BachelorProject\trainImages\*.tiff")
print(image_names)


image_names.sort()

     

image_names_subset = image_names[0:num_images]
     

images = [cv2.imread(img, 0) for img in image_names_subset]
     

#image_dataset = np.array(images)
#image_dataset = np.expand_dims(image_dataset, axis = 3)
#image_dataset = np.array([np.repeat(img, 3, axis=-1) for img in images])
image_dataset = np.array([np.repeat(img[:, :, np.newaxis], 3, axis=-1) for img in images])




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
     



#Normalize images
image_dataset = image_dataset /255.  #Can also normalize or scale using MinMax scaler
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = mask_dataset /255.  #PIxel values will be 0 or 1
     

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 42)

y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)
#Sanity check
import random

image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number,:,:,0], cmap='gray')
plt.show()




input_shape = (SIZE, SIZE, 3)
# Modify the build_unet function to include the hyperparameters
def build_unet(input_shape, n_classes, num_filters_start, dropout_rate):
    def conv_block(input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def encoder_block(input, num_filters):
        x = conv_block(input, num_filters)
        p = MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = conv_block(x, num_filters)
        return x

    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, num_filters_start)
    s2, p2 = encoder_block(p1, num_filters_start * 2)
    s3, p3 = encoder_block(p2, num_filters_start * 4)
    s4, p4 = encoder_block(p3, num_filters_start * 8)

    b1 = conv_block(p4, num_filters_start * 16)

    d1 = decoder_block(b1, s4, num_filters_start * 8)
    d1 = Dropout(dropout_rate)(d1)
    d2 = decoder_block(d1, s3, num_filters_start * 4)
    d2 = Dropout(dropout_rate)(d2)
    d3 = decoder_block(d2, s2, num_filters_start * 2)
    d3 = Dropout(dropout_rate)(d3)
    d4 = decoder_block(d3, s1, num_filters_start)
    d4 = Dropout(dropout_rate)(d4)

    outputs = Conv2D(n_classes, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model
from keras.losses import binary_crossentropy

def create_tuned_unet(input_shape=(128, 128, 3), output_channels=1, num_filters_start=32, dropout_rate=0.5, learning_rate=1e-4):
    model = build_unet(input_shape, n_classes=1, num_filters_start=num_filters_start, dropout_rate=dropout_rate)
    model.compile(optimizer=Adam(lr=learning_rate), loss=binary_crossentropy, metrics=["accuracy"])
    return model

import numpy as np
import random
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

def iou(y_true, y_pred, epsilon=1e-7):
    y_true = np.round(y_true.flatten())
    y_pred = np.round(y_pred.flatten())
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + epsilon)

def perform_kfold_cv(X, y, create_model_func, params, num_folds=3):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = create_model_func(**params)
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, verbose=1)]
        model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), callbacks=callbacks)
        y_pred = model.predict(X_val_fold)
        iou_score = iou(y_val_fold, y_pred)
        scores.append(iou_score)
    return np.mean(scores, axis=0)


def random_search(create_model_func, param_grid, X, y, num_iter=5, num_folds=3):
    keys, values = zip(*param_grid.items())
    best_score = -np.inf
    best_params = None

    for _ in range(num_iter):
        params = dict(zip(keys, [random.choice(v) for v in values]))
        score = perform_kfold_cv(X, y, create_model_func, params, num_folds=num_folds)
        print("Score:", score, "Params:", params)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

# Define the parameter grid for random search
param_grid = {
    'num_filters_start': [16, 32, 64],
    'dropout_rate': [0.1, 0.3, 0.5],
    'learning_rate': [0.01, 0.001, 0.0001]
}

# Perform random search
best_params, best_score = random_search(create_tuned_unet, param_grid, X_train, y_train, num_iter=100, num_folds=3)
print("Best Params:", best_params, "Best Score:", best_score)

# Train the best model with the best parameters
best_model = create_tuned_unet(**best_params)
best_model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=10, verbose=1)])




# Save the best model
best_model.save("best_unet_model.hdf5")

# Evaluate the best model on the test set
score = best_model.evaluate(X_test, y_test, verbose=1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])