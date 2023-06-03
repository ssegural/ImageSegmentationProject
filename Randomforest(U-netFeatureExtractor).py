# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:56:42 2023

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
image_directory = Path("")
mask_directory = Path("")

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

    model_save_path = Path("")
    model.save(model_save_path)

    return history_list


# Call the modified train_and_visualize function with the entire dataset
train_and_visualize(model, image_dataset, mask_dataset, epochs=100, batch_size=16, n_splits=5)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from tensorflow.keras.models import load_model, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = load_model("")

# Define a new model that will input the same as your original model but output the layer prior to the output layer
feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Assuming X_train, X_test, y_train, y_test are previously loaded or defined...

# Get the features for train and test datasets
X_train_features = feature_model.predict(X_train)
X_test_features = feature_model.predict(X_test)

# Reshape the features to 2D array
n_samples_train = X_train_features.shape[0]
n_features_train = np.prod(X_train_features.shape[1:])  
X_train_features_2d = X_train_features.reshape((n_samples_train, n_features_train))

n_samples_test = X_test_features.shape[0]
n_features_test = np.prod(X_test_features.shape[1:])  
X_test_features_2d = X_test_features.reshape((n_samples_test, n_features_test))

# Reshape the masks to 2D array
y_train_2d = y_train.reshape((y_train.shape[0], -1))
y_test_2d = y_test.reshape((y_test.shape[0], -1))

from sklearn.metrics import accuracy_score, jaccard_score

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the entire training data
rf.fit(X_train_features_2d, y_train_2d)

# Lists to store accuracy and jaccard scores
accuracy_scores = []
jaccard_scores = []
predicted_images = []

for i in range(X_test.shape[0]):
    # Extract features for the corresponding test image
    X_test_features = feature_model.predict(X_test[i:i+1])
    X_test_features_2d = X_test_features.reshape((-1, X_test_features.shape[-1]))

    # Predict on the test data
    y_pred = rf.predict(X_test_features_2d)

    # Reshape the test mask and the predicted mask to compute the accuracy and jaccard score
    y_test_1d = y_test[i].reshape((-1,))
    y_pred = y_pred.reshape(y_test[i].shape)

    # Compute the accuracy and jaccard score
    accuracy_scores.append(accuracy_score(y_test_1d, y_pred.flatten()))
    jaccard_scores.append(jaccard_score(y_test_1d, y_pred.flatten()))
    predicted_image = y_pred.reshape(X_test[i].shape[:2])
    predicted_images.append(predicted_image)

# Print average accuracy and jaccard score
print(f"Average Random Forest Classifier accuracy: {np.mean(accuracy_scores)}")
print(f"Average Random Forest Classifier jaccard score: {np.mean(jaccard_scores)}")

import matplotlib.pyplot as plt

# Choose the index of the image you want to display
idx = 0

# Plot the ground truth mask and the predicted mask side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Ground Truth')
plt.imshow(y_test[idx], cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Prediction')
plt.imshow(predicted_images[idx], cmap='gray')
plt.show()

