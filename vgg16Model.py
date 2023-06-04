# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 07:39:43 2023

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

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Check if TensorFlow is running on GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# ... (Same code for loading and preprocessing images and masks)
#Load images and masks in order so they match

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
image_dataset = np.array([np.repeat(img, 3, axis=-1) for img in images])
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

     

#Sanity check
import random

image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number,:,:,0], cmap='gray')
plt.show()
     
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
"""
# Define helper function for the decoder block
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = BatchNormalization()(x)  # Add batch normalization layer
    x = Dropout(0.4)(x)
    x = Concatenate()([x, skip_features])
    x = Conv2D(num_filters, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)  # Add batch normalization layer
    x = Dropout(0.4)(x)
    x = Conv2D(num_filters, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)  # Add batch normalization layer
    x = Dropout(0.4)(x)  # Add dropout layer

    return x

# Build the VGG16-based segmentation model
def build_vgg16_unet(input_shape, n_classes):
    base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')

    # Encoder part of the architecture (VGG16 model)
    s1 = base_model.get_layer("block1_conv2").output
    s1 = Dropout(0.4)(s1)
    s2 = base_model.get_layer("block2_conv2").output
    s2 = Dropout(0.4)(s2)
    s3 = base_model.get_layer("block3_conv3").output
    s3 = Dropout(0.4)(s3)
    s4 = base_model.get_layer("block4_conv3").output
    s4 = Dropout(0.4)(s4)
    b1 = base_model.get_layer("block5_conv3").output
    

    # Decoder part of the architecture
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    if n_classes == 1:  # Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  # Change the activation based on n_classes
    print(activation)

    model = Model(base_model.input, outputs, name="VGG16_U-Net")
    return model
"""


from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

# Define helper function for the decoder block
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = BatchNormalization()(x)  # Add batch normalization layer
    x = Dropout(0.4)(x)
    x = Concatenate()([x, skip_features])
    x = Conv2D(num_filters, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)  # Add batch normalization layer
    x = Dropout(0.4)(x)
    x = Conv2D(num_filters, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)  # Add batch normalization layer
    x = Dropout(0.4)(x)  # Add dropout layer

    return x

# Build the VGG16-based segmentation model
def build_vgg16_unet(input_shape, n_classes):
    base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')

    # Encoder part of the architecture (VGG16 model)
    s1 = base_model.get_layer("block1_conv2").output   # 64 filters
    s1 = Dropout(0.4)(s1)
    s2 = base_model.get_layer("block2_conv2").output   # 128 filters
    s2 = Dropout(0.4)(s2)
    s3 = base_model.get_layer("block3_conv3").output   # 256 filters
    s3 = Dropout(0.4)(s3)
    s4 = base_model.get_layer("block4_conv3").output   # 512 filters
    s4 = Dropout(0.4)(s4)
    b1 = base_model.get_layer("block5_conv3").output   # 512 filters

    # Decoder part of the architecture
    d1 = decoder_block(b1, s4, 512)   # Match number of filters with block4 of encoder
    d2 = decoder_block(d1, s3, 256)   # Match number of filters with block3 of encoder
    d3 = decoder_block(d2, s2, 128)   # Match number of filters with block2 of encoder
    d4 = decoder_block(d3, s1, 64)    # Match number of filters with block1 of encoder

    if n_classes == 1:  # Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  # Change the activation based on n_classes
    print(activation)

    model = Model(base_model.input, outputs, name="VGG16_U-Net")
    return model

# Instantiate the VGG16-based model and compile it
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = build_vgg16_unet(input_shape, n_classes=1)

# Freeze the layers of the VGG16 model
for layer in model.layers[:19]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ... (Same code for training, saving, plotting, and evaluating the model)


history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=100, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save(r"C:\Users\ssegu\Documents\Uni\BachelorProject\Models\Vgg16_480Datasets_100epochsDropOut3.hdf5")


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






from keras.models import load_model
model = load_model(r"C:\Users\ssegu\Documents\Uni\BachelorProject\Models\Vgg16_480Datasets_100epochsDropOut2.hdf5", compile=False)

#X_test_rgb = np.repeat(X_test, 3, axis=-1)
#y_pred = model.predict(X_test_rgb)
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5
     
from tensorflow.keras.metrics import MeanIoU
     

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_pred_thresholded, y_test)
print("Mean IoU =", IOU_keras.result().numpy())

from sklearn.metrics import jaccard_score



def mean_iou(y_true, y_pred, num_classes=2):
    # Flatten the input
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    ious = []
    for c in range(num_classes):
        # Convert the scalar to a tensor
        c_tensor = tf.constant(c, dtype=y_true.dtype)
        
        # Calculate the true positives, false positives and false negatives
        tp = tf.reduce_sum(tf.cast(tf.equal(y_pred, c_tensor) & tf.equal(y_true, c_tensor), dtype=tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.equal(y_pred, c_tensor) & tf.not_equal(y_true, c_tensor), dtype=tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, c_tensor) & tf.equal(y_true, c_tensor), dtype=tf.float32))
        
        # Calculate IoU for this class and append to the list of IoUs
        iou = tp / (tp + fp + fn + tf.keras.backend.epsilon())  # adding a small constant to avoid division by zero
        ious.append(iou)
    
    # Take the mean over all classes
    mean_iou = tf.reduce_mean(ious)
    return mean_iou


# Make predictions (assumes your model's output is a binary mask)
y_pred = model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

# Compute the mean IoU using the custom function
mean_iou_value = mean_iou(y_test, y_pred_thresholded)
print("Mean IoU =", mean_iou_value.numpy())


# Function to compute the intersection over union (IoU) metric

threshold = 0.5
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
print(prediction.shape)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()


prediction2 = (model.predict(test_img_input)[0,:,:,0])
plt.hist(prediction2.ravel(), bins=50, range=(0, 1))
plt.xlabel("Prediction Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Probabilities")
plt.show()

#threshold = 0.5
#test_img_number = random.randint(0, len(X_test)-1)
#test_img = X_test[test_img_number]
#ground_truth=y_test[test_img_number]
#test_img_rgb = np.repeat(test_img[..., np.newaxis], 3, -1) # convert to RGB
#test_img_input=np.expand_dims(test_img_rgb, 0)
#prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

#plt.figure(figsize=(16, 8))
#plt.subplot(231)
#plt.title('Testing Image')
#plt.imshow(test_img[:,:,0], cmap='gray')
#plt.subplot(232)
#plt.title('Testing Label')
#plt.imshow(ground_truth[:,:,0], cmap='gray')
#plt.subplot(233)
#plt.title('Prediction on test image')
#plt.imshow(prediction, cmap='gray')

#plt.show()



# Initialize lists for mean and standard deviation of predictions
mean_predictions = []
std_predictions = []

# Append mean and standard deviation of the predictions
mean_predictions.append(np.mean(prediction))
std_predictions.append(np.std(prediction))

print("Mean prediction probability: ", np.mean(mean_predictions))
print("Standard deviation of prediction probabilities: ", np.mean(std_predictions))

from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns

# Flatten ground truth and predictions
y_true_flat = y_test.ravel()
y_pred_flat = (model.predict(X_test).ravel() > 0.5)

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
discrepancies = ground_truth[:,:,0] - prediction

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


