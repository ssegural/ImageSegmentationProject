# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:33:24 2023

@author: ssegu
"""


from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import cv2
import base64
from PIL import Image
import io
import json
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt

app = Flask(__name__)

# Load your trained models
model1 = load_model(r"C:\Users\ssegu\Documents\Uni\BachelorProject\Models\480DataSets_100epochs_augmented_crossval.hdf5", compile=False)
model2 = load_model(r"C:\Users\ssegu\Documents\Uni\BachelorProject\Models\570DataSets2_30epochs.hdf5", compile=False)

models = {'model1': model1, 'model2': model2}

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/segment', methods=['POST'])
def segment():
    try:
        if request.method == 'POST':
            print("Received segment request")
            img_str = request.form['image']
            model_name = request.form['model']
            img_data = base64.b64decode(img_str.split(',')[1])

            # Select the appropriate model
            selected_model = models[model_name]

            # Convert the base64 image data to a numpy array
            img = Image.open(io.BytesIO(img_data)).convert('L')
            img = img.resize((128, 128))
            img_arr = np.array(img)
            img_arr = np.expand_dims(img_arr, axis=(0, 3))
            img_arr = img_arr / 255.

            # Predict the mask
            print("Predicting mask")
            pred = selected_model.predict(img_arr)
            print(pred.shape)
            if model_name == 'model1':  # U-Net model
                pred =  np.squeeze(pred > 0.5)
                print(pred.shape)
                plt.imshow(pred, cmap='gray')
                plt.show()
                img = Image.fromarray(np.uint8(pred * 255)).convert("RGB")
            elif model_name == 'model2':  # VGG16 U-Net model
                # Add the appropriate handling for VGG16 U-Net output
                pred = (pred[0,:,:,0] > 0.5).astype(np.uint8)
                img = Image.fromarray(pred * 255, 'L')
            # Encode the mask as base64
            print("Encoding mask")
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            return jsonify({'mask': img_base64})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500



def generate_lines(binary_image, num_lines, min_length_ratio=0.5):
    h, w = binary_image.shape[:2]
    min_length = min(h, w) * min_length_ratio

    line_images = []
    line_lengths = []
    for i in range(num_lines):
        x0 = np.random.randint(0, w)
        y0 = np.random.randint(0, h)
        angle = np.random.uniform(0, 180)  
        length = np.random.uniform(min_length, np.sqrt(h**2 + w**2))

        x1 = x0 + int(length * np.cos(np.deg2rad(angle)))
        y1 = y0 + int(length * np.sin(np.deg2rad(angle)))

        line_image = cv2.line(np.zeros_like(binary_image, dtype=np.uint8), (x0, y0), (x1, y1), (255), 1)

        line_images.append(line_image)
        line_lengths.append(length)

    return line_images, line_lengths

def calculate_grain_size(binary_image, num_lines=10):
    binary_image = (binary_image * 255).astype(np.uint8)
    line_images, line_lengths = generate_lines(binary_image, num_lines)
    combined_lines = np.zeros_like(binary_image)

    grain_sizes = []
    for line_image, line_length in zip(line_images, line_lengths):
        intersections = cv2.bitwise_and(binary_image, binary_image, mask=line_image)
        num_intersections = np.count_nonzero(intersections)
        grain_size = line_length / num_intersections if num_intersections else np.nan
        grain_sizes.append(grain_size)
        combined_lines = cv2.bitwise_or(combined_lines, line_image)

    return np.nanmean(grain_sizes), combined_lines

def calculate_area(binary_image):
    labeled_image, num_grains = label(binary_image, return_num=True)
    
    areas = []
    for region in regionprops(labeled_image):
        areas.append(region.area)
    
    average_area = np.mean(areas) if areas else np.nan
    return average_area, areas

def count_grains(binary_image):
    labeled_image, num_grains = label(binary_image, return_num=True)
    return num_grains, labeled_image

def plot_area_distribution(areas, bins=10):
    plt.hist(areas, bins=bins)
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.title('Area Distribution')
    plt.show()



@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if request.method == 'POST':
            print("Received analyze request")
            img_str = request.form['image']
            img_data = base64.b64decode(img_str.split(',')[1])

            # Convert the base64 image data to a numpy array
            img = Image.open(io.BytesIO(img_data)).convert('L')
            img_arr = np.array(img)

            # Now use the binary image (img_arr) for your grain analysis functions
            average_grain_size, combined_lines = calculate_grain_size(img_arr)
            average_area, areas = calculate_area(img_arr)
            num_grains, _ = count_grains(img_arr)

            # Create histogram
            hist, bin_edges = np.histogram(areas, bins=10)

            # Encode the combined_lines image as base64 for returning
            img_combined_lines = Image.fromarray(combined_lines, 'L')
            img_buffer = io.BytesIO()
            img_combined_lines.save(img_buffer, format='PNG')
            img_combined_lines_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            result = {
                'averageGrainSize': average_grain_size,
                'averageArea': average_area,
                'numGrains': num_grains,
                'combinedLines': img_combined_lines_base64,
                'histogram': {
                    'frequencies': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                },
                # You may also return other variables like 'areas' if needed
            }
            return jsonify(result)
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, use_reloader = False)
    