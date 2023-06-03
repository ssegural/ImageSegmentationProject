# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 21:20:09 2023

@author: ssegu
"""

import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image
from patchify import patchify 
import tifffile as tiff
import glob 
from skimage import io 
import numpy as np
import os 
import cv2 


dir = "C:/Users/ssegu/Documents/Uni/BachelorProject/Pictures/prueba.tif"

#org = Image.open(dir)

#img_arr = np.asarray(org)
#print(img_arr)
#patches = patchify(img_arr, (128,128,3),step = 128)

#print(patches.shape)

#patch_img_arr = patches[1,1,0,:,:,:]
#patch_img = Image.fromarray(patch_img_arr)
#print(patch_img)
#patch_img.show()
#for i in range(patches.shape[0]): 
#    for j in range(patches.shape[1]): 
#        patch = patches[i,j,0,:,:,:]
#        patchImg = Image.fromarray(patch)
#        num = i * patches.shape[1] + j
#        patchImg.save("C:/Users/ssegu/Documents/Uni/BachelorProject/TiffStack/Img_{num}.png")
        
        
        
org2 = cv2.imread(dir)
patchesImg = patchify(org2,(128,128,3),step=128)

for i in range(patchesImg.shape[0]): 
    for j in range(patchesImg.shape[1]):
        patch2 = patchesImg[i,j,0,:,:,:]
        if not cv2.imwrite("C:/Users/ssegu/Documents/Uni/BachelorProject/TestImages/"+"img_12"+"_"+str(i)+str(j)+".png",patch2):
            raise Exception("Could not write the image")
            
            

dir2 = r"C:\Users\ssegu\Documents\Uni\BachelorProject\Masks\img5Mask.tiff"

org = cv2.imread(dir2)


print(org)

plt.imshow(org, cmap='gray')

thres = 0 
org[org>thres] = 255

io.imshow(org)
io.show()

print(org)

cv2.imwrite("C:/Users/ssegu/Documents/Uni/BachelorProject/Masks/"+"img5MaskTiff"+".tiff",org)


dir3 = r"C:\Users\ssegu\Documents\Uni\BachelorProject\masksApeer\Limg12_finalprediction.ome.tiff"
org = cv2.imread(dir3)


print(org)

plt.imshow(org, cmap='gray')

inverted_image = cv2.bitwise_not(org, 255)

plt.imshow(inverted_image,cmap = 'gray')

cv2.imwrite("C:/Users/ssegu/Documents/Uni/BachelorProject/Masks/"+"img_12"+"MaskTiff.tiff",inverted_image)