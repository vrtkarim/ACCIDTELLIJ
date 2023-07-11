import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from matplotlib.image import imread
import cv2
import numpy as np

image = imread("C:/Users/elyaa/Downloads/fwekjgrrgkj.png")

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
filtre = np.array([[-1,-2,-1],
                    [9, 9, 9],
                    [1,2,1]])
conv = cv2.filter20(image, 1, filtre)
plt.imshow(filter, cmap='viridis')
plt.imshow(conv, cmap='viridis')

def relu(x):
    return np.maximum(0,x) 

reluimage = relu(conv)
plt.imshow(reluimage, cmap='viridis')

poolimage = block_reduce(reluimage, (10,10), func=np.max)
plt.imshow(poolimage)