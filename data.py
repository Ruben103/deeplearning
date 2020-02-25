import numpy as np
import pandas as pd
import os
from scipy import misc
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import argparse
import random
import cv2
import os
from imutils import paths
from keras.preprocessing.image import img_to_array


image_path = "Train_images/Sharif/"
image_paths = ["Train_images/Sharif/", "Train_images/Robin/",  "Train_images/Vincent/", "Train_images/Pelle/"]
name_list = ['Sharif', 'Robin', 'Vincent', 'Pelle']
cols = ['pix' + str(i) for i in range(27648)]
MAX_SIZE = (256, 192)

class Data():


    def load_data_test(self):

        data = []
        labels = []

        # grab the image paths and randomly shuffle them
        # loop over the input images
        for i in range(4):
            imagePaths = sorted(list(paths.list_images(image_paths[i])))
            l = len(imagePaths)
            np.random.shuffle(imagePaths)
            for imagePath in imagePaths:
                # load the image, pre-process it, and store it in the data list
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (64, 48))
                image = img_to_array(image)
                data.append(image)
                # labels list
                np.repeat(labels.append(i), l)

        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        return data, labels



