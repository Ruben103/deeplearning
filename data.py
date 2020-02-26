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
from cv2 import cvtColor
import os
from imutils import paths
from keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle
import sys


image_path = "Train_images/Sharif/"
image_paths = ["Train_images/Sharif/", "Train_images/Robin/",  "Train_images/Vincent/", "Train_images/Pelle/",
               "test_images/Sharif/", "test_images/Robin/",  "test_images/Vincent/", "test_images/Pelle/"]
name_list = ['Sharif', 'Robin', 'Vincent', 'Pelle']
cols = ['pix' + str(i) for i in range(27648)]
MAX_SIZE = (256, 192)

class Data():


    def face_extr(self):
        faceCascade = cv2.CascadeClassifier("test_images/Sharif_extracted/figure1.xml")
        image = cv2.imread("test_images/Sharif/")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pass

    def load_data_test(self):

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        # grab the image paths and randomly shuffle them
        # loop over the input images
        for i in range(8):
            imagePaths = sorted(list(paths.list_images(image_paths[i])))
            l = len(imagePaths)
            for imagePath in imagePaths:
                # load the image, pre-process it, and store it in the data list
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (64, 48))
                image = img_to_array(image)
                if(i < 4):
                    train_data.append(image)
                else:
                    test_data.append(image)
            # labels list
                if(i < 4):
                    np.repeat(train_labels.append(i), l)
                else:
                    np.repeat(test_labels.append(i-4), l)

        # scale the raw pixel intensities to the range [0, 1]
        train_data = np.array(train_data, dtype="float") / 255.0
        train_labels = np.array(train_labels)
        test_data = np.array(test_data, dtype="float") / 255.0
        test_labels = np.array(test_labels)
        #Shuffle the test and training data
        shuffle(train_data, train_labels)
        shuffle(test_data, test_labels)

        return train_data, train_labels, test_data, test_labels



