import keras
import data
from keras.layers import LeakyReLU, Activation, Dense, Flatten, Conv2D, Conv1D, MaxPooling2D, Dropout

import data as dt

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

DIMENSIONS = (256, 192, 1)

#########################################
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


class LeNet:
    @staticmethod

    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model

    def model_test(self, data, labels):
        EPOCHS = 25
        INIT_LR = 1e-3
        BS = 32

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(data,
                                                          labels, test_size=0.25, random_state=42)
        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=4)
        testY = to_categorical(testY, num_classes=4)

        # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")

        # initialize the model
        print("[INFO] compiling model...")
        model = LeNet.build(width=48, height=64, depth=3, classes=4)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

        # train the network
        print("[INFO] training network...")
        H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                                validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                                epochs=EPOCHS, verbose=1)
        pass

