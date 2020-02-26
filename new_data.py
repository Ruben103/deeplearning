import numpy as np
import pandas as pd
import os
from scipy import misc
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import argparse
from imutils import paths


image_path = "Train_images/Sharif/"
image_paths = ["Train_images/Sharif/", "Train_images/Robin/",  "Train_images/Vincent/", "Train_images/Pelle/"]
name_list = ['Sharif', 'Robin', 'Vincent', 'Pelle']
cols = ['pix' + str(i) for i in range(27648)]
MAX_SIZE = (256, 192)

class Data():

    def __init__(self):
        self.sharif = []
        self.robin = []
        self.vincent= []
        self.pelle = []

        # Keep track of correct classifications
        self.correct_s = []
        self.incorrect_s = []
        self.correct_r = []
        self.incorrect_r = []
        self.correct_v = []
        self.incorrect_v = []
        self.correct_p = []
        self.incorrect_p = []
        self.cols = ['pix' + str(i+1) for i in range(27648)]
        self.img = ['img' + str(i+1) for i in range(636)]

    def resize_images(self):
        for image_path in image_paths:
            photos = os.listdir(image_path)
            for photo in photos:
                with Image.open(image_path+str(photo)) as image:
                    image.thumbnail(MAX_SIZE)
                    image.save(image_path+str(photo))

    def get_data(self):
        self.resize_images()

        ap = argparse.ArgumentParser()
        # ap.add_argument("-d, ", "--dataset")
        args = vars(ap.parse_args())

        imagePaths = sorted(list(paths.list_images(args['Train_images'])))

        for image_path in image_paths:
            photos = os.listdir(image_path)
            for photo in photos:
                with Image.open(image_path+str(photo)) as image:
                    pass


def show_image(image):
    """

    :param image: series column of image --> crop labels
    :return:
    """
    image = image[0:len(image) - 1]
    image = np.resize(image, (256, 192))
    plt.imshow(image, cmap='gray')


