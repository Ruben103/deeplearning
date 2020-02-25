import numpy as np
import pandas as pd
import os
from scipy import misc
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


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
        cols = ['pix' + str(i) for i in range(27648)]
        df = pd.DataFrame(columns=cols)
        df = pd.DataFrame({"columns": cols})
        it = 1; it2 = 1

        labels = []

        for image_path in image_paths:
            photos = os.listdir(image_path)
            for photo in photos:
                with Image.open(image_path+str(photo)) as image:
                    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
                    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
                    im = np.ndarray.flatten(rgb2gray(im_arr))
                    df['img'+str(it)] = pd.Series(im)

                    labels.append(name_list[it2-1])
                    it += 1
            it2 += 1
        df = df.set_index("columns")
        df = df.transpose()
        df['labels'] = labels
        df = df.transpose()
        return df

def show_image(image):
    """

    :param image: series column of image --> crop labels
    :return:
    """
    image = image[0:len(image) - 1]
    image = np.resize(image, (256, 192))
    plt.imshow(image, cmap='gray')


