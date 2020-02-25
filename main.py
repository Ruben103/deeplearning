import data
import model
import scipy
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == '__main__':

    # Prepoces data to be smaller
    # data.resize_images()
    dt = data.Data()
    piccas = dt.get_data()

    # data.show_image(piccas[piccas.columns[0]])

    model = model.Model('relu')
    print("model works")
    model.train(piccas)


    # image = rgb2gray(image)
    # plt.imshow(image,cmap="gray")

    print("BUGSTOPP")