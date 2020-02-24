import numpy as np
import pandas as pd
import os
from scipy import misc
from scipy import ndimage
from PIL import Image
from skimage.color import rgb2gray


image_path = "Train_images/Sharif/"
image_paths = ["Train_images/Sharif/", "Train_images/Robin/", "Train_images/Vincent/", "Train_images/Pelle/"]
MAX_SIZE = (256, 192)

def resize_images():
    for image_path in image_paths:
        photos = os.listdir(image_path)
        for photo in photos:
            with Image.open(image_path+str(photo)) as image:
                image.thumbnail(MAX_SIZE)
                image.save(image_path+str(photo))

def get_data():
    cols = ['pix' + str(i) for i in range(27648)]
    df = pd.DataFrame(columns=cols)
    df = pd.DataFrame({"columns": cols})
    it = 1; it2 = 0
    name_list = ['Sharif', 'Robin', 'Vincent', 'Pelle']
    labels = []

    for image_path in image_paths:
        photos = os.listdir(image_path)
        for photo in photos:
            with Image.open(image_path+str(photo)) as image:
                im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
                im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
                im = np.ndarray.flatten(rgb2gray(im_arr))
                df['img'+str(it)] = pd.Series(im)

                labels.append(name_list[it2])
                it += 1
        it2 += 1
    df = df.set_index("columns")
    df = df.transpose()
    df['labels'] = labels
    df = df.transpose()
    return df

def show_image(image):
    pass