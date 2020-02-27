import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data
import cv2
from keras.preprocessing.image import img_to_array

def multi(data):
    f = plt.figure()
    f.cmap = 'gray'
    for i in range(25):
        f.add_subplot(5,5,i+1)
        plt.imshow(data[i])
        plt.axis("off")
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=1, bottom=0, left=0,
                            right=1)
    plt.savefig("multi_faces.png")
