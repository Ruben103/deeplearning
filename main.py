import data
import model
import scipy
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == '__main__':

    dt = data.Data()
    dt.face_extr()
    #train_data, train_labels, test_data, test_labels = dt.load_data_test()
    #dingutje = model.LeNet()
    #dingutje.model_test(train_data, train_labels, test_data, test_labels)

    print("BUGSTOPP")