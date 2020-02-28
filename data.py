import numpy as np
import cv2
import os
from imutils import paths
from keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle

image_path = "Train_images/Sharif/"
image_paths = ["extracted_faces/extr_pelle", "extracted_faces/extr_robin",  "extracted_faces/extr_sharif",
               "extracted_faces/extr_vincent", "textracted_faces/est_extr_pelle", "extracted_faces/test_extr_robin",
               "extracted_faces/test_extr_sharif", "extracted_faces/test_extr_vincent"]
name_list = ['Sharif', 'Robin', 'Vincent', 'Pelle']
cols = ['pix' + str(i) for i in range(27648)]
MAX_SIZE = (256, 192)

class Data():


    def face_extr(self):
        osgetcwd = os.getcwd()
        path = os.path.join(osgetcwd, '/haarcascade_frontalface_default.xml')
        faceCascade = cv2.CascadeClassifier(path)

        for name in name_list:
            imagePaths = sorted(list(paths.list_images("test_images_raw" + '/' + name)))

            faces = []
            i = 1
            for imagepath in imagePaths:
                image = cv2.imread(imagepath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                faces.append(face)

                print("Found {0} faces!".format(len(face)))
                if(len(face) > 0):
                    p = 0
                    # extract largest face found
                    for (x, y, w, h) in face:
                        if(w+h > p):
                            frame = image[y:y + h, x:x + w]
                            p = w+h
                    cv2.imshow("Faces found", frame)
                    cv2.waitKey(1)

                    # status = cv2.imwrite('/home/sharif/Master/deep_learning/deeplearning/test_extr_vincent/faces_detected' + str(i) +
                    #                      '.jpg', frame)
                    new_path = os.path.join(os.getcwd(), "/extraced_faces", "/test_extr_" + "name", "/faces_detected")
                    status = cv2.imwrite(new_path + str(i) +
                        '.jpg', frame)
                    i = i+1

        pass

    def load_data_test(self, toggle_large_data):

        print(os.getcwd())

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
                if toggle_large_data:
                    width = 192; height = 256
                else:
                    width = 48; height = 64
                image = cv2.resize(image, (width, height))
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
        train_data, train_labels = shuffle(train_data, train_labels, random_state=13)
        test_data, test_labels = shuffle(test_data, test_labels, random_state=13)

        return train_data, train_labels, test_data, test_labels, width, height

    def load_full_data(self):

        data = []

        # grab the image paths and randomly shuffle them
        # loop over the input images
        for i in range(4):
            imagePaths = sorted(list(paths.list_images(image_paths[i])))
            l = len(imagePaths)
            for imagePath in imagePaths:
                # load the image, pre-process it, and store it in the data list
                image = cv2.imread(imagePath)
                data.append(image)

        # scale the raw pixel intensities to the range [0, 1]
        dat = np.array(data) / 255.0

        return dat



