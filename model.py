import keras
from keras.layers import LeakyReLU, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

import data as dt

DIMENSIONS = dt.return_dimensions

class model(activation="sigmoid"):

    def __init__(self, model_type):
        self.model_type = model_type

        model = keras.Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=DIMENSIONS))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))



        if model_type != "sigmoid":
            self.activation = model_type
        else:
            #dosmtelse