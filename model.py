import keras
import data
from keras.layers import LeakyReLU, Activation, Dense, Flatten, Conv2D, Conv1D, MaxPooling2D, Dropout

import data as dt

DIMENSIONS = (256,192, 1)


class Model():
    activation_type = None
    model = None

    def __init__(self, activation_type):
        self.activation = activation_type # 'relu', 'sigmoid' etc


        model = keras.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation=self.activation_type, input_shape=DIMENSIONS))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Softmax)


        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self, piccas):
        print("Training ...")
        if piccas.empty:
            print("ERROR: Dataframe empty")
            return
        self.model.fit((636, piccas[piccas.columns][0:len(piccas[piccas.columns[1]])-1], 1)
                       ,(636, piccas.loc['labels'][0], 1), epochs=1, batch_size=40)
