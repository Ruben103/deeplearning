import keras
import data
from keras.layers import LeakyReLU, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

import data as dt

DIMENSIONS = (256,192)

class Model():

    def __init__(self, activation_type):
        self.activation = activation_type # 'relu', 'sigmoid' etc


        model = keras.Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=DIMENSIONS))
        model.add(Activation(self.activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(self.activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(self.activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dense(64))
        model.add(Activation(self.activation))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        self.model = model

    def train(self, piccas):
        print("Training ...")
        if piccas.empty:
            print("ERROR: Dataframe empty")
            return
        self.model.fit(piccas[piccas.columns][0:len(piccas[piccas.columns[1]])-1], piccas.loc['labels'])

