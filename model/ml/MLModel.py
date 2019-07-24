import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.models import load_model

MAX_X=801

class DNNModel:
    def __init__(self, model_path, kernel_reg = 0.01, bias_reg = 0.01):
        self.model = None
        self.model_path = model_path
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg

    def train(self, x_train, y_train, rep):
        self.model = Sequential()
        kreg = self.kernel_reg
        breg = self.bias_reg
        self.model.add(
            Dense(1602, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), input_dim=1602, activation='sigmoid'))
        self.model.add(Dense(801, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
        self.model.add(Dense(400, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
        self.model.add(Dense(100, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
        self.model.add(Dense(2, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        self.model.fit(x_train, y_train, epochs=rep)

    def save(self):
        self.model.save(self.model_path)
        print("Saved model to disk.")

    def load(self):
        self.model = load_model(self.model_path);

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))
