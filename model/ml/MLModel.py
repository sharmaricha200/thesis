import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from distutils.version import StrictVersion

class DNNModel:
    def __init__(self, model_path, kernel_reg = 0.003, bias_reg = 0.003):
        self.model = None
        self.model_path = model_path
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg

    def train(self, x_train, y_train, rep):
        self.model = Sequential()
        kreg = self.kernel_reg
        breg = self.bias_reg
        self.model.add(
            Dense(1000, input_dim=2000, activation='sigmoid'))
        self.model.add(Dense(100, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
        self.model.add(Dense(10, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
        self.model.add(Dense(2, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        self.model.summary()
        history = self.model.fit(x_train, y_train, epochs=rep)

        fig_path = os.path.join(os.path.dirname(self.model_path), "model_performance.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        fig = plt.figure()
        if StrictVersion(keras.__version__) > StrictVersion('2.2.5'):
            plt.plot(history.history['accuracy'])
        else:
            plt.plot(history.history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        pdf.savefig(fig)
        plt.close()

        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        pdf.savefig(fig)
        plt.close()
        pdf.close()

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
