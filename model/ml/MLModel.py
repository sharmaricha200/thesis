value = 7
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
random.seed(value)
import numpy as np
np.random.seed(value)
import tensorflow as tf
tf.random.set_seed(value)
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from distutils.version import StrictVersion

class DNNModel:
    def __init__(self, model_path, kernel_reg = 0.003, bias_reg = 0.003):
        self.model = None
        self.model_path = model_path
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg

    def train(self, x, y, rep):
        self.model = Sequential()
        kreg = self.kernel_reg
        breg = self.bias_reg
        self.model.add(Dense(1000, input_dim=2000, activation='softsign'))
        self.model.add(Dense(100, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softsign'))
        self.model.add(Dense(10, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softsign'))
        self.model.add(Dense(2, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x, y, epochs= rep, batch_size = 128, verbose = 0, shuffle=True)

    def cross(self, x, y, rep):
        kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = value)
        all_scores = []
        fig_path = os.path.join(os.path.dirname(self.model_path), "model_performance.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        for train, val in kfold.split(x, y.argmax(axis = 1)):
            model = Sequential()
            kreg = 0.003
            breg = 0.003
            model.add(Dense(1000, input_dim=2000, activation='softsign'))
            model.add(Dense(100, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softsign'))
            model.add(Dense(10, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softsign'))
            model.add(Dense(2, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()
            history = model.fit(x[train], y[train], epochs = rep, batch_size = 128, validation_data=(x[val], y[val]), verbose = 0)
            scores = model.evaluate(x[val], y[val], verbose = 0)
            # predictions = model.predict(x[val])
            # matrix = confusion_matrix(y[val].argmax(axis=1), predictions.argmax(axis=1))
            # print(matrix)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            all_scores.append(scores[1] * 100)

            fig = plt.figure()
            x_labs = [i for i in range(1, len(history.history['loss']) + 1)]
            plt.plot(x_labs, history.history['loss'])
            x_labs = [i for i in range(1, len(history.history['val_loss']) + 1)]
            plt.plot(x_labs, history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            pdf.savefig(fig)
            plt.close()

        pdf.close()

        print('%.2f%% (+/- %.2f%%)' % (np.mean(all_scores), np.std(all_scores)))

    def create_initial_model(self, activation = 'relu'):
        self.model = Sequential()
        kreg = self.kernel_reg
        breg = self.bias_reg
        self.model.add(Dense(1000, input_dim=2000, activation=activation))
        self.model.add(Dense(100, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation=activation))
        self.model.add(Dense(10, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation=activation))
        self.model.add(Dense(2, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def grid(self, x, y):
        model = KerasClassifier(build_fn = self.create_initial_model, verbose = 0)
        activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid']
        batch_size = [10, 20, 40, 64, 80, 100, 128]
        epochs = [3, 5, 30, 50, 100, 300]
        param_grid = dict(activation = activation, batch_size = batch_size, epochs = epochs)
        grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1)
        grid_result = grid.fit(x, y)
        print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (mean, stdev, param))

    def save(self):
        self.model.save(self.model_path)
        print("Saved model to disk.")

    def load(self):
        self.model = load_model(self.model_path)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        classes = ['Low', 'High']
        predictions = self.model.predict(x_test)
        fig_path = os.path.join(os.path.dirname(self.model_path), "measures.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        fig = plt.figure()

        matrix = confusion_matrix(y_test.argmax(axis = 1), predictions.argmax(axis = 1))
        sns.heatmap(matrix, annot = True, cbar = True, xticklabels = classes, yticklabels = classes, cmap = 'Blues', fmt = 'g')
        plt.ylabel('True Match Quality')
        plt.xlabel('Predicted Match Quality')
        pdf.savefig(fig)
        plt.close()

        probabilities = self.model.predict(x_test)
        probabilities = probabilities[:, 1]
        area = roc_auc_score(y_test.argmax(axis = 1), probabilities)
        fp, tp, _ = roc_curve(y_test.argmax(axis = 1), probabilities)
        fig = plt.figure()
        plt.plot(fp, tp, marker = '.', label = 'ROC (area = {:.3f})'.format(area))
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc = 'best')
        pdf.savefig(fig)
        plt.close()

        pdf.close()

        score = self.model.evaluate(x_test, y_test, verbose = 0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))
