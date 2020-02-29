import seaborn as sns
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

    def train(self, x, y, rep):
        seed = 7
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
        all_scores = []
        for train, test in kfold.split(x, y.argmax(axis = 1)):
            self.model = Sequential()
            kreg = self.kernel_reg
            breg = self.bias_reg
            self.model.add(Dense(1000, input_dim=2000, activation='sigmoid'))
            self.model.add(Dense(100, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
            self.model.add(Dense(10, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
            self.model.add(Dense(2, kernel_regularizer=l2(kreg), bias_regularizer=l2(breg), activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.summary()
            history = self.model.fit(x[train], y[train], epochs = rep, batch_size = 10, verbose = 0)
            scores = self.model.evaluate(x[test], y[test], verbose = 0)
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
            all_scores.append(scores[1] * 100)
        print('%.2f%% (+/- %.2f%%)' % (np.mean(all_scores), np.std(all_scores)))

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
        classes = ['Low', 'High']
        predictions = self.model.predict(x_test)
        fig_path = os.path.join(os.path.dirname(self.model_path), "measures.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        fig = plt.figure()

        matrix = confusion_matrix(y_test.argmax(axis = 1), predictions.argmax(axis = 1))
        sns.heatmap(matrix, annot = True, cbar = True, xticklabels = classes, yticklabels = classes, cmap = 'Blues')
        plt.ylabel('True Match Quality')
        plt.xlabel('Predicted Match Quality')
        pdf.savefig(fig)
        plt.close()

        probabilities = self.model.predict_proba(x_test)
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

        score = self.model.evaluate(x_test, y_test)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))
