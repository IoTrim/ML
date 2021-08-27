from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.metrics import CategoricalAccuracy
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import tensorflow as tf
import sys
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import random as rn

class NetworkTrainer():

    def __init__(self, f, randomSeed):
        np.random.seed(randomSeed)
        rn.seed(randomSeed)
        tf.random.set_seed(randomSeed)
        self.databaseFile = "datasets/" + f

    def basic_architecture(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.inputDims, activation='relu'))
        model.add(Dense(64, activation='relu'))
        return model

    def plot_history(self, save = False):
        for metric in self.history.history.keys():
            plt.plot(self.history.history[metric])
            plt.title('Categorical Accuracy and Loss for ' + self.databaseFile[9:] + " dataset")
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.legend([k for k in self.history.history.keys()], loc='upper right')
            if save:
                plt.savefig("images/" + self.databaseFile[9:] + "_history.png")
        return plt

    def plot_confusion_matrix(self, save = False):
        cm = confusion_matrix(self.y_test.argmax(axis=1), self.testPred.argmax(axis=1))
        df_cm = pd.DataFrame(cm, self.labels, self.labels)
        plt.figure(figsize=(20,14))
        plt.title("Confusion Matrix for " + self.databaseFile[9:] + " dataset")
        sn.heatmap(df_cm, annot=True)
        if save:
            plt.savefig("images/" + self.databaseFile[9:] + "_confusion_matrix.png")
        return plt

    def process_data(self, sampling = "under", testSize = 0.4):
        # read data from csv
        df = pd.read_csv(self.databaseFile, index_col=0)
        df = df.sample(frac=1).reset_index(drop=True) # shuffle dataset
        df = df.dropna()

        dataset = df.values
        X = dataset[:,1:].astype(float) # using all features
        y = dataset[:,0]

        # Encode labels
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)

        # convert integers to dummy variables (i.e. one hot encoded)
        categorical_y = np_utils.to_categorical(encoded_Y)
        self.numLabels = len(set(y))
        
        # scale X values
        min_max_scaler = MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, categorical_y, test_size=testSize)

        # Train val split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        # sampling 
        ran = {"random_state":42}
        sampler = RandomOverSampler(**ran) if sampling == "over" else RandomUnderSampler(**ran)

        X_train, y_train = sampler.fit_resample(X_train, y_train)
        X_test, y_test = sampler.fit_resample(X_test, y_test)

        self.inputDims = X.shape[1]
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.labels = np.unique(y)

    def create_model(self):
        # create model
        self.model = self.basic_architecture()
        self.model.add(Dense(self.numLabels, activation='softmax'))

        self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


    def train_model(self, epochs=15):
        # early stopping criteria
        es = EarlyStopping(monitor='val_categorical_accuracy', 
                        mode='max',
                        verbose=0, 
                        patience=4)

        # fit model
        self.history = self.model.fit(self.X_train,
                            self.y_train,
                            epochs=epochs,
                            batch_size=25, 
                            verbose=0, 
                            validation_data=(self.X_val, self.y_val),
                            callbacks=[es])

    def evaluate_model(self):
        # evaluate model
        self.testPred = self.model.predict(self.X_test)
        self.loss, self.categorical_accuracy = self.model.evaluate(self.X_test, self.y_test)
        return self.loss, self.categorical_accuracy


if __name__ == '__main__':
    NN = NetworkTrainer(sys.argv[1], 43)
    NN.process_data()
    NN.create_model()
    NN.train_model()
    loss, categorical_accuracy = NN.evaluate_model()
    NN.plot_history(save = True)
    NN.plot_confusion_matrix(save = True)
    print("Loss =", loss)
    print("Categorical Accuracy =", categorical_accuracy)