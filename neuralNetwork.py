from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import Input
import tensorflow as tf
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import math

import os

from utility.dataReader import data_reader
from utility.dataReader import datasetToFolds

accuracy = []
mse = []
ce = []

def makePlot(history):
    global accuracy
    global mse
    global ce
    input(str(0.5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    ################################################
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('model loss ( CE )')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    ################################################

    plt.plot(history.history['MSE'])
    plt.plot(history.history['val_MSE'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('model MSE')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    accuracy.append(history.history.get('accuracy')[-1])
    mse.append(history.history.get('MSE')[-1])
    ce.append(history.history.get('loss')[-1])


def main():
    global accuracy
    global mse
    global ce

    data = None
    data_coordinates = None
    # read csv
    if os.path.exists(os.getcwd() + "\\utility\\processedDataset.csv"):
        data_coordinates = pd.read_csv(os.getcwd()+"\\utility\\processedDataset.csv", delimiter=";",low_memory = False)
    else:
        data_coordinates = data_reader()
    # shuffle data
    data = datasetToFolds(data_coordinates)
    history = None
    for i in range(5):
        tf.keras.backend.clear_session()
        # Define the model
        model = Sequential()
        model.add(Dense(17,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.5),))
        model.add(Dense(5 ,activation=tf.keras.activations.softmax))
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.6),
                      metrics=["accuracy","MSE","categorical_crossentropy"])
        # test data
        test_data = np.array(data[i].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
        labels = data[i].iloc[:]["class"]
        # training data
        trainingData = pd.concat(data[:i]+data[1+i:], axis=0,join="inner")
        x = trainingData.iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]]
        y = trainingData.iloc[:]["class"]
        test_data = np.array(data[i].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
        test_labels = data[i].iloc[:][["class"]]
        # early stopping
        callback = tf.keras.callbacks.EarlyStopping(
                                        monitor="loss",
                                        min_delta=1e-2,
                                        patience=2,
                                        verbose=1,
                                        )
        # train the models
        history = model.fit(x, pd.get_dummies(y.iloc[:]), epochs=15, batch_size=1,callbacks=[callback],
                            validation_data = (test_data , pd.get_dummies(test_labels.iloc[:] , columns=['class'])) )
        makePlot(history)
        model.summary(show_trainable=True,)

    '''
    model= tf.keras.models.load_model('attempt_2_1')
    '''
    model.save('attempt_2_2')
    a = 0
    b = 0
    c = 0
    for i in range(5):
        a+=accuracy[i]
        b+=mse[i]
        c+=ce[i]
    print("Accuracy : " + str(a/5))
    print("MSE : " + str(b/5))
    print("Cross Entropy : " + str(c/5))


if __name__ == "__main__":
    main()
