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

def makePlot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['MSE'])
    plt.plot(history.history['val_MSE'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    class_names = ["sitting", "walking", "standing", "standingup", "sittingdown"]
    #class_namesDict = {"sitting" :1, "walking" :2, "standing":3, "standingup":4, "sittingdown":5}
    #class_namesDict = {"sitting" :[00001], "walking" :[00010], "standing":[00100], "standingup":[01000], "sittingdown":[10000]}
    data = None
    data_coordinates = None
    # read csv
    if os.path.exists(os.getcwd() + "\\utility\\processedDataset.csv"):
        data_coordinates = pd.read_csv(os.getcwd()+"\\utility\\processedDataset.csv", delimiter=";",low_memory = False)
    else:
        data_coordinates = data_reader()
    # shuffle data
    data = datasetToFolds(data_coordinates)
    # Define the model
    model = Sequential()
    model.add(Dense(4,activation='relu',))
    model.add(Dense(5 ,activation=tf.keras.activations.softmax))
    # Compile the model tf.keras.losses.MeanSquaredError()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,),metrics=["accuracy","MSE","categorical_crossentropy"])
    #model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,),metrics=["mean_squared_error"])
    history = None
    for i in range(5):
        # test data
        test_data = np.array(data[i].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
        labels = data[i].iloc[:]["class"]
        # training data
        trainingData = pd.concat(data[:i]+data[1+i:], axis=0,join="inner")
        x = trainingData.iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]]
        y = trainingData.iloc[:]["class"]
        test_data = np.array(data[i].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
        test_labels = data[i].iloc[:][["class"]]
        history = model.fit(x, pd.get_dummies(y.iloc[:]), epochs=5, batch_size=1,
                            validation_data = (test_data , pd.get_dummies(test_labels.iloc[:] , columns=['class'])) )
        makePlot(history)
        model.summary(show_trainable=True,)

    '''
    model= tf.keras.models.load_model('attempt_2_1')
    '''
    model.save('attempt_2_1')



if __name__ == "__main__":
    main()
