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
    model.add(Dense(1 ,activation=tf.keras.activations.softmax))
    # Compile the model tf.keras.losses.MeanSquaredError()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,),metrics=["accuracy"])
    #model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,),metrics=["mean_squared_error"])
    for i in range(5):
        # test data
        test_data = np.array(data[i].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
        labels = data[i].iloc[:]["class"]
        # training data
        trainingData = pd.concat(data[:i]+data[1+i:], axis=0,join="inner")
        x = trainingData.iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]]
        y = trainingData.iloc[:]["class"]
        print(pd.get_dummies(y.iloc[:]))
        #print(tf.keras.activations.sigmoid(tf.constant(x.loc[0][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])))
        model.fit(x, pd.get_dummies(y.iloc[:]), epochs=2, batch_size=1)
        model.summary(show_trainable=True,)
        #model.fit(tf.constant(trainingData.loc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]],shape=(len(trainingData.index),4,3)), trainingData.loc[:]["class"], epochs=2, batch_size=1)
    '''
    model= tf.keras.models.load_model('attempt_1')
    '''
    test_data = np.array(data[4].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
    labels = data[4].iloc[:]["class"]
    model.save('attempt_1_3')
    '''
    '''
    #predictions=model.predict(test_data.reshape(len(data[4].loc[:].drop(axis=1,labels=["class"]).index),4,3))
    predictions=model.predict(test_data)
    test_data = data[4].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]]
    results = pd.concat([test_data, labels , pd.DataFrame(data = {"results" : pr}) ], axis=1 , join="outer")
    print(results[["class","results"]])

if __name__ == "__main__":
    main()
