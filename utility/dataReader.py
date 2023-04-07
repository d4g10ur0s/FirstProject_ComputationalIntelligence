from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import Input
import tensorflow as tf

import pandas as pd
import numpy as np

import math

import os

def datasetToFolds(data):
    # shuffle data
    data = data.sample(n=len(data),axis=0,ignore_index=True)
    #splice data to 5-fold
    fnum = int(input("Number of Folds : "))
    ffold = []
    batchlen = math.floor(len(data)/fnum)
    rem = len(data) - batchlen * fnum
    for i in range(fnum):
        if(i==fnum-1):
            try :
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen+rem][:].reset_index().drop(axis=1,labels=["index"]))
            except:
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen+rem][:].reset_index())
        else:
            try :
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen][:].reset_index().drop(axis=1,labels=["index"]))
            except:
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen][:].reset_index())
    return ffold

def data_reader():
    print("+mvainei")
    # read csv
    data = pd.read_csv(os.getcwd()+"\\dataset.csv", delimiter=";",low_memory = False)
    data.replace({"sitting" :1, "walking" :2, "standing":3, "standingup":4, "sittingdown":5},inplace=True)
    # data preprocess
    # x,y,z in [-617, 533]
    j=0
    data_coordinates = []
    for i in range(math.floor(len(data.index))):
        try :
            data_coordinates.append(data.loc[i][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]].astype('int64'))
            j+=1
        except :
            data.drop(axis=0,index=j,inplace=True)
    data_coordinates = pd.DataFrame(data=data_coordinates,columns = ["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"] )
    data_coordinates = (data_coordinates + 617)/(617+533)#data in [0,1]
    data_coordinates = data_coordinates - ((533 + 617)/2)/(617+533)#data in (-1,1)
    data_coordinates = pd.concat([data_coordinates , data.iloc[:math.floor(len(data.index))]["class"]] , axis=1 , join="outer")
    # save to csv file
    data_coordinates.to_csv(path_or_buf="F:\\5oEtos\\EarinoEksamhno\\YpologistikhNohmosunh\\Project_A\\utility\\processedDataset.csv", sep=';')

    return data_coordinates

def main():
    class_names = ["sitting", "walking", "standing", "standingup", "sittingdown"]
    class_namesDict = {"sitting" :1, "walking" :2, "standing":3, "standingup":4, "sittingdown":5}
    data = None
    data_coordinates = None
    # read csv
    if os.path.exists(os.getcwd() + "\\processedDataset.csv"):
        data_coordinates = pd.read_csv(os.getcwd()+"\\processedDataset.csv", delimiter=";",low_memory = False)
    else:
        data_coordinates = data_reader()
    # shuffle data
    data = datasetToFolds(data_coordinates)
    # test data
    test_data = np.array(data[4].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
    labels = data[4].iloc[:]["class"]/5
    # training data
    trainingData = pd.concat(data[:4], axis=0,join="inner")
    # Define the model
    model = Sequential()
    model.add(Dense(4,activation='relu',))
    model.add(Dense(1 ,activation=tf.keras.activations.sigmoid))
    # Compile the model tf.keras.losses.MeanSquaredError()
    model.compile(loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM), optimizer=tf.keras.optimizers.SGD(learning_rate=0.015,),metrics=["accuracy"])
    x = trainingData.iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]]
    y = trainingData.iloc[:]["class"]/5
    #print(tf.keras.activations.sigmoid(tf.constant(x.loc[0][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])))
    model.fit(x, y, epochs=2, batch_size=1)
    #model.fit(tf.constant(trainingData.loc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]],shape=(len(trainingData.index),4,3)), trainingData.loc[:]["class"]/5, epochs=2, batch_size=1)
    model.save('my_model')
    model.summary(show_trainable=True,)
    '''
    model= tf.keras.models.load_model('my_model')
    '''
    #predictions=model.predict(test_data.reshape(len(data[4].loc[:].drop(axis=1,labels=["class"]).index),4,3))
    predictions=model.predict(test_data)
    print(predictions)
    pr = []
    mcounter = 0
    test_data = data[4].iloc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]]
    for i in predictions:
        pr.append(i[0])
    results = pd.concat([test_data, labels , pd.DataFrame(data = {"results" : pr}) ], axis=1 , join="outer")
    print(results)


if __name__ == "__main__":
    main()
