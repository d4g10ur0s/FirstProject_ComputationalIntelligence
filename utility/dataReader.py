from keras.models import Sequential
from keras.layers import Dense

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
    # read csv
    dpath = os.getcwd()
    data = pd.read_csv(dpath+"\\dataset.csv", delimiter=";",low_memory = False)
    return data

def main():
    class_names = ["sitting", "walking", "standing", "standingup", "sittingdown"]
    class_namesDict = {"sitting" :1, "walking" :2, "standing":3, "standingup":4, "sittingdown":5}
    data = None
    data_coordinates = None
    # read csv
    if os.path.exists(os.getcwd() + "\\processedDataset.csv"):
        data_coordinates = pd.read_csv(os.getcwd()+"\\processedDataset.csv", delimiter=";",low_memory = False)
    else:
        data = pd.read_csv(os.getcwd()+"\\dataset.csv", delimiter=";",low_memory = False)
        data = data_reader()
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

    # shuffle data
    data = datasetToFolds(data_coordinates)
    # training data
    trainingData = pd.concat(data[:4], axis=0,join="inner")
    # Define the model
    model = Sequential()
    model.add(Dense(4, input_dim=3, activation='relu', input_shape=(4,3)))
    model.add(Dense(1 ,activation='softmax', input_shape=(1,4) ))
    # Compile the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.05,))
    model.fit(tf.constant(trainingData.loc[:][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]],shape=(len(trainingData.index),4,3)), trainingData.loc[:]["class"], epochs=5, batch_size=1)

    test_data = np.array(data[4].loc[:len(data[4].index)][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]])
    predictions=model.predict(np.array(data[4].loc[:].drop(axis=1,labels=["class"])).reshape(len(data[4].index),4,3))
    pr = []
    mcounter = 0
    for i in predictions:
        pr.append(i[0])
    results = pd.concat([data[4] , pd.DataFrame(data = {"results" : pr}) ], axis=1 , join="outer")
    print(results)


if __name__ == "__main__":
    main()
