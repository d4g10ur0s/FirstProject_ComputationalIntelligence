
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
            ffold.append(data.iloc[i*batchlen:(i+1)*batchlen+rem][:].reset_index())
        else:
            ffold.append(data.iloc[i*batchlen:(i+1)*batchlen][:].reset_index())
    return ffold

def data_reader():
    # read csv
    dpath = os.getcwd()
    data = pd.read_csv(dpath+"\\dataset.csv", delimiter=";",low_memory = False)
    return data

def main():
    # read csv
    data = data_reader()
    # data preprocess
    # x,y,z in [-617, 533]
    j=0
    data_coordinates = []
    for i in range(math.floor(len(data.index))):
        data_coordinates.append(data.loc[i][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]].astype('int64'))
    data_coordinates = pd.DataFrame(data=data_coordinates,columns = ["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"] )
    data_coordinates = (data_coordinates + 617)/(617+533) - (-617+533)/2
    data_coordinates = pd.concat([data_coordinates , data.iloc[:]["class"]] , axis=1 , join="outer")
    print(str(data_coordinates))
    # shuffle data
    #data = datasetToFolds(data)
    #for i in data :
    #    print(str(i))

if __name__ == "__main__":
    main()
