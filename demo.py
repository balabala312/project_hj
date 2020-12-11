# # import tensorflow as tf
# #
# # A = tf.constant([[1, 2], [3, 4]])
# # B = tf.constant([[5, 6], [7, 8]])
# # C = tf.matmul(A, B)
# #
# # print(C)
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import math
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout, LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from keras import optimizers
# import time
#
# def creat_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i: (i+look_back)]
#         dataX.append(a)
#         dataY.append(dataset[i+look_back])
#     return np.array(dataX), np.array(dataY)
# path = "D:\project\project2"
# dataframe = pd.read_csv(path+'/zgpa_train.csv',
#                         header=0, parse_dates=[0],
#                         index_col=0, usecols=[0, 5], squeeze=True)
# dataset = dataframe.values
# dataframe.head(10)
#
# data = pd.read_csv('D:\\project\\project2\\dataset\\Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv',encoding='utf-8')
# data = data.groupby('Vehicle_ID')
#

import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a,b)
print(c)
print(c)
print(c)
print(c)

