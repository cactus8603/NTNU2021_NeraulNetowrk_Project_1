# -*- coding: utf-8 -*-
"""
# Project#1 Keras Tutorial: Stock prediction

2021/3/10 Neural Network

For your references:

*   [Keras official website](https://keras.io/)

*   [Google Colab official tutorial](https://colab.research.google.com/notebooks/welcome.ipynb?hl=zh-tw#scrollTo=gJr_9dXGpJ05)

*   [Using outer files in Google colab](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=BaCkyg5CV5jF)

# (OPTIONAL) Step 0: Upload dataset

參閱[Using outer files in Google colab](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=BaCkyg5CV5jF)。
"""
"""
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
"""
# 按下選擇檔案鍵，待上傳完成即可使用。（注意檔名變化）
# 每次開啟colab都會淨空上傳檔案，如有變動資料且往後仍須使用，請保存檔案。
# Click the bottom below and select file, file is avaliable after uploaded. (Notice the file name may change.)
# Google colab will clean up the file upload every browser session. If you modified the file and will use it again, remember to save your file.

import numpy as np

# 讀取檔案 
# Read files
data = np.genfromtxt('6269.csv', delimiter=',', dtype=None)[1:]
print('Num of samples:', len(data))


# (OPTIONAL)
# 保存每一行到變數裡
# Save all the columns to variables
date = data[:, 0] # data的第一個欄位 
                  # the first column of data
open = data[:, 1]
high = data[:, 2]
low  = data[:, 3]
close= data[:, 4]


prices = np.array([close for date, open, high, low, close in data]).astype(np.float64) #

# (OPTIONAL)
# 視覺化資料集
# Visualize dataset

import matplotlib.pyplot as plt

plt.plot(range(0, len(data)), prices)
plt.show()

# 將訓練跟測試資料分開       
# Divide trainset and test set
# 這邊切成 8(訓練): 2(測試) 
# Ration of samples 8(for training): 2(for testing)
train_size = int(len(prices) * 0.8)
train, test = prices[:train_size], prices[train_size:]

def transform_dataset(dataset, look_back=5):
    # 前 N 天的股價 
    # N days as training sample
    dataX = [dataset[i:(i + look_back)]
             for i in range(len(dataset) - look_back-1)]
    # 第 N 天的股價 
    # 1 day as groundtruth
    dataY = [dataset[i + look_back]
             for i in range(len(dataset) - look_back-1)]
    return np.array(dataX), np.array(dataY)

# 設定 N(look_back)=5
# Set the N(look_back)=5
look_back = 5
trainX, trainY = transform_dataset(train, look_back)
testX, testY = transform_dataset(test, look_back)

# rint(trainX)
# print(trainY)

# print(testX)
# print(testY)

from keras.models import Sequential
from keras.layers import Dense

# 宣告一個 Sequential 模型
# Define a Sequential model
model = Sequential()

# input_dim = 輸入資料的維度
# input_dim = the dimension of input data
# 給 hidden layer 設定 input_dim = 5
# Set the input_dim of hidden layer as 5
model.add(Dense(units=32, activation='relu', input_dim=look_back))
model.add(Dense(units=32, activation='relu', input_dim=look_back))
model.add(Dense(units=32, activation='relu', input_dim=look_back))
model.add(Dense(units=32, activation='relu', input_dim=look_back))
# 給模型加入 output layer: 只有1個 neuron 的 dense layer
# Add an output layer (dense layer with only 1 neuron) to the model
model.add(Dense(units=5))

# 設定 loss function
# Set loss function
model.compile(loss='mse', optimizer='adam')

# 訓練資料
# Set the training / test data to model
train_history = model.fit(trainX, trainY, epochs=100, batch_size=32, 
                          shuffle=False, verbose=2, validation_data=(testX, testY))

import matplotlib.pyplot as plt

# 視覺化 loss
# Visualize loss
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

plt.subplot(121)
plt.plot(loss)
plt.subplot(122)
plt.plot(val_loss)
plt.show()
plt.savefig('1.jpg')

# 保存本次訓練的模型跟參數
# Save model parameters of this training
model.save('stock_DNN_model.h5')
# files.download('stock_DNN_model.h5') # for colab

from keras.models import load_model

# 載入模型
# Load model
model = load_model('stock_DNN_model.h5')

# 測試模型
# Test model
test = model.evaluate(testX, testY, batch_size=5)

# 顯示模型預測的結果與實際結果的誤差 (loss)
# Show the difference between predict and groundtruth (loss)
print('Test Result: ', test)

# 將未知的資料用在已經訓練好的模型上
# Predict the future price with trained model
predict = model.predict([[123, 123.5, 124, 123, 123.5]])
print('Predict Result', predict)
