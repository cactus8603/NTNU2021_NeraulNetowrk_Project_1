# import twstock
import pandas as pd
import numpy as np
from numpy import zeros, newaxis
# twstock.__update_codes()

# use twstock
"""
def getdata(year, month):
    stock = twstock.Stock('6269')
    return stock.fetch_from(2021, 3)

"""
# use data form moodle

def getdata(lookforward):
    df = pd.read_csv('D://others//桌面//project//NTNU2021_NeraulNetowrk_Project_1//Project1_stock_0326//6269.csv')  
    
    data = np.array(df[:4338])

    # remove date
    data = np.array(data[:, 1:5])

    # normalization
    # norm = np.linalg.norm(data)
    # data = data/norm

    # getdata
    op = np.array(data[:, 0])
    h = np.array(data[:, 1])
    low = np.array(data[:, 2])
    close = np.array(data[:, 3])
    # return close
    # print(close)
    # calculate upp and lw
    
    temp_upper = np.array([h-op, h-close])
    upp = np.amax(temp_upper, axis=0)
    upp = np.reshape(upp, (len(upp),1))

    temp_lw = np.array([low-op, low-close])
    lw = np.amax(temp_lw, axis=0)
    lw = np.reshape(lw, (len(lw),1))

    # combine data with upp, lw
    data = np.hstack((data, upp, lw))

    # replace date with yesterday close price
    data = np.delete(data, (0,len(data)-1), axis=0)
    temp_close = np.reshape(close, (len(close), 1))

    target = np.delete(temp_close, (0,1), axis=0)
    temp_close = np.delete(temp_close, (len(close)-1, len(close)-2), axis=0)

    data = np.hstack((temp_close, data))
    close = np.delete(close, 0)
    length = len(data)

    data = data[newaxis, :, :]
    # b = a[:, :, newaxis]
    # print("data " , data.shape[1])
    data = np.reshape(data, (data.shape[1], 1, 7))
    #data = np.reshape(data, (867, 5, 7))
    #temp_close = np.reshape(close, (867, 5))
    return  data.astype(float), target.astype(float)
    

if __name__ == '__main__':
    x, y = getdata(32)
    # print(x.shape)
    # print(x)
    # print(y.shape)
    # print(y)
   
    
"""
df = pd.read_csv('D://others//桌面//project//NTNU2021_NeraulNetowrk_Project_1//Project1_stock_0326//6269.csv')  

data = np.array(df[4318:])

data = np.array(data[:, 1:5])

op = np.array(data[:, 0])
h = np.array(data[:, 1])
low = np.array(data[:, 2])
close = np.array(data[:, 3])
temp_upper = np.array([h-op, h-close])
upp = np.amax(temp_upper, axis=0)
upp = np.reshape(upp, (len(upp),1))

temp_lw = np.array([low-op, low-close])
lw = np.amax(temp_lw, axis=0)
lw = np.reshape(lw, (len(lw),1))

data = np.hstack((data, upp, lw))

data = np.delete(data, 0, axis=0)
temp_close = np.reshape(close, (len(close), 1))
temp_close = np.delete(temp_close, len(close)-1, axis=0)

data = np.hstack((temp_close, data))
print(data)
print(close)
"""



