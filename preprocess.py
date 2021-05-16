from sklearn import preprocessing
from data import *
import numpy as np

def normalize(data):
    preprocess_data = []
    scaler = preprocessing.StandardScaler()
    data = np.array(data)

    for rows in range(data.shape[0]):
        preprocess_data.apend(scaler.fit_transform(data[rows]))
    
    return preprocess_data
