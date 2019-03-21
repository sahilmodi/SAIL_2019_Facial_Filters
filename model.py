from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

''' Load the facial keypoints data. 
    Return: training face images with corresponding key points. ''' 
def load_data():
    # load & read data
    fname = 'data/training.csv'
    df = pd.read_csv(os.path.expanduser(fname))  
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # drop missing values
    df = df.dropna() 

    # normalize
    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)
    
    # reshape (96, 96, 1)
    X = X.reshape(-1, 96, 96, 1) 

    y = df[df.columns[:-1]].values

    # normalize
    y = (y - 48) / 48  

    X, y = shuffle(X, y, random_state=42)  
    y = y.astype(np.float32)

    return X, y


''' This class builds a simple convolutional neural network. 
    Predicts facial keypoints, given an image of a face. '''
class CNNModel:
    def __init__(self, filename):
        # load model if exists
        # else, train new model
        self.filename = filename
        if os.path.exists(self.filename):
            self.model = load_model(self.filename)
        else:
            self.build_model()
    
    ''' Constructs a new convolutional neural network, and
        saves as a .h5 file. ''''
    def build_model(self): 
        self.model = Sequential()
        self.model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.1))

        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Convolution2D(30, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(30))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        x_train, y_train = load_data()
        self.model.fit(x_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)
        
        self.model.save(self.filename)

    ''' Predict keypoints, given a face image. '''
    def predict(self, face): 
        return self.model.predict(face)
