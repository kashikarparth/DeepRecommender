# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:33:44 2018

@author: Parth
"""

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import RMSprop

model1 = Sequential()
model1.add(Dense(1024, kernel_initializer='lecun_uniform', input_shape=(16,)))
model1.add(Activation('selu'))
model1.add(Dropout(0.9))
model1.add(Dense(512, kernel_initializer='lecun_uniform'))
model1.add(Activation('selu'))
model1.add(Dropout(0.9))
model1.add(Dense(256, kernel_initializer='lecun_uniform'))
model1.add(Activation('selu'))
model1.add(Dropout(0.9))
model1.add(Dense(512, kernel_initializer='lecun_uniform'))
model1.add(Activation('linear')) 
model1.add(Dense(1024, kernel_initializer='lecun_uniform'))
model1.add(Activation('linear'))
rms = RMSprop()
model1.compile(loss='mmse', optimizer=rms)

model2 = Sequential()
model2.add(Dense(1024, kernel_initializer='lecun_uniform', input_shape=(16,)))
model2.add(Activation('selu'))
model2.add(Dropout(0.9))
model2.add(Dense(512, kernel_initializer='lecun_uniform'))
model2.add(Activation('selu'))
model2.add(Dropout(0.9))
model2.add(Dense(256, kernel_initializer='lecun_uniform'))
model2.add(Activation('selu'))
model2.add(Dropout(0.9))
model2.add(Dense(128, kernel_initializer='lecun_uniform'))
model2.add(Activation('selu'))
model2.add(Dropout(0.9))
model2.add(Dense(512, kernel_initializer='lecun_uniform'))
model2.add(Activation('linear')) 
model2.add(Dense(1024, kernel_initializer='lecun_uniform'))
model2.add(Activation('linear'))
rms = RMSprop()
model2.compile(loss='mmse', optimizer=rms)  

