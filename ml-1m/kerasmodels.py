# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:33:44 2018

@author: Parth
"""

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import pandas as pd
import numpy as np
import scipy.sparse as sp

data = pd.read_csv("ratings.dat", sep='::', header=None, skiprows=0, engine = 'python')
group_data = data.groupby([0, 1]).sum().reset_index()
group_data = group_data.drop(3, axis = 1)
movies_lookup = pd.read_csv("movies.dat", sep = '::', header = None, skiprows = 0, engine = 'python')
users = list(np.sort(group_data[0].unique()))
products = list(np.sort(group_data[1].unique()))
rating = list(group_data[2])
rows = group_data[0].astype('category',users).cat.codes
columns = group_data[1].astype('category',products).cat.codes
sparse = sp.csr_matrix((rating, (rows, columns)), shape = (len(users), len(products)))      
#print(np.shape(sparse.toarray()))

def predict(x):
    model2 = load_model('NvidiaDeepRec.h5')
    return(np.round(model2.predict(x)))
        
if(0):
    model2 = Sequential()
    model2.add(Dense(512, kernel_initializer='lecun_uniform', input_shape=(3706,)))
    model2.add(Activation('selu'))
    model2.add(Dense(512, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dense(1024, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dropout(0.8))
    model2.add(Dense(512, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dense(512, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dense(3706, kernel_initializer='lecun_uniform'))
    rms = RMSprop()
    model2.compile(loss='mmse', optimizer=rms)
    model2.fit(x=sparse.toarray(),y=sparse.toarray(),batch_size=128,epochs = 20, verbose = 1) 
    model2.save('NvidiaDeepRec.h5')
    
if(0):
    model2 = load_model('NvidiaDeepRec.h5')
    model2.fit(x=sparse.toarray(),y=sparse.toarray(),batch_size=256,epochs = 10, verbose = 1) 
    model2.save('NvidiaDeepRec.h5')

if(0):                                            #DATA REFEEDING
    model2 = load_model('NvidiaDeepRec.h5')
    for i in range(6040):
        x1 = predict(sparse[i].toarray())
        model2.fit(x=x1,y=x1,batch_size=128,epochs = 1, verbose = 1)
    model2.save('NvidiaDeepRec.h5')
    
if(0):    
    print(sparse[8].toarray())
    print(predict(sparse[8].toarray()))

if(0):
    sum = 0
    ssum = 0
    for i in range(10):
        temp = predict(sparse[i].toarray())[0]
        for j in range(3706):
            if(sparse[i].toarray()[0][j]!=0):
                sum+=1
                ssum = (temp[j]-sparse[i].toarray()[0][j])**2
    print((ssum/sum)**0.5)
            
            
