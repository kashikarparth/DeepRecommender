# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:33:44 2018

@author: Parth
"""
# IMPORTS
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import pandas as pd
import numpy as np
import scipy.sparse as sp
############################################################################################

#DATA PREPROCESSING
data = pd.read_csv("ratings.dat", sep='::', header=None, skiprows=0, engine = 'python')
group_data = data.groupby([0, 1]).sum().reset_index()
group_data = group_data.drop(3, axis = 1)
users = list(np.sort(group_data[0].unique()))
products = list(np.sort(group_data[1].unique()))
rating = list(group_data[2])
rows = group_data[0].astype('category',users).cat.codes
columns = group_data[1].astype('category',products).cat.codes
sparse = sp.csr_matrix((rating, (rows, columns)), shape = (len(users), len(products)))      
#print(np.shape(sparse.toarray()))
#############################################################################################

#HELPER FUNCTIONS 
def predict(x):
    model2 = load_model('NvidiaDeepRec4.h5')
    return(model2.predict(x))
    
def gen(x,y):                                           
    sparse1 = []
    for i in range(6039, -1, -1):
        sparse1.append(np.round(y.predict(x[i].toarray())))
        if(i%1000==0):
            print("customers done: ", 6040-i)
    sparse1 = np.reshape(sparse1,newshape=[-1,3706])
    np.save('predicted_vals.npy',sparse1)
#############################################################################################
if(0):                                                      #MODEL TWEAK-TESTS
    model2 = Sequential()
    model2.add(Dense(128, kernel_initializer='lecun_uniform', input_shape=(3706,)))
    model2.add(Activation('selu'))
    model2.add(Dense(128, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dense(128, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dropout(0.8))
    model2.add(Dense(128, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dense(128, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dense(128, kernel_initializer='lecun_uniform'))
    model2.add(Activation('selu'))
    model2.add(Dense(3706, kernel_initializer='lecun_uniform'))
    rms = RMSprop()
    model2.compile(loss='mmse', optimizer=rms)
    model2.fit(x=sparse.toarray(),y=sparse.toarray(),batch_size=128,epochs = 50, verbose = 1) 
    model2.save('NvidiaDeepRec4.h5')
    
if(0):                                                      #MODEL RETRAINER
    model2 = load_model('NvidiaDeepRec4.h5')
    model2.fit(x=sparse.toarray(),y=sparse.toarray(),batch_size=256,epochs = 50, verbose = 1) 
    model2.save('NvidiaDeepRec4.h5')

if(0):
    model2 = load_model('NvidiaDeepRec4.h5')
    gen(sparse, model2)                                                    #DATA REFEEDING
    sparse1 = np.load("predicted_vals.npy")
    model2.fit(x=sparse1,y=sparse1,batch_size=256,epochs = 30, verbose = 1) 
    model2.save('NvidiaDeepRec4.h5')
    
if(0):                                                      #OUTPUT CHECKER
    print(sparse[8].toarray())
    print(predict(sparse[8].toarray()))

if(1):                                                      #RMSE CALCULATOR
    model2 = load_model('NvidiaDeepRec4.h5')
    gen(sparse, model2)
    sparse1 = np.load("predicted_vals.npy")
    sparse2 = sparse.toarray()
    num = np.count_nonzero(sparse2)
    sparsetemp = np.multiply(sparse2,sparse1)
    sparse1 = np.divide(sparsetemp,sparse2, where=sparse2!=0)
    sparsefinal = np.sum(np.square(np.subtract(sparse2,sparse1)))*(1/num)
    print(np.sqrt(sparsefinal))
##############################################################################################