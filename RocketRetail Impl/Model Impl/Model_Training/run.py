'''Imports and Global Variables for loading in dependencies and flexibilty in model generation
    The optimizers and action function strings are to be used subsequently with the "init_model" function'''

import numpy as np
import tflearn
from tensorflow import reset_default_graph
import scipy.sparse as sp
               
np.random.seed(1) #Fixing the random seed to generate the same train and test data over multiple runtimes

rms = tflearn.optimizers.RMSProp()
sgd = tflearn.optimizers.SGD()
adam = tflearn.optimizers.Adam()
adagrad = tflearn.optimizers.AdaGrad()

selu = 'selu'
relu = 'relu'
elu = 'elu'
sigmoid = 'sigmoid'
tanh = 'tanh'

#Whenever using model.save() function, edit the following path to the desired location
PATH_TO_TRAINED_MODEL = "/media/parth/D:/Parth Kashikar/AI/CoutureWork/DeepRecommender/Rocket Retail Impl/Model Impl/Trained_Models/" 

############################################################################################

def delete_row_csr(mat, i):

    '''This function deletes an entire row from an inputted CSR sparse matrix in-place at a given row-index

        Parameters
        -------
        mat: CSR sparse matrix
            The matrix from which the row has to be deleted

        i : int32
            The row number of the row to be deleted from the matrix

        Returns
        -------
        None (all operations are in-place)'''

    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

def data_preprocessing():

    '''This function loads in the sparse matrix and filters out customers on the basis of a minimum allowed number of interactions,
        and then from this filters out items on the basis of a minimum allwed number of interactions.

        Parameters
        -------
        None

        Returns
        -------
        final: NumPy array
            The final processed matrix after filtering customers and items'''


    MIN_CUSTOMER_INTERACTIONS = 5 #Minimum allowed customer interactions
    MIN_ITEM_INTERACTIONS = 5 #Minimum allowed item interactions

    #Paths to multiple datasets and arrays, change accordingly
    PATH_TO_SPARSE_MATRIX = "/media/parth/D:/Parth Kashikar/AI/CoutureWork/DeepRecommender/Rocket Retail Impl/Model Impl/Dataset and processed Data/sparse.npz"
    PATH_TO_CUSTOMER_FILTER = "/media/parth/D:/Parth Kashikar/AI/CoutureWork/DeepRecommender/Rocket Retail Impl/Model Impl/Dataset and processed Data/customer_filter.npz"
    PATH_TO_ITEM_FILTER = "/media/parth/D:/Parth Kashikar/AI/CoutureWork/DeepRecommender/Rocket Retail Impl/Model Impl/Dataset and processed Data/item_filter.npz"

    sparse_data = sp.load_npz(PATH_TO_SPARSE_MATRIX) 
    array_temp = np.ones(shape=[1,sparse_data.shape[1]]) 
    customer_filter = sp.csr_matrix(array_temp,shape=(1,sparse_data.shape[1])) 
    non_zero = sparse_data.getnnz(1) 
    ls = np.arange(sparse_data.shape[0]) 
    ls = ls[np.where(non_zero>MIN_CUSTOMER_INTERACTIONS)]

    for i in range(len(ls)):
        customer_filter = sp.vstack(blocks = [customer_filter,sparse_data.getrow(ls[i])])
    delete_row_csr(customer_filter,0)

    sp.save_npz(PATH_TO_CUSTOMER_FILTER,customer_filter)
    customer_filter = sp.load_npz(PATH_TO_CUSTOMER_FILTER.tocsc())
    non_zero = customer_filter.getnnz(0)
    ls = np.arange(customer_filter.shape[1])
    ls = ls[np.where(non_zero>MIN_ITEM_INTERACTIONS)]
    item_filter = sp.csc_matrix(np.ones(shape=[customer_filter.shape[0],1]),shape=(customer_filter.shape[0],1))
    for i in range(len(ls)):
        item_filter = sp.hstack(blocks = [item_filter,customer_filter.getcol(ls[i])])
    sp.save_npz(PATH_TO_ITEM_FILTER,item_filter)
    final_matrix = sp.load_npz(PATH_TO_ITEM_FILTER)

    print("Data Preprocessing almost complete")
    final = final_matrix.toarray()
    return final

def init_data():   

    '''This function intializes training and testing data (as the random seed is set, the generated data is the same every time) 
        It needs no parameters, and returns nothing. The commented out portion is essential for first time use-case only

        Parameters
        -------
        None

        Returns
        -------
        None''' 

    #Change accordingly, the following path to training data
    PATH_TO_TRAIN_DATA = "/media/parth/D:/Parth Kashikar/AI/CoutureWork/DeepRecommender/Rocket Retail Impl/Model Impl/Dataset and processed Data/train_data.npy"

    #train_data = data_preprocessing()
    #train_data = train_data[:,np.any(train_data!=0,axis = 0)]
    #np.place(train_data,train_data==3,2)
    #np.place(train_data,train_data==5,4)
    #np.place(train_data,train_data==6,4)
    #np.save(PATH_TO_TRAIN_DATA,train_data)

    train_data = np.load(PATH_TO_TRAIN_DATA)
    rand_idx = np.random.randint(26985,size=5000)
    global test_data = train_data[rand_idx,:]
    global train_data = np.delete(train_data,rand_idx,0)
    print("Train and Test data made")

def init_model(activation_function = 'selu' ,optimizer = rms, dropout_keep_prob = 0.35):

    '''This function initializes the neural network model with tflearn, with 9 hidden layers and specific hidden node values. It should be called before any
        training or testing. The activation, optimizer technique and dropout rates can be changed by inputting as parameters


        Parameters
        -------
        activation_function : String (predefined in global variables)
            The desired activation function to be used for the hidden nodes across the model (possible cases : selu,relu,elu,sigmoid and tanh)

        optimizer : tflearn.optimizers method
            The desired optimization technique to be used for training the model (possible cases : rms,sgd,adam and adagrad)

        Returns
        -------
        None'''


    reset_default_graph()                                                       
    net = tflearn.input_data(shape=[None, np.shape(train_data)[1]])
    net = tflearn.fully_connected(net, 128,activation=activation_function)
    net = tflearn.fully_connected(net, 128,activation=activation_function)
    net = tflearn.fully_connected(net, 128,activation=activation_function)
    net = tflearn.fully_connected(net, 256,activation=activation_function)
    net = tflearn.fully_connected(net, 256,activation=activation_function)
    net = tflearn.dropout(net,dropout_keep_prob)
    net = tflearn.fully_connected(net, 256,activation=activation_function)
    net = tflearn.fully_connected(net, 256,activation=activation_function)
    net = tflearn.fully_connected(net, 128,activation=activation_function)
    net = tflearn.fully_connected(net, 128,activation=activation_function)
    net = tflearn.fully_connected(net, np.shape(train_data)[1])
    net = tflearn.regression(net,optimizer=optimizer,loss = 'masked_mse')
    global model = tflearn.DNN(net,tensorboard_verbose=0)


def train_model(batch_size = 512):

    '''This function trains the defined neural network for one epoch over the train_data for a given batch_size 


        Parameters
        -------
        batch_size : int32
            The batch size to be used for training the model

        Returns
        -------
        None'''   

    model.fit(train_data, train_data, n_epoch=1, batch_size=batch_size, show_metric=True)    
    data_refeeding()

def data_refeeding(batch_size = 512):

    '''This function is the Data Refeeding implementation inspired by the NVIDIA DeepRecommender where the neural network is trained on its own current output as input data to attain function stability
       for a given batch_size for one epoch

        Parameters
        -------
        batch_size : int32
            The batch size to be used for training the model

        Returns
        -------
        None'''

    print("Refeeding")
    sparse1 = model.predict(train_data)                                                
    model.fit(sparse1,sparse1,batch_size=batch_size,n_epoch = 1, show_metric = True) 

def epochs(n_epochs = 1,batch_size = 512):

    '''This function trains the neural network in a cycle of training and refeeding alternatively over n_epochs epochs for a given batch_size

        Parameters
        -------
        n_epochs : int32
            The number of epochs for which the neural network should be trained over the training data

        batch_size : int32
            The batch size to be used for training the model

        Returns
        -------
        None'''

    j = n_epochs
    while(j>0):
        train_model(batch_size)
        j = j - 1    

def RMMSE_on_train():

    '''This function calculates and returns the RMMSE of a neural network over the training data

        Parameters
        -------
        None

        Returns
        -------
        RMMSE_train : float64
        Root Mased Mean Squared Error of the neural network over the training data'''    

    num = np.count_nonzero(train_data)
    sparse_masked = np.multiply(np.clip(train_data,0,1),model.predict(train_data))
    sparsefinal = np.sum(np.square(np.subtract(train_data,sparse_masked)))
    RMMSE_train = np.sqrt(sparsefinal/num)
    return RMMSE_train

def RMMSE_on_test():

    '''This function calculates and returns the RMMSE of a neural network over the testing data

        Parameters
        -------
        None

        Returns
        -------
        RMMSE_test : float64
        Root Mased Mean Squared Error of the neural network over the testing data'''                 

    num = np.count_nonzero(test_data)
    sparse_masked = np.round(np.multiply(np.clip(test_data,0,1),model.predict(test_data)))
    sparsefinal = np.sum(np.square(np.subtract(test_data,sparse_masked)))
    RMMSE_test = np.sqrt(sparsefinal/num)
    return RMMSE_test

                                                            
def RMMSE_monkey():

    '''This function calculates and returns the RMMSE of a entirely random recommender over the testing data, used for establishing a reference of model performance and learning

        Parameters
        -------
        None

        Returns
        -------
        RMMSE_test : float64
        Root Mased Mean Squared Error of a random recommender over the testing data''' 

    num = np.count_nonzero(test_data)
    sparse_masked = np.multiply(np.clip(test_data,0,1),5*np.random.rand(np.shape(test_data)[0],np.shape(test_data)[1]))
    sparsefinal = np.sum(np.square(np.subtract(test_data,sparse_masked)))
    RMMSE_random = np.sqrt(sparsefinal/num)
    return RMMSE_random


def f1score():

    '''This function calculates and returns the F1 score for a given neural network recommender engine

        Parameters
        -------
        None

        Returns
        -------
        f1 : float64
        F1 score of the neural network over the testing data'''

    f1 = 0
    predicted_data_masked = np.absolute(np.subtract(train_data,np.multiply(np.clip(train_data,0,1),model.predict(train_data))))
    good_predictions = np.count_nonzero(predicted_data_masked<0.5,axis = 1)
    for i in range(len(train_data)):
        num = np.count_nonzero(train_data[i])
        if(num>0):
            recall = good_predictions[i]/num
            precision = good[i]/16885
            f1 = f1 + (2 *precision*recall)/(precision + recall)
    return f1

##########################################################################################    
        
        
        
        
        

        
        
        
        
        
        
        
        
        