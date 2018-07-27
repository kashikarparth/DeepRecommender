#Imports for numpy, tensorflow and skopt dependencies
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import RMSprop
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

#Global variables initialized for using with skopt functions for bayezian optimization 

#Training Data
path_training_data = "/media/parth/D:/Parth Kashikar/AI/CoutureWork/DeepRecommender/Rocket Retail Impl/Model Impl/Dataset and processed Data/train_data.npy"
train_data = np.load(path_training_data)

#Global current best loss and best model save directory
path_best_model = '19_best_model.keras'
best_loss = 5.0 #set high intentionally


#Search Space for hyperparameters, and compilation for usage by skopt functions
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=5, high=12, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=128, high=256, name='num_dense_nodes')
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes]
default_parameters= [1e-2, 5, 128]

def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes):

    '''This function takes in values for learning rate, number of layers and number of nodes in each layer, and creates,compiles and returns a Keras model
        with those hyperparameters

        Parameters
       -------
       learning_rate: float32
          The learning rate for the optimization technique for the neural network

       num_dense_layers: int32
          The number of hidden layers to be implemented in the neural network

       num_dense_nodes: int32
          The number of hidden units per layer of the neural network

        Returns
        -------
        model: Keras model'''

    model = Sequential()
    model.add(InputLayer(input_shape=(np.shape(train_data)[0],)))
    for i in range(num_dense_layers):
        model.add(Dense(num_dense_nodes,
                        activation='selu'))
    model.add(Dense(np.shape(train_data)[0], activation='linear'))
    optimizer = RMSprop(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mmse',
                  metrics=['accuracy'])
    
    return model

def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes):

    '''This function takes in hyperparameter values and returns a log directory for the TensorBoard logs to be stored systematically

        Parameters
       -------
       learning_rate: float32
          The learning rate for the optimization technique for the neural network

       num_dense_layers: int32
          The number of hidden layers to be implemented in the neural network

       num_dense_nodes: int32
          The number of hidden units per layer of the neural network

        Returns
        -------
        log_dir: String
          Generated log directory for TensorBoard logging'''

    s = "19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}/"
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes)

    return log_dir

@use_named_args(dimensions=dimensions) #Added to easily wrap the "dimensions" global variable as a usable entity by the following fitness function
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes):

    '''This function takes in hyperparameter values and fits a neural network of that architecture on the training data, for a fixed number of epochs.
       It also creates a TensorBoard log of the neural network being fitted.
       It return the final loss function to be minimized by skopt's bayesian optimization implementation "gp_minimize"

       Parameters
       -------
       learning_rate: float32
          The learning rate for the optimization technique for the neural network

       num_dense_layers: int32
          The number of hidden layers to be implemented in the neural network

       num_dense_nodes: int32
          The number of hidden units per layer of the neural network

       Returns
       -------
       loss: float32
          The final training loss function value of the neural network'''

    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes)

    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes)

    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)

    history = model.fit(x=train_data,
                        y=train_data,
                        epochs= 30,
                        batch_size=256,
                        validation_split=0.2,
                        callbacks=[callback_log])

    loss = history.history['val_loss'][-1]
    global best_loss
    if loss < best_loss:
        model.save(path_best_model)
        best_loss = loss
    del model
    K.clear_session()
    return loss

search_result = gp_minimize(func=fitness,dimensions = dimensions, acq_func="EI",n_calls = 40,x0 = default_parameters) #The skopt bayesian search for hyperparameters, with results stored in this variable
