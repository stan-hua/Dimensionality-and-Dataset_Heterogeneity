# Import libraries
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping

# tf.random.set_random_seed(2020-12-10)
#%% AUTOENCODER
def autoencode(X_train, dim=10, epochs=3000, batch_size=None):
    """
    Parameters
    ----------
    X_train : np.ndarray
        Contains arrays with 'input_features' dimensions.
    dim : int, optional
        The desired reduced dimensionality. Default is 10.
    epochs : int, optional
        Number of epochs used in training autoencoder.
    batch_size: int, optional
        Batch size for training autoencoder, defaults to 20% of training data.

    Returns
    -------
    encoder : tf.keras.Functional
        Model that takes in input array and returns array with dim dimensions.
    decoder : tf.keras.Functional
        Model that takes in encoded array and returns array with original dimensions.
    df_history : pd.DataFrame
        Contains history of model training.
    """
    #Default batch_size to 1/5 of training data
    if batch_size==None:
        batch_size=1
    
    #AUTOENCODER
    input_array=Input(shape=(X_train.shape[1],))
    encoded1=Dense(1024, activation="relu", kernel_initializer="normal")(input_array)
    encoded2=Dense(512, activation="relu", kernel_initializer="normal")(encoded1)
    encoded3=Dense(128, activation="relu", kernel_initializer="normal")(encoded2)
    last_encoded=Dense(dim, activation="relu", kernel_initializer="normal")(encoded3)
    decoded1=Dense(128, activation="relu", kernel_initializer="normal")(last_encoded)
    decoded2=Dense(512, activation="relu", kernel_initializer="normal")(decoded1)
    decoded3=Dense(1024, activation="relu", kernel_initializer="normal")(decoded2)
    last_decoded=Dense(X_train.shape[1], activation=None, kernel_initializer="normal")(decoded3)
    
    autoencoder=Model(input_array, last_decoded)
    
    #Optimizer
    sgd=SGD(lr=0.006,momentum=0.9,nesterov=False)
    
    ada=Adagrad(learning_rate=0.1)
    
    
    #Compile and Fit
    autoencoder.compile(optimizer=ada, loss='mean_absolute_error')
    history=autoencoder.fit(X_train,X_train,
                    epochs=epochs,
                    validation_data=[X_train, X_train],
                    batch_size=batch_size,
                    shuffle=True)
    
    #Encoder
    encoder = Model(input_array, last_encoded)
    
    #Decoder
    decoder_layers=autoencoder.layers[-4:]
    decoder=Sequential()
    decoder.add(Input(shape=(dim,)))
    for layer in decoder_layers:
        decoder.add(layer)
        
    return encoder, decoder, pd.DataFrame(history.history)

# =============================================================================
#     #%% Sample
#     encoded_array = encoder.predict(X[0].reshape(-1, 512,))
#     decoded_array = decoder.predict(X[0].reshape(-1, 512,))
#     
#     print(mean_squared_error(X[0], decoded_array[0]))
# 
# =============================================================================
# df_data=df.loc[:, col_indices]


#%% Test Code

#File Paths
absolute_dir="/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/"
model_save_dir="z:/home/stanley_hua/pca_clustering/models/"
data_dir="data/"

#INPUT: Dataset and Type of Model (used for feature extraction)
if int(input("DATASET: boneage or psp_plates (1/2) | "))==1:
    boneage_or_psp="boneage"
else:
    boneage_or_psp="psp_plates"

#Get csv file paths
paths=[]
for root, dirs, files in os.walk(absolute_dir+data_dir+boneage_or_psp, topdown=False):
   for name in files:
      paths.append(os.path.join(root, name))

#%%
col_indices=[str(i) for i in range(512)]

which_datasets=int(input("One or All datasets (1/0) | "))

if which_datasets==0:
    which_datasets=range(len(paths))
    
else:
    # which_datasets=random.sample(range(len(paths)), 1)
    which_datasets=[0]

for dataset_num in which_datasets:
    df=pd.read_csv(paths[dataset_num], index_col=False)
    try:
        df=df.drop("Unnamed: 0", axis=1)
    except:
        pass
    
    #GET TRAINING DATA & VAL & TEST DATA
    df_train=df[df.phase=="train"].loc[:,col_indices]
    df_test=df[df.phase=="val"].loc[:,col_indices]
    
    included_features=[1,2,3,5,10,15,20,30,50,70]
    
    for num_features in included_features:
        encoder,decoder,df_history=autoencode(df_train.to_numpy(), dim=int(num_features))
        
        save_filename=paths[dataset_num].replace(absolute_dir+data_dir+boneage_or_psp,"").replace("\\","").replace(".csv","")
        save_path=model_save_dir+boneage_or_psp+"/"+save_filename
        try:
            os.mkdir(save_path)
        except:
            pass
        #%%
        encoder.save_weights(save_path+"/encoder_"+str(num_features)+".h5")
        decoder.save_weights(save_path+"/decoder_"+str(num_features)+".h5")
        df_history.loss.to_csv(save_path+"/"+"autoencoder_loss_"+str(num_features)+".csv", index=False)
        plt.plot(df_history.loss)
        df_history.loss.tolist()[-1]
