# Import libraries
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

#%%
#INPUT: Dataset and Type of Model (used for feature extraction)
if int(input("DATASET: boneage or psp_plates (1/2) | "))==1:
    boneage_or_psp="boneage"
    model_goal="regression"
else:
    boneage_or_psp="psp_plates"
    model_goal="classification"

project_dir="z:/home/stanley_hua/pca_clustering/"
#%%#Get model directories
model_dir="%smodels/%s/" % (project_dir, boneage_or_psp)

paths=[]
for root, dirs, files in os.walk(model_dir, topdown=False):
   for name in files:
      paths.append(root)
paths=list(set(paths))
paths.sort()

#%%
def load_autoencoder(dir, initial_dim=512, red_dim=10):
    #Autoencoder Architecture
    input_array=Input(shape=(initial_dim,))
    encoded1=Dense(128, activation='relu', kernel_initializer="normal")(input_array)
    encoded2=Dense(red_dim, activation='relu', kernel_initializer="normal")(encoded1)
    decoded1=Dense(128, activation='relu', kernel_initializer="normal")(encoded2)
    decoded2=Dense(initial_dim, activation=None, kernel_initializer="normal")(decoded1)
    autoencoder=Model(input_array, decoded2)
    
    #Encoder
    encoder = Model(input_array, encoded2)
    encoder.load_weights("%s/encoder_%d.h5" % (dir, red_dim))
    #Decoder
    decoder_layers=autoencoder.layers[-2:]
    encoded_input=Input(shape=(red_dim,))
    decoded_1=decoder_layers[0](encoded_input)
    decoded_2=decoder_layers[1](decoded_1)
    decoder=Model(encoded_input, decoded_2)
    decoder.load_weights("%s/decoder_%d.h5" % (dir, red_dim))
    return encoder, decoder

#FUNCTION: Return Mean Absolute Error between Original Features and PCA-Transformed Features for Each Observation
def compare_autoencoder_inverse(original_df, transformed_df):
    df_diff=pd.DataFrame(original_df.to_numpy()-transformed_df.to_numpy())
    return df_diff.apply(abs).mean(axis=1)


#%%
col_indices=[str(i) for i in range(512)]
which_datasets=int(input("One or All datasets (1/0) | "))
num_clusters=int(input("Number of Clusters: "))
elbow_bool=int(input("Include Elbow Plot (1/0) | "))
if elbow_bool not in [1,0]:
    raise "Invalid Input!"

if which_datasets==0:
    which_datasets=range(len(paths))
else:
    which_datasets=random.sample(range(len(paths)), 1)

#FOR TESTING PURPOSES
inverse_train_diff=np.array([])
inverse_test_diff=np.array([])
loss_acc=[]

for dataset_num in which_datasets:
    df=pd.read_csv(paths[dataset_num].replace("models", "data")+".csv", index_col=False)        #careful
    try:
        df=df.drop("Unnamed: 0", axis=1)
    except:
        pass
    
    #Split Training and Test Data
    df_train=df[df.phase=="train"]
    df_test=df[df.phase=="val"]
    
    df_train_data=df_train.loc[:,col_indices]
    df_test_data=df_test.loc[:,col_indices]
    df_whole_data=pd.concat([df_train_data, df_test_data])
    
    #Dimensionality Reduction
    # chosen_features=[1,2,3,5,10,15,20,30,50,70]
    chosen_features=[5]
    for num_features in chosen_features:
        encoder,decoder=load_autoencoder(paths[dataset_num], red_dim=num_features)
        
        loss=pd.read_csv("%s/autoencoder_loss_%d.csv"% (paths[dataset_num], num_features), index_col=False).iloc[-1,:].loss
        loss_acc.append(loss)
        
        df_trans=encoder.predict(df_train_data)
        df_inverse_trans=pd.DataFrame(decoder.predict(df_trans))
        inverse_train_diff=np.append(inverse_train_diff, compare_autoencoder_inverse(df_train_data, df_inverse_trans).mean())

        df_trans=encoder.predict(df_test_data)
        df_inverse_trans=pd.DataFrame(decoder.predict(df_trans))
        inverse_test_diff=np.append(inverse_test_diff, compare_autoencoder_inverse(df_test_data, df_inverse_trans).mean())

print(inverse_train_diff)
print(inverse_test_diff)
        