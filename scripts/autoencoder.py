# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Input, Model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD,Adam

tf.random.set_random_seed(2020-12-10)

def autoencode(df_train, df_test, reduced_dim=10, input_features=512, batch_size=16):

    X_train=df_train.to_numpy()
    X_test=df_test.to_numpy()
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    #Autoencoder
    input_array = Input(shape=(input_features,))
    encoded1 = Dense(128, activation='relu', kernel_initializer="normal")(input_array)
    encoded2 = Dense(reduced_dim, activation='relu', kernel_initializer="normal")(encoded1)
    decoded1 = Dense(128, activation='relu', kernel_initializer="normal")(encoded2)
    decoded2 = Dense(input_features, activation=None, kernel_initializer="normal")(decoded1)
    autoencoder = Model(input_array, decoded2)
    
    sgd=SGD(lr=0.008,momentum=0.9,nesterov=False)
    
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error')
    autoencoder.fit(X_train,X_train,
                    epochs=500,
                    batch_size=batch_size,
                    shuffle=True)
    # Encoder
    encoder = Model(input_array, encoded2)
    # Decoder
    decoder = Model(input_array, decoded2)
    
    return encoder.predict(X_train), encoder.predict(X_test)

# =============================================================================
#     #%% Sample
#     encoded_array = encoder.predict(X[0].reshape(-1, 512,))
#     decoded_array = decoder.predict(X[0].reshape(-1, 512,))
#     
#     print(mean_squared_error(X[0], decoded_array[0]))
# 
# =============================================================================
# df_data=df.loc[:, col_indices]



#%%

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, pairwise_distances_argmin, mean_squared_error, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.cluster import KElbowVisualizer

from factor_analyzer.factor_analyzer import calculate_kmo    

import scipy
from scipy.stats import bartlett
from scipy.stats import variation
from scipy.spatial.distance import cdist

import pingouin


model_goal="classification"

#%% Display Max-Min Distance
def get_cluster_distance(cluster_centers, n_clusters):
    #Get distances
    cluster_distances = euclidean_distances(cluster_centers)
    
    #Compute Max Distance, Average Distance and Minimum Distance
    tri_dists = cluster_distances[np.triu_indices(n_clusters,1)]
    max_dist, median_dist, min_dist = tri_dists.max(), np.median(tri_dists), tri_dists.min()
    
    #Raise Error if Cluster Distance is 0
    max_min=max_dist-min_dist
    if round(max_min)==0:
        raise "Max Distance - Min Distance = 0"
    else:
        return max_dist, median_dist, min_dist

#%% Clustering Models
def cluster_kmeans(train_data, test_data, num_clusters, n_iter, r_state=None, include_elbow=False):
    kmeans_model=KMeans(n_clusters=num_clusters, random_state=r_state, n_init=n_iter)
    cluster_prediction=kmeans_model.fit(train_data).predict(test_data)
    
    if include_elbow==True:
        elbow_model=kelbow_visualizer(KMeans(n_clusters=num_clusters, random_state=r_state, n_init=n_iter),train_data, k=(2,10))
        try:
            elbow_k=int(elbow_model.elbow_value_)
        except:
            elbow_k=0
        
    #Alternative Method for getting Mean Cluster Distance
    cluster_distances=np.average(np.min(cdist(train_data, kmeans_model.cluster_centers_, 'euclidean'), axis=1))
    
    #Clustering Metrics
    sil_score=silhouette_score(train_data, kmeans_model.labels_, metric='euclidean')
    cal_har_score=calinski_harabasz_score(train_data, kmeans_model.labels_)
    dav_bou_score=davies_bouldin_score(train_data, kmeans_model.labels_)
    
    if include_elbow==True:
        return cluster_prediction, cluster_distances, (sil_score, cal_har_score, dav_bou_score), elbow_k
    else:
        return cluster_prediction, cluster_distances, (sil_score, cal_har_score, dav_bou_score)
#%%
def create_plots(include_elbow):
    global cluster_accuracies, cluster_distances, cluster_metrics, num_pc, cv, fold_cv, fold_mean_acc, dataset_num, paths, pca_model, model_goal, num_clusters, optimal_ks

    ##FIGURE: General plots [CV, Test Accuracy/RMSE, % Variance Explained]
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    plt.tight_layout(pad=2.5)
    new_title=paths[dataset_num].replace(absolute_dir+data_dir,"").replace(".csv","").replace("\\", "||")
    fig.suptitle(new_title,y=1.05)
    
    #PREPROCESSING: Flattening cv + cluster_acc for use in scatter plot + x_values
    cv_flattened=np.array([])
    cluster_accuracies_flattened=np.array([])
    num_pcs_cv=np.array([])
    num_pcs_2=np.array([])
    
    for arr in cv: cv_flattened=np.append(cv_flattened, arr)
    
    cv_accuracy=np.array([])   ##TENTATIVE
    for arr in cluster_accuracies: 
        cluster_accuracies_flattened=np.append(cluster_accuracies_flattened, arr)
        cv_accuracy=np.append(cv_accuracy, variation(arr))##TENTATIVE
    
    idx=1
    for num_cv in np.array([len(indiv_cv) for indiv_cv in cv]):
        num_pcs_cv=np.append(num_pcs_cv, np.array([num_pc[idx-1]]*num_cv))
        num_pcs_2=np.append(num_pcs_2, np.array([idx]*num_cv))
        idx+=1

    #SUBPLOT: CV vs. # of Principal Components
    ax1.plot(num_pc, cv_accuracy, color='#4daf4a', marker="D", markersize=7)
    
    ax1.set_ylabel('Coefficient of Variation')
    x_idx = list(range(len(num_pc)+1))
    x_num_pc=num_pc.copy()
    ax1.set_ylim((min(0, min(cv_accuracy)), round(np.nan_to_num(cv_accuracy).max(), 1)+0.1))
    ax1.tick_params(axis='x', labelsize=7)
    ax1.set_xlabel('Number of Features')
    
    #SUBPLOT: Boxplot of Cluster Testing Accuracies vs. # of Principal Components
    ax2.tick_params(axis='x', labelsize=7)
    ax2.boxplot(cluster_accuracies,labels=num_pc)
    ax2.scatter(num_pcs_2, cluster_accuracies_flattened,
                s=20,
                c="black",
                alpha=0.4,
                edgecolors="none",
                )
    ax2.axhline(fold_mean_acc,
                c="red",
                alpha=0.8)
    
    if model_goal=="classification":
        ax2.set_ylabel('Testing Accuracy')
        ax2.set_ylim((min(0, min(cluster_accuracies_flattened)), 1))
    else:
        ax2.set_ylabel('Testing RMSE')
        ax2.set_ylim((min(0, min(cluster_accuracies_flattened)), 100))
    ax2.tick_params(axis='x', labelsize=7)
    ax2.set_xlabel('Number of Features')
    
    fig.align_ylabels()
    fig.savefig(absolute_dir+"results/graphs/general/"+boneage_or_psp+"_dataset_"+str(dataset_num)+".png", bbox_inches='tight')
     
    ##FIGURE: Automatic Elbow Method for KMeans Clustering
    if include_elbow==True:
        plt.figure()
        plt.xlabel('Number of Features')
        plt.ylabel('Optimal Number of KMeans Clusters')
        plt.xticks(fontsize=7)
        plt.ylim((0, 10))
        plt.plot(num_pc, optimal_ks, "^--", markersize=7)
        plt.title(new_title, y=1.05)
        plt.savefig(absolute_dir+"results/graphs/optimal_k/"+new_title.replace("||", "-")+".png", bbox_inches='tight')
          
    ##FIGURE: Cluster Metrics
    (sil_score, cal_har_score, dav_bou_score)=cluster_metrics
    fig2 = plt.figure()
    bx1 = fig2.add_subplot(221)
    bx2 = fig2.add_subplot(222)
    bx3 = fig2.add_subplot(223)
    bx4 = fig2.add_subplot(224)
    plt.tight_layout(pad=2.5)
    fig2.suptitle(new_title,y=1.05)
    
    bx1.set_ylabel('Silhouette Coefficient')
    bx1.plot(num_pc, sil_score, marker="o", markersize=7)
    bx1.tick_params(axis='x', labelsize=7)
    
    bx2.set_ylabel('Calinski-Harabasz Index')
    bx2.plot(num_pc, cal_har_score, marker="o", markersize=7)
    bx2.tick_params(axis='x', labelsize=7)
    
    bx3.set_xlabel('Number of Features')
    bx3.set_ylabel('Davies-Bouldin Index')
    bx3.plot(num_pc, dav_bou_score, marker="o", markersize=7)
    bx3.tick_params(axis='x', labelsize=7)
        
    
    #FIGURE: Euclidean Distance
    bx4.set_xlabel('Number of Features')
    bx4.set_ylabel('''Euclidean Distance
between Centroids''')
    bx4.set_ylim((0, 75))
    bx4.tick_params(axis='x', labelsize=7)
    bx4.plot(num_pc, cluster_distances, "or")

    fig2.savefig(absolute_dir+"results/graphs/cluster_metrics/"+boneage_or_psp+"_dataset_"+str(dataset_num)+".png", bbox_inches='tight')
#%% Main Code
def main(chosen_pcs, num_cluster=4, n_iter=1, random_state=None, include_elbow=False):
    global df_test, df_test_data, df_train_data, model_goal
    
    df_data=df_test.copy()
    cv=[]
    num_pc=[]
    cluster_accuracies=[]
    cluster_distance_metrics=[]
    sil_accumulator=[]
    cal_har_accumulator=[]
    dav_bou_score_accumulator=[]
    optimal_ks=[]
    for g in chosen_pcs:
        #Get train and test data
        cluster_train, cluster_val=autoencode(df_train_data, df_test_data, reduced_dim=g)
        
        #Fit Training, Predict Cluster
        if include_elbow==True:
            cluster_prediction,cluster_distances,metrics, elbow_k=cluster_kmeans(cluster_train,
                                                                             cluster_val, 
                                                                             num_clusters=num_cluster, 
                                                                             n_iter=n_iter, 
                                                                             r_state=random_state,
                                                                             include_elbow=include_elbow)
            #Append Elbow K
            optimal_ks.append(elbow_k)
        else:
            cluster_prediction,cluster_distances,metrics=cluster_kmeans(cluster_train,
                                                                     cluster_val, 
                                                                     num_clusters=num_cluster, 
                                                                     n_iter=n_iter,
                                                                     r_state=random_state,
                                                                     include_elbow=include_elbow)
        #Getting cluster sizes
        cluster_indiv_num=pd.Series(cluster_prediction).value_counts().index.to_list()
        cluster_sizes=pd.Series(cluster_prediction).value_counts().values
        
        sorted_cluster_sizes=pd.DataFrame(cluster_sizes, index=cluster_indiv_num).sort_index().values.flatten()

        #Get cluster test accuracies
        df_data["cluster"]=cluster_prediction
        if model_goal=="regression":
            df_data["prediction_accuracy"]=df_data.apply(lambda x: np.sqrt(((x.predictions - x.labels) ** 2)), axis=1)
        else:
            df_data["prediction_accuracy"]=(df_data.predictions==df_data.labels)
        df_cluster_accuracies=df_data.groupby(by=["cluster"]).mean()["prediction_accuracy"]
        df_cluster_accuracies_std=df_data.groupby(by=["cluster"]).std()["prediction_accuracy"]
        
        if (df_cluster_accuracies==0).sum()>0:
            zero_idx=np.where(df_cluster_accuracies==0)
            print("At "+str(g)+ " PCs, the cluster/s "+str(zero_idx[0].tolist())+" with "+str(sorted_cluster_sizes[zero_idx])+" values have 0 accuracy.")
            
            for cluster_idx in np.where(df_cluster_accuracies==0)[0]:
                print("Cluster "+str(cluster_idx)+"has images with 0 accuracy at indices: " + str(np.where(cluster_prediction==cluster_idx)[0]))
        
        #Trying to prevent NA in CV and Test Accuracy    #REMOVE
        accuracy_values=[]
        accuracy_std=[]
        test_accuracy=[]
        for k in range(num_cluster):
            try:
                accuracy_values.append(df_cluster_accuracies[k])
                accuracy_std.append(df_cluster_accuracies_std[k])
                test_accuracy.append(df_cluster_accuracies[k])
            except:
                accuracy_values.append(1)
                accuracy_std.append(-1)
                test_accuracy.append(-1)
        #Get Coefficient of Variation of Testing Acc + Sd (SAFE)
        cv_individuals=np.array(accuracy_std)/np.array(accuracy_values)
        cv.append(cv_individuals)
        #Append PC number
        num_pc.append(g)
        #Append cluster testing accuracies
        cluster_accuracies.append(test_accuracy)
        #Append euclidean distance (max-min, median)
        cluster_distance_metrics.append(cluster_distances)
        #Get and append metrics
        sil_score, cal_har_score, dav_bou_score=metrics
        sil_accumulator.append(sil_score)
        cal_har_accumulator.append(cal_har_score)
        dav_bou_score_accumulator.append(dav_bou_score)
        
    return cv, num_pc, cluster_accuracies, cluster_distance_metrics, (sil_accumulator, cal_har_accumulator, dav_bou_score_accumulator), optimal_ks

#%% Test Code

#File Paths
absolute_dir="/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/"
home_dir="/home/stanley_hua/scripts/pca_clustering/"
data_dir="data/"

#INPUT: Dataset and Type of Model (used for feature extraction)
if int(input("DATASET: boneage or psp_plates (1/2) | "))==1:
    boneage_or_psp="boneage"
    model_goal="regression"
else:
    boneage_or_psp="psp_plates"
    model_goal="classification"

#Get csv file paths
paths=[]
for root, dirs, files in os.walk(absolute_dir+data_dir+boneage_or_psp, topdown=False):
   for name in files:
      paths.append(os.path.join(root, name))

#%%
col_indices=[str(i) for i in range(512)]

num_clusters=int(input("Number of Clusters: "))
which_datasets=int(input("One or All datasets (1/0) | "))
elbow_bool=int(input("Include Elbow Plot (1/0) | "))

if which_datasets==0:
    which_datasets=range(len(paths))
else:
    which_datasets=random.sample(range(len(paths)), 1)

for dataset_num in which_datasets:
    df=pd.read_csv(paths[dataset_num], index_col=False)
    try:
        df=df.drop("Unnamed: 0", axis=1)
    except:
        pass
    
    #GET TRAINING DATA & VAL & TEST DATA
    df_train=df[df.phase=="train"]
    df_train_data=df_train.loc[:,col_indices]
    df_test=df[df.phase=="val"]
    df_test_data=df_test.loc[:,col_indices]
        
    included_features=[1,2,3,5,10,15,20,30,50,70]                                     #CHANGE THIS
                                                                               #CHANGE n_cluster, n_iter
    cv, num_pc, cluster_accuracies, cluster_distances, cluster_metrics, optimal_ks=main(included_features, 
                                                        num_cluster=num_clusters,
                                                        n_iter=100,
                                                        include_elbow=bool(elbow_bool)
                                                        )
    
    #Get Fold CV
    if model_goal=="regression":
        df_test["prediction_accuracy"]=df_test.apply(lambda x: np.sqrt(((x.predictions - x.labels) ** 2)), axis=1)
    else:
        df_test["prediction_accuracy"]=(df_test.predictions==df_test.labels)
    
    fold_cv=df_test["prediction_accuracy"].std()/df_test["prediction_accuracy"].mean()
    fold_mean_acc=df_test["prediction_accuracy"].mean()
    
    #Plots
    create_plots(include_elbow=bool(elbow_bool))
