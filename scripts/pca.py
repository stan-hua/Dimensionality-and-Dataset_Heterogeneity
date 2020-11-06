import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances

from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from sklearn.metrics import pairwise_distances_argmin

from factor_analyzer.factor_analyzer import calculate_kmo    
from scipy.stats import bartlett

import scipy
import pingouin
import os
import math
import random

#File Paths
absolute_dir="/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/"
home_dir="/home/stanley_hua/scripts/pca_clustering/"
data_dir="data/"

#Dataset ("boneage" or "psp_plates")
boneage_or_psp="psp_plates"

#Get csv file paths
paths=[]
for root, dirs, files in os.walk(absolute_dir+data_dir+boneage_or_psp, topdown=False):
   for name in files:
      paths.append(os.path.join(root, name))

#%% Defining PCA class
class pca:
    'Principal Component Analysis Object'
    
    global chosen_PCs
    
    def __init__(self):
        pass
        
    def compute(self, df_train_data, df_test_data):
        """
        1) Standardizes data. 
        2) Runs Principal Component Analysis. 
        3) Saves model and principal components (as Pandas DataFrame) to object
        
        Returns
        -------
        None.

        """
        #Standardize Data
        df_train_normalized=StandardScaler().fit_transform(df_train_data)
        df_test_normalized=StandardScaler().fit_transform(df_test_data)
        #Create PCA object. Fit and transform
        pca_model=PCA()
        train_pcs=pca_model.fit_transform(df_train_normalized)
        test_pcs=pca_model.transform(df_test_normalized)
        #Save instance and principal components
        self.pcs_train=pd.DataFrame(train_pcs)
        self.pcs_test=pd.DataFrame(test_pcs)
        self.model=pca_model
        self.max_pcs=len(self.pcs_train.columns)-1
    
    def get_graph_items(self, chosen_pcs=None):
        """
        Plots explained variance ratio versus each PCA component.

        Raises
        ------
        Error
            When compute() method has not been called.

        Returns
        -------
        None.

        """
        
        if self.model!=None:
            #If chosen_pcs not defined
            if chosen_pcs==None or len(chosen_pcs)>20:
                chosen_pcs=[int(self.max_pcs/i) for i in range(15,0,-2)]
            chosen_pcs_unique=np.unique(chosen_pcs)
            
            #Cumulative
            variance_explained=pd.Series(self.model.explained_variance_ratio_[:]).cumsum()
            
            y_var=variance_explained.iloc[chosen_pcs_unique].values
            return (chosen_pcs_unique, y_var)
        else:
            raise "Method compute() has not been passed!"
        
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
        return max_min, median_dist
    
#%% Validate Assumptions
def validate_assumptions(df):
    """
    Return result of Bartlett's Test and KMO Test on df.
    """
    #Bartlett's Test of Sphericity      #Are the variables correlated with one another?
        #chi_square_value,p_value=calculate_bartlett_sphericity(self.df)
    sphericity_test=pingouin.homoscedasticity(df,method="bartlett")
    p_value=sphericity_test["pval"].iloc[0]
    
    #KMO Test for sampling adequacy     #Is most variance common variance?
    kmo_all,kmo_model=calculate_kmo(df)
    
    if math.isnan(kmo_model): kmo_model=0.5
    #Printout
    if p_value > 0.05 and kmo_model < 0.6:
        print("Fails both tests!\n p-value: %d \n KMO value: %f" % (p_value,kmo_model))
        result="Invalid"
    elif p_value > 0.05:
        print("Fails Bartlett's Test of Sphericity \n p-value: %d" % p_value)
        result="Invalid"
    elif kmo_model < 0.6:
        print("Fails the Kaiser-Meyer-Olkin Test \n KMO Value: %f" % kmo_model)
        result="Invalid"
    elif p_value < 0.05 and kmo_model > 0.6:
        print("Assumptions are valid!\n p-value: %d \n KMO value: %f" % (p_value,kmo_model))
        result="Valid"
    else:
        print("NaN Calculated for KMO!")
        result="Invalid"
    
    return result

#%% KMO https://github.com/Sarmentor/KMO-Bartlett-Tests-Python/blob/master/correlation.py

#%% Clustering Models
def cluster_kmeans(train_data, test_data, num_clusters=4):
    kmeans_model=KMeans(n_clusters=num_clusters, random_state=2004).fit(train_data)         #REMOVE RANDOM STATE
    cluster_prediction=kmeans_model.predict(test_data)    
    cluster_distances=get_cluster_distance(kmeans_model.cluster_centers_, num_clusters)

    # Fuzzy K Means
# =============================================================================
#     fuzzy_model=FuzzyKMeans(k=num_clusters,m=2).fit(train_data)
#     cluster_prediction=pairwise_distances_argmin(test_data,fuzzy_model.cluster_centers_)
# =============================================================================
    
    return cluster_prediction, cluster_distances

#%%
def create_plots():
    global cluster_accuracies, cluster_distances, num_pc, cv, i, paths, pca_model
    
    max_min_dist, median_dist=cluster_distances
    #CREATING FIGURE:
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    plt.tight_layout()
    fig.suptitle(paths[i].replace(absolute_dir+data_dir,""),y=1.05)
    
    #FIGURE: CV vs. # of Principal Components
    ax1.plot(num_pc, cv, color='#4daf4a', marker="D", markersize=4)
    ax1.set_ylabel('Coefficient of Variation')
    ax1.set_ylim((0, 1))
    ax1.tick_params(axis='x', labelsize=7)
    
    #Preprocessing for use in scatter plot
    num_pcs_2=np.ndarray.flatten(np.array([[g]*4 for g in range(1,len(num_pc)+1)]))     #depends on number of clusters
    cluster_accuracies_2=np.ndarray.flatten(np.array(cluster_accuracies))
    
    #FIGURE: Boxplot of Cluster Testing Accuracies vs. # of Principal Components
    ax2.boxplot(cluster_accuracies,labels=num_pc)
    ax2.scatter(num_pcs_2, cluster_accuracies_2,
                s=9,
                c="black",
                alpha="0.25",
                edgecolors="none",
                )
    ax2.set_ylabel('Testing Accuracy')
    ax2.set_ylim((0.7, 1))
    ax2.tick_params(axis='x', labelsize=7)
    
    #FIGURE: Percent Explained Variance vs. Number of Principal Components
    ax3.set_xlabel('Number of Principal Components')
    ax3.set_ylabel('Percent Explained Variance')
    ax3.tick_params(axis='x', labelsize=7)
    ax3.set_ylim((0, 1))
    
    x,y=pca_model.get_graph_items(chosen_pcs)
    ax3.plot(x, y, marker="o", markersize=4)

    #FIGURE: Euclidean Distance
    ax4.set_xlabel('Number of Principal Components')
    ax4.set_ylabel('Euclidean Cluster Distance')
    ax4.set_ylim((0, 50))
    ax4.tick_params(axis='x', labelsize=7)
    ax4.plot(num_pc, max_min_dist, label="max-min")
    ax4.plot(num_pc, median_dist, label="median")
    ax4.legend()
    
# =============================================================================
#     fig.savefig(absolute_dir+"results/graphs/"+boneage_or_psp+"_dataset_"+str(i)+".png")
# =============================================================================
    
#%% Main Code for Getting cv, num_pc, and cluster_accuracies
def main(chosen_pcs, n_iter=1):
    global df_test, pca_model
    iteration=0
    while iteration < n_iter:
        df_data=df_test.copy()
        
        #chosen_pcs=[h for h in range(1, pca_train.max_pcs)]
        cv=[]
        num_pc=[]
        cluster_accuracies=[]
        cluster_distance_metrics=[]
        for g in chosen_pcs:
            #Get train and test data
            cluster_train=pca_model.pcs_train.loc[:,:g]
            cluster_val=pca_model.pcs_test.loc[:,:g]
            
            #Fit Training, Predict Test
            cluster_prediction,cluster_distances=cluster_kmeans(cluster_train,cluster_val)
            
            #Getting cluster sizes
            cluster_num=pd.Series(cluster_prediction).value_counts().index.to_list()
            values=pd.Series(cluster_prediction).value_counts().values

            #Get cluster test accuracies
            df_data["cluster"]=cluster_prediction
            df_data["prediction_bool"]=df_data.predictions==df_data.labels
            df_cluster_accuracies=df_data.groupby(by=["cluster"]).mean()["prediction_bool"]
            
            accuracy_values=[]
            for k in range(4):
                try:
                    accuracy_values.append(df_cluster_accuracies[k])
                except:
                    accuracy_values.append(0)
            
            #Get and append Coefficient of Variation
            cv.append(scipy.stats.variation(values))
            #Append PC number
            num_pc.append(g)
            #Append cluster testing accuracies
            cluster_accuracies.append(accuracy_values)   
            #Append euclidean distance (max-min, median)
            cluster_distance_metrics.append(cluster_distances)
        #Append to accumulator
        if iteration==0:
            cv_array=[cv]
            num_pc_array=[num_pc]
            cluster_acc_array=[cluster_accuracies]
            cluster_distance_array=[cluster_distance_metrics]
        else:
            cv_array.append(cv)
            num_pc_array.append(num_pc)
            cluster_acc_array.append(cluster_accuracies)
            cluster_distance_array.append(cluster_distance_metrics)
        
        iteration+=1
    
    #Average of n_iter trials
    final_cvs=np.average(cv_array, axis=0)
    final_num_pcs=np.average(num_pc_array, axis=0).astype(int)
    final_cluster_acc=np.average(cluster_acc_array, axis=0).transpose()
    final_cluster_distances=np.average(cluster_distance_array, axis=0).transpose()
    return final_cvs, final_num_pcs, final_cluster_acc, final_cluster_distances

#%% Test Code
            
col_indices=[str(i) for i in range(512)]

# =============================================================================
# for i in range(len(paths)):
# =============================================================================
i=9
df=pd.read_csv(paths[i], index_col=False)
try:
    df=df.drop("Unnamed: 0", axis=1)
except:
    pass

random.seed(2004)

#GET TRAINING DATA & VAL & TEST DATA
df_train=df[df.phase=="train"]
df_test=df[df.phase=="val"]

df_train_data=df_train.loc[:,col_indices]
df_test_data=df_test.loc[:,col_indices]

#Check if Assumptions are valid
if validate_assumptions(df_train_data)=="Valid":
    pca_model=pca()
    pca_model.compute(df_train_data,df_test_data)
    
    chosen_pcs=[1,2,3,5,10,15,20,30,50,80]
    
    cv, num_pc, cluster_accuracies, cluster_distances=main(chosen_pcs, n_iter=5)    #CHANGE n_iter
    
    #Plots
    create_plots()
    
    #Save Results
    df_results=pd.DataFrame([cv, num_pc, cluster_accuracies.transpose(), cluster_distances.transpose()]).transpose()
    df_results=df_results.rename({0:"cv", 1:"num_pc", 2:"cluster_accuracy", 3:"cluster_distance"}, axis=1)
# =============================================================================
#         df_results.to_csv(absolute_dir+"results/dataset/"+boneage_or_psp+"_dataset_"+str(i)+".csv", index="False")
# =============================================================================
    
else:
    print("Dataset: " + paths[i].replace(absolute_dir+data_dir,""))
    print("       with "+str(len(df))+" observations is invalid!")






#FOR CONSIDERATION
    #compare different num of clusters
    #another type of dimensionality reduction / feature selection
    #what is the effect of clustering on features from a highly overfit model versus an underfit model

#Data Analysis
    #One-way ANOVA


#DONE
    #compare standardscaler (with_std=False)
        #decreases euclidean distance metrics
        #increases spread of testing accuracy, percent explained variance

