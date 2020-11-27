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
from scipy.stats import bartlett
import scipy
import pingouin
from scipy.spatial.distance import cdist

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
    
# =============================================================================
# if int(input("DATASET: Model goal is regression or classification (1/0) "))==1:
#     model_goal="regression"
# else:
#     model_goal="classification"
# =============================================================================


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
    
    def get_cum_variance(self, chosen_pcs=None):
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
        
#%% Validate Assumptions
def validate_efa_assumptions(df):
    """
    Return result of Bartlett's Test and KMO Test on df. Used to determine applicability of EFA on dataset.
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
        
# =============================================================================
#     cluster_distances=get_cluster_distance(kmeans_model.cluster_centers_, num_clusters)
# =============================================================================
    #Alternative Method for getting Mean Cluster Distance
    cluster_distances=np.average(np.min(cdist(train_data, kmeans_model.cluster_centers_, 'euclidean'), axis=1))
    
    #Clustering Metrics
    sil_score=silhouette_score(train_data, kmeans_model.labels_, metric='euclidean')
    cal_har_score=calinski_harabasz_score(train_data, kmeans_model.labels_)
    dav_bou_score=davies_bouldin_score(train_data, kmeans_model.labels_)
    
# =============================================================================
#     # Fuzzy K Means
#     fuzzy_model=FuzzyKMeans(k=num_clusters,m=2).fit(train_data)
#     cluster_prediction=pairwise_distances_argmin(test_data,fuzzy_model.cluster_centers_)
# =============================================================================
    if include_elbow==True:
        return cluster_prediction, cluster_distances, (sil_score, cal_har_score, dav_bou_score), elbow_k
    else:
        return cluster_prediction, cluster_distances, (sil_score, cal_har_score, dav_bou_score)
#%%
def create_plots(include_elbow):
    global cluster_accuracies, cluster_distances, cluster_metrics, num_pc, cv, fold_cv, fold_mean_acc, dataset_num, paths, pca_model, model_goal, num_clusters, optimal_ks

    #Getting max, median, min Euclidean distances between all clustroid centres
# =============================================================================
#     max_dist=[k[0] for k in cluster_distances]
#     median_dist=[k[1] for k in cluster_distances]
#     min_dist=[k[2] for k in cluster_distances]
# =============================================================================

    ##FIGURE: General plots [CV, Test Accuracy/RMSE, % Variance Explained]
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    plt.tight_layout(pad=2.5)
    new_title=paths[dataset_num].replace(absolute_dir+data_dir,"").replace(".csv","").replace("\\", "||")
    fig.suptitle(new_title,y=1.05)
    
    
# =============================================================================
#     #PREPROCESSING: Get (number of pcs) for use as x values in scatter plot (Fig 1 and 2)
#     num_pcs_2=np.array([])
#     num_pcs_cv=np.array([])
#     cv_index=1
#     for num_cv in np.array([len(indiv_cv) for indiv_cv in cv]):
#         num_pcs_2=np.append(num_pcs_2, np.array([cv_index]*num_cv))    
#         num_pcs_cv=np.append(num_pcs_cv, np.array([num_pc[cv_index-1]]*num_cv))
#         cv_index+=1
# =============================================================================
    
    #PREPROCESSING: Flattening cv + cluster_acc for use in scatter plot + x_values
    cv_flattened=np.array([])
    cluster_accuracies_flattened=np.array([])
    num_pcs_cv=np.array([])
    num_pcs_2=np.array([])
    
    for arr in cv: cv_flattened=np.append(cv_flattened, arr)
    for arr in cluster_accuracies: cluster_accuracies_flattened=np.append(cluster_accuracies_flattened, arr)
    
    idx=1
    for num_cv in np.array([len(indiv_cv) for indiv_cv in cv]):
        num_pcs_cv=np.append(num_pcs_cv, np.array([num_pc[idx-1]]*num_cv))
        num_pcs_2=np.append(num_pcs_2, np.array([idx]*num_cv))
        idx+=1

    #SUBPLOT: CV vs. # of Principal Components
# =============================================================================
#     ax1.plot(num_pc, cv, color='#4daf4a', marker="D", markersize=7)
# =============================================================================
    # ax1.boxplot(cv, labels=num_pc)
    ax1.scatter(num_pcs_2, cv_flattened,
                s=30,
                c=num_pcs_2,
                cmap="Dark2",
                alpha=0.8,
                edgecolors="none",
                )
    ax1.axhline(fold_cv,
                c="red",
                alpha=0.8)
    
    ax1.set_ylabel('Coefficient of Variation')
    x_idx = list(range(len(num_pc)+1))
    x_num_pc=num_pc.copy()
    x_num_pc.insert(0,0)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_num_pc)    
    
# =============================================================================
#     if cv_flattened.max() <= 1:
#         ax1.set_ylim((0, 1))
#     else:
#         ax1.set_ylim((0, round(cv_flattened.max())+0.1))
# =============================================================================
    if np.nan_to_num(cv).max() <= 1:
        ax1.set_ylim((min(0, min(cv_flattened)), 1))
    else:
        ax1.set_ylim((min(0, min(cv_flattened)), round(np.nan_to_num(cv).max())+0.1))
    ax1.tick_params(axis='x', labelsize=7)
    
    
        
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
    
    #SUBPLOT: Percent Explained Variance vs. Number of Principal Components
    ax3.set_xlabel('Number of Principal Components')
    ax3.set_ylabel('Percent Explained Variance')
    ax3.tick_params(axis='x', labelsize=7)
    ax3.set_ylim((0, 1))
    
    x,y=pca_model.get_cum_variance(chosen_pcs)
    ax3.plot(x, y, marker="o", markersize=7)
    
    #SUBPLOT: Absolute sum difference from mean CV
    absolute_cv_difference=abs((np.array(cv)-fold_cv)).sum(axis=1)

    ax4.set_xlabel('Number of Principal Components')
    ax4.set_ylabel('''Sum Absolute Difference between
Cluster and Overall CV''', fontsize=10)
    ax4.tick_params(axis='x', labelsize=7)
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(x_num_pc)   
    ax4.set_ylim((0, max(1, max(absolute_cv_difference))))
    
    ax4.plot(range(1,len(num_pc)+1), absolute_cv_difference, "o--")
    
    fig.align_ylabels()
    fig.savefig(absolute_dir+"results/graphs/general/"+boneage_or_psp+"_dataset_"+str(dataset_num)+".png", bbox_inches='tight')
    
    
    ##FIGURE: Automatic Elbow Method for KMeans Clustering
    if include_elbow==True:
        plt.figure()
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Optimal Number of KMeans Clusters')
        plt.xticks(fontsize=7)
        plt.ylim((0, 10))
        plt.plot(num_pc, optimal_ks, "^--", markersize=7)
        plt.title(new_title, y=1.05)
# =============================================================================
#         plt.savefig(absolute_dir+"results/graphs/optimal_k/"+new_title.replace("||", "-")+".png", bbox_inches='tight')
# =============================================================================
        
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
    
    bx3.set_xlabel('Number of Principal Components')
    bx3.set_ylabel('Davies-Bouldin Index')
    bx3.plot(num_pc, dav_bou_score, marker="o", markersize=7)
    bx3.tick_params(axis='x', labelsize=7)
        
    
    #FIGURE: Euclidean Distance
    bx4.set_xlabel('Number of Principal Components')
    bx4.set_ylabel('Euclidean Cluster Distance')
    bx4.set_ylim((0, 75))
    bx4.tick_params(axis='x', labelsize=7)
# =============================================================================
#     bx4.plot(num_pc, max_dist, label="max")
#     bx4.plot(num_pc, median_dist, label="median")
#     bx4.plot(num_pc, min_dist, label="min")
#     bx4.legend(fontsize=8, markerscale=0.5)
# =============================================================================
    bx4.plot(num_pc, cluster_distances, "or")
    
    fig2.savefig(absolute_dir+"results/graphs/cluster_metrics/"+boneage_or_psp+"_dataset_"+str(dataset_num)+".png", bbox_inches='tight')

#%% Main Code for Getting Effects of PCs on Clustering
def main(chosen_pcs, num_cluster=4, n_iter=1, random_state=None, include_elbow=False):
    global df_test, pca_model, model_goal
    
    df_data=df_test.copy()
    
    #chosen_pcs=[h for h in range(1, pca_train.max_pcs)]
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
        cluster_train=pca_model.pcs_train.loc[:,:g]
        cluster_val=pca_model.pcs_test.loc[:,:g]
        
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
                
# =============================================================================
#         cv_individuals=(df_cluster_accuracies_std/df_cluster_accuracies).values
# =============================================================================
        
# =============================================================================
#         #Get weighted mean CV
#         cv_weighted_mean=np.average(cv_individuals, weights=sorted_cluster_sizes)
# =============================================================================
        
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
            
col_indices=[str(i) for i in range(512)]

num_clusters=int(input("Number of Clusters: "))

which_datasets=int(input("One or All datasets (1/0) | "))

elbow_bool=int(input("Include Elbow Plot (1/0) | "))
if elbow_bool not in [1,0]:
    raise "Invalid Input!"

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
    df_test=df[df.phase=="val"]
    
    df_train_data=df_train.loc[:,col_indices]
    df_test_data=df_test.loc[:,col_indices]
    
    #Get, Plot and Save results
    pca_model=pca()
    pca_model.compute(df_train_data,df_test_data)
    
    chosen_pcs=[1,2,3,5,10,15,20,30,50,70]                                     #CHANGE THIS
                                                                               #CHANGE n_cluster, n_iter
    cv, num_pc, cluster_accuracies, cluster_distances, cluster_metrics, optimal_ks=main(chosen_pcs, 
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
    
    #Save Results
# =============================================================================
#     df_results=pd.DataFrame([cv, num_pc, cluster_accuracies, cluster_distances]).transpose()
#     df_results=df_results.rename({0:"cv", 1:"num_pc", 2:"cluster_accuracy", 3:"cluster_distance"}, axis=1)
#     df_results.to_csv(absolute_dir+"results/dataset/"+boneage_or_psp+"_dataset_"+str(i)+".csv", index="False")
# =============================================================================





#%%FOR CONSIDERATION
    #Get MNIst image features
    
    #compare feature selection on image features vs. feature selection on PCs

    #compare other DimRed / feature selection methodologies to results
    
    #what is the effect of clustering on features from a highly overfit model versus an underfit model
        #CV of clusters is always going to be dependent on the accuracy of the model

    #try getting a graph of absolute difference from dataset CV

#DONE
    #compare standardscaler (with_std=False)
        #decreases euclidean distance metrics
        #increases spread of testing accuracy, percent explained variance
    #Do elbow method to get optimal number of clusters
        #compare different num of clusters
