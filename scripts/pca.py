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

#File Paths
absolute_dir="/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/"
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
    
    def __init__(self, df):
        self.df=df
        self.pcs_df=None
        self.model=None
        self.max_pcs=None
        
        
    def compute(self):
        """
        1) Standardizes data. 
        2) Runs Principal Component Analysis. 
        3) Saves model and principal components (as Pandas DataFrame) to object
        
        Returns
        -------
        None.

        """
        #Standardize Data
        df_normalized=StandardScaler().fit_transform(self.df)
        #Create PCA object. Fit and transform
        pca_model=PCA()
        principalComponents=pca_model.fit_transform(df_normalized)
        #Save instance and principal components
        self.pcs_df=pd.DataFrame(principalComponents)
        self.model=pca_model
        self.max_pcs=len(self.pcs_df.columns)-1
    
    def graph(self, chosen_pcs=None):
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
            # a=self.model.explained_variance_ratio_>0
            # num_useful=len(a[a])
            
            if chosen_pcs==None or len(chosen_pcs)>20:
                chosen_pcs=[int(self.max_pcs/i) for i in range(15,0,-2)]

            chosen_pcs_unique=np.unique(chosen_pcs)
            
            #Cumulative
            variance_explained=pd.Series(self.model.explained_variance_ratio_[:]).cumsum()
            
            y_var=variance_explained.iloc[chosen_pcs_unique].values
            #x_axis=np.fromiter(range(1,num_useful+1), dtype="int64")
            
            plt.plot(chosen_pcs_unique, y_var,'r')
            plt.xlabel("Number of Principal Components")
            plt.ylabel("Percent Explained Variance")
            #plt.xticks(np.fromiter(range(1,num_useful), dtype="int64"))
            plt.xticks(chosen_pcs_unique)
            plt.ylim((0, 1))
            plt.show()
            
            return (chosen_pcs_unique, y_var)
        else:
            raise "Method compute() has not been passed!"
    
    def validate_assumptions(self):
        """
        Prints out results of Bartlett's Test and KMO Test.
        """
        #Bartlett's Test of Sphericity      #Are the variables correlated with one another?
            #chi_square_value,p_value=calculate_bartlett_sphericity(self.df)
        sphericity_test=pingouin.homoscedasticity(self.df,method="bartlett")
        p_value=sphericity_test["pval"].iloc[0]
        
        #KMO Test for sampling adequacy     #Is most variance common variance?
        kmo_all,kmo_model=calculate_kmo(self.df)
    
        #Printout
        if p_value > 0.05 and kmo_model < 0.6:
            print("Fails both tests!\n p-value: %d \n KMO value: %d" % (p_value,kmo_model))
            result="Invalid"
        elif p_value > 0.05:
            print("Fails Bartlett's Test of Sphericity \n p-value: %d" % p_value)
            result="Invalid"
        elif kmo_model < 0.6:
            print("Fails the Kaiser-Meyer-Olkin Test \n KMO Value: %d" % kmo_model)
            result="Invalid"
        elif p_value < 0.05 and kmo_model > 0.6:
            print("Assumptions are valid!\n p-value: %d \n KMO value: %d" % (p_value,kmo_model))
            result="Valid"
        else:
            print("NaN Calculated!")
            result="Invalid"
        
        return result
        
#%% Display Max-Min Distance
def get_cluster_distance(cluster_centers):
    #Get distances
    dists = euclidean_distances(cluster_centers)
    
    #Compute Max Distance, Average Distance and Minimum Distance
    tri_dists = dists[np.triu_indices(4,1)]
    max_dist, min_dist = tri_dists.max(), tri_dists.min()
    
    #Raise Error if Cluster Distance is 0
    max_min=max_dist-min_dist
    if round(max_min)==0:
        raise "Max Distance - Min Distance = 0"
    

#%% Clustering Models
def cluster_kmeans(train_data, test_data, num_clusters=4):
    kmeans_model=KMeans(n_clusters=num_clusters, random_state=0).fit(train_data)    
    cluster_prediction=kmeans_model.predict(test_data)    
    get_cluster_distance(kmeans_model.cluster_centers_)

    # Fuzzy K Means
# =============================================================================
#     fuzzy_model=FuzzyKMeans(k=num_clusters,m=2).fit(train_data)
#     cluster_prediction=pairwise_distances_argmin(test_data,fuzzy_model.cluster_centers_)
# =============================================================================
    
    return cluster_prediction

#%% Test Code
            
col_indices=[str(i) for i in range(512)]

for i in range(len(paths)):
    df=pd.read_csv(paths[i], index_col=False)
    try:
        df=df.drop("Unnamed: 0", axis=1)
    except:
        pass
    
    
    #GET TRAINING DATA & VAL & TEST DATA
    df_train=df[df.phase=="train"]
    df_test=df[df.phase=="val"]


    pca_train=pca(df.loc[:,col_indices])
    
    #Check if Assumptions are valid
    if pca_train.validate_assumptions()=="Valid":
        pca_train.compute()
        
        #CHANGE THIS
        chosen_pcs=[1,2,3,5,10,15,20,30,50,80]
        #chosen_pcs=[h for h in range(1, pca_train.max_pcs)]
        cv=[]
        num_pc=[]
        cluster_accuracies=[]
        #range(1, pca_train.max_pcs)
        for g in chosen_pcs:
            #Get train and test data
            cluster_train=pca_train.pcs_df.loc[:df_train.index[-1]+1,:g]
            cluster_val=pca_train.pcs_df.loc[df_train.index[-1]+1:df_test.index[-1]+1,:g]
            
            #Fit Training, Predict Test
            cluster_prediction=cluster_kmeans(cluster_train,cluster_val)

# Prints out number of observations in each of four clusters            
# =============================================================================
#             print(str(g) + " Principal Components")
#             print(pd.Series(kmeans).value_counts())
#             print()
# =============================================================================
        
            cluster_num=pd.Series(cluster_prediction).value_counts().index.to_list()
            values=pd.Series(cluster_prediction).value_counts().values
            
# Bar plots showing cluster sizes
# =============================================================================
#             plt.bar(cluster_num, values, align="edge")
#             plt.tick_params(
#                 axis='x',          
#                 which='both',      
#                 bottom=False,      
#                 top=False,         
#                 labelbottom=False)
#             plt.title("KMeans with " + str(len(pca_test.pcs_df.columns)-g) + " Principal Components")
#             plt.show()
# =============================================================================

            #Get cluster test accuracies
            df_test["cluster"]=cluster_prediction
            df_test["prediction_bool"]=df_test.predictions==df_test.labels
            df_cluster_accuracies=df_test.groupby(by=["cluster"]).mean()["prediction_bool"]
            
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
            
            
        #CREATING FIGURE:
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
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
        ax2.set_xlabel('Number of Principal Components')
        ax2.set_ylabel('Testing Accuracy')
        ax2.set_ylim((0.7, 1))
        ax2.tick_params(axis='x', labelsize=7)
        
        #FIGURE: Percent Explained Variance vs. Number of Principal Components
        ax3.set_xlabel('Number of Principal Components')
        ax3.set_ylabel('Percent Explained Variance')
        ax3.tick_params(axis='x', labelsize=7)
        ax3.set_ylim((0, 1))
        
        x,y=pca_train.graph(chosen_pcs)
        ax3.plot(x, y)

        
    else:
        print("Dataset: " + paths[i].replace(absolute_dir+data_dir,""))
        print("       with "+str(len(df))+" observations is invalid!")



#FOR CONSIDERATION
    #selecting for len(df)/10 features, and PCA on that
    #another type of dimensionality reduction / feature selection








