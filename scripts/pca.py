import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo    
from scipy.stats import bartlett
import scipy
import pingouin
import os

#File Paths
absolute_dir="/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/"
data_dir="data/"

#Dataset
boneage_or_psp="psp_plates"

#Get csv file paths
paths=[]
for root, dirs, files in os.walk(absolute_dir+data_dir+boneage_or_psp, topdown=False):
   for name in files:
      paths.append(os.path.join(root, name))

#%%
class pca:
    'Principal Component Analysis Object'
    def __init__(self, df):
        self.df=df
        self.pcs_df=None
        self.model=None
        self.num_graphed=None
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
    
    def graph(self):
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
            a=self.model.explained_variance_ratio_>0.01
            num_useful=len(a[a])
            
            #Cumulative
            
            #x_axis=np.fromiter(range(1,self.model.n_components_+1), dtype="int64")
            x_axis=np.fromiter(range(1,num_useful+1), dtype="int64")
            
            plt.plot(x_axis, self.model.explained_variance_ratio_[:num_useful])
            plt.xlabel("Num of Principal Components")
            plt.ylabel("Percent Explained Variance")
            #plt.xticks(np.fromiter(range(self.model.n_components_), dtype="int64"))
            plt.xticks(np.fromiter(range(1,num_useful), dtype="int64"))
            plt.show()
            self.num_graphed=num_useful
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
        elif p_value > 0.05:
            print("Fails Bartlett's Test of Sphericity \n p-value: %d" % p_value)
        elif kmo_model < 0.6:
            print("Fails the Kaiser-Meyer-Olkin Test \n KMO Value: %d" % kmo_model)
        elif p_value < 0.05 and kmo_model > 0.6:
            print("Assumptions are valid!\n p-value: %d \n KMO value: %d" % (p_value,kmo_model))
        else:
            print("NaN Calculated!")

#%% Test Code
            
col_indices=[str(i) for i in range(512)]

for i in range(len(paths)):
    df=pd.read_csv(paths[i], index_col=False)
    try:
        df=df.drop("Unnamed: 0", axis=1)
    except:
        pass
    
    df_train=df[df.phase=="train"]
    df_val=df[df.phase=="val"]
    
    #GET TRAINING DATA & VAL & TEST DATA
    df_data=df.loc[:,col_indices]
        
    pca_model=pca(df_data)
    pca_model.validate_assumptions()
    pca_model.compute()
    pca_model.graph()
    
    num_useful=pca_model.num_graphed
    
    cv=[]
    num_pc=[]
    for g in range(1, num_useful):
        #Fit Training, Predict Test
        kmeans=KMeans(n_clusters=4, random_state=0).fit_predict(pca_model.pcs_df[:g])
        
        print(str(len(pca_model.pcs_df.columns)-g) + " Principal Components")
        print(pd.Series(kmeans).value_counts())
        print()
    
        cluster_num=pd.Series(kmeans).value_counts().index.to_list()
        values=pd.Series(kmeans).value_counts().values
        # plt.bar(cluster_num, values, align="edge")
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False)
        # plt.title("KMeans with " + str(len(pca_model.pcs_df.columns)-g) + " Principal Components")
        # plt.show()
        
        cv.append(scipy.stats.variation(values))
        num_pc.append(len(pca_model.pcs_df.columns)-g)
    
    plt.plot(num_pc, cv)
    plt.xlabel("Num of Principal Components")
    plt.ylabel("Coefficient of Variation")
    plt.title(paths[i].replace(absolute_dir+data_dir,""))
    plt.ylim((0, 0.5))
    plt.show()












