import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo    

class pca:
    'Principal Component Analysis Object'
    def __init__(self, df):
        self.df=df
        self.pcs_df=None
        self.model=None
        
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
            
            #x_axis=np.fromiter(range(1,self.model.n_components_+1), dtype="int64")
            x_axis=np.fromiter(range(1,num_useful+1), dtype="int64")
            
            plt.plot(x_axis, self.model.explained_variance_ratio_[:num_useful])
            plt.xlabel("PCA Components")
            plt.ylabel("Percent Explained Variance")
            #plt.xticks(np.fromiter(range(self.model.n_components_), dtype="int64"))
            plt.xticks(np.fromiter(range(1,num_useful), dtype="int64"))
            plt.show()
        else:
            raise "Method compute() has not been passed!"
    
    def validate_assumptions(self):
        """
        Prints out results of Bartlett's Test and KMO Test.
        """
        #Bartlett's Test of Sphericity      #Are the variables correlated with one another?
        chi_square_value,p_value=calculate_bartlett_sphericity(self.df)
        #KMO Test for sampling adequacy     #Is most variance common variance?
        kmo_all,kmo_model=calculate_kmo(self.df)
    
        #Printout
        if p_value > 0.05 and kmo_model < 0.6:
            print("Fails both tests!\n p-value: %d \n KMO value: %d" % p_value,kmo_model)
        elif p_value > 0.05:
            print("Fails Bartlett's Test of Sphericity \n p-value: %d" % p_value)
        elif kmo_model < 0.6:
            print("Fails the Kaiser-Meyer-Olkin Test \n KMO Value: %d" % kmo_model)
        else:
            print("Assumptions are valid!\n p-value: %d \n KMO value: %d" % p_value,kmo_model)

#%% Test Code
import os
from scipy.stats import bartlett
import pingouin

absolute_dir="/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/"
data_dir="data/"
boneage_or_psp="psp_plates"

paths=[]
for root, dirs, files in os.walk(absolute_dir+data_dir+boneage_or_psp, topdown=False):
   for name in files:
      paths.append(os.path.join(root, name))

col_indices=[str(i) for i in range(512)]

for i in range(len(paths)):
    df=pd.read_csv(paths[i], index_col=False)
    try:
        df=df.drop("Unnamed: 0", axis=1)
    except:
        pass
    
    
    df_data=df.loc[:,col_indices]
    pca_model=pca(df_data)
    #pca_model.validate_assumptions()
    #pingouin.homoscedasticity(df_data,method="bartlett")
    pca_model.compute()
    pca_model.graph()



















