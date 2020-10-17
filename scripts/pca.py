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
            x_axis=np.fromiter(range(1,self.model.n_components_+1))
            
            plt.plot(x_axis, self.model.explained_variance_ratio)
            plt.xlabels("PCA Components")
            plt.ylabels("Percent Explained Variance")
            plt.ticks(np.fromiter(range(self.model.n_components_)))
            plt.show(x_axis)
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



#PCA
df=pd.read_csv()
pca_model=pca(df)
pca_model.validate_assumptions()
pca_model.compute()
pca_model.graph()