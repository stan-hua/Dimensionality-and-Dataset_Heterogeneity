import numpy as np
import pandas as pd
import os
import ast

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin

results_dir="/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/results/dataset"
home_dir="/home/stanley_hua/scripts/pca_clustering/"

useful_bool="useful"

#Get csv file paths
filenames=[]
for root, dirs, files in os.walk(results_dir+"/"+useful_bool, topdown=False):
   for name in files:
      filenames.append(name)

for filename in filenames:
    df=pd.read_csv(results_dir+"/"+useful_bool+"/"+filename)
    df["dataset"]=filename
    
    try:
        full_df=pd.concat([full_df, df])
    except:
        full_df=df

try:
    full_df=full_df.drop("Unnamed: 0", axis=1)
except:
    pass
#%%Data Parsing
#Get MAD and IQR of Cluster Accuracies
full_df["cluster_acc_mad"]=full_df["cluster_accuracy"].map(ast.literal_eval).map(stats.median_absolute_deviation)
full_df["cluster_acc_iqr"]=full_df["cluster_accuracy"].map(ast.literal_eval).map(stats.iqr)

#Parse Max, Median, Min from Cluster Distance
full_df["cluster_distance"]=full_df["cluster_distance"].map(ast.literal_eval)
full_df["cluster_distance_max"]=full_df["cluster_distance"].map(lambda x: x[0])
full_df["cluster_distance_median"]=full_df["cluster_distance"].map(lambda x: x[1])
full_df["cluster_distance_min"]=full_df["cluster_distance"].map(lambda x: x[2])

#%% FUNCTIONS
def anova_table(dv, iv="num_pc"):
    global full_df
    model = ols('%s ~ C(%s)' % (dv, iv), data=full_df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

#%% Analysis

def test(dv):
    global full_df
    
    print(stats.shapiro(full_df[dv].values))
    print(pingouin.homoscedasticity([group[dv].values for name, group in full_df.groupby("num_pc")],method="bartlett"))
    print(anova_table(dv))
    print(stats.kruskal(*[group[dv].values for name, group in full_df.groupby("num_pc")]))

response=['cv', 'cluster_acc_mad', 'cluster_acc_iqr', 'cluster_distance_max', 'cluster_distance_median', 'cluster_distance_min']



for var in response: 
    print(var)
    print("------------------------------------------------------------")
    test(var)
    print()
    print()
    














