import pandas as pd
import numpy as np
import os
from svcca import get_sample_sizes
from scipy.stats import mode

dataset = "psp_plates"
#%% Get Paths to Saved DataFrames
def get_paths(dir="dataset", contains=None) -> pd.Series:
    home_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/results/"
    paths = []
    for root, dirs, files in os.walk(home_dir+dir, topdown=False):
       for name in files:
          paths.append(os.path.join(root, name))
    
    paths_series = pd.Series(paths)
    idx = paths_series.str.contains(contains)
    return paths_series[idx].reset_index(drop=True)

# Get paths for df_results from iterative clustering
results_paths = get_paths(contains=dataset)

# Get paths for number of PCs selection methods
pc_selection_paths = get_paths(dir="pc_selection", contains=dataset)
#%%
# Get dataset number
def get_dataset_num(x: str) -> str:
    "Return dataset number from filename."
    return int(x.replace(".csv", "").split("dataset_")[1])

dataset_nums = results_paths.map(get_dataset_num).tolist()

# Get variance of CV for each dataset
def get_variance_cv(x: str) -> float:
    "Return variance of CV of dataset corresponding to filename."
    df = pd.read_csv(x)
    return np.var(df.cv)

cv_variances = results_paths.map(get_variance_cv).tolist()

# Get mode of CV for each dataset
def get_mode_cv(x: str) -> float:
    "Return variance of CV of dataset corresponding to filename."
    df = pd.read_csv(x)
    return mode(df.cv)

cv_mode = results_paths.map(get_mode_cv).tolist()


# Get CV at CPV > 0.8
    # Get results from selection methods
df_selection = pd.read_csv(pc_selection_paths[0])

def get_cv_cpv(x: str) -> float:
    # Get dataset number
    dataset_num = get_dataset_num(x)
    
    # Get number of pcs for CPV > 0.8
    pcs_cpv = df_selection.loc[dataset_num, "Cum. Perc. Var. (0.8)"]
    
    # Get df_results
    df = pd.read_csv(x)
    idx = df.features_kept == pcs_cpv
    try:
        return df.loc[idx].cv.values[0]
    except:
        return -1

cv_at_cpv = results_paths.map(get_cv_cpv).tolist()

# Get training sample size
training_sample_sizes = get_sample_sizes()[0]

# Readjust training sample size for boneage
if dataset == "boneage":
    for i in range(len(training_sample_sizes)):
        if training_sample_sizes[i] == 2127:
            training_sample_sizes[i] = 2128
        elif training_sample_sizes[i] == 4255:
            training_sample_sizes[i] = 4256
#%% Concatenate Results
df = pd.DataFrame()
df["cv_var"] = cv_variances
df["cv_mode"] = cv_mode
df["cv at CPV > 0.8"] = cv_at_cpv
df["training_size"] = np.array(training_sample_sizes)[dataset_nums]
df.sort_values(by="training_size", inplace=True, ignore_index=True)


print(df.groupby("training_size").mean())
