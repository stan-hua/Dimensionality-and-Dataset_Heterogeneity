import pandas as pd
import numpy as np
import os
from svcca import get_sample_sizes
from scipy.stats import mode, iqr

# File Paths
absolute_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-" \
               "Project/"
data_dir = "data/"


# Get Paths to Saved DataFrames
def get_paths(dir="dataset", contains=None) -> pd.Series:
    home_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/" \
               "PCA-Clustering-Project/results/"
    paths = []
    for root, dirs, files in os.walk(home_dir+dir, topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))

    paths_series = pd.Series(paths)
    idx = paths_series.str.contains(contains)
    return paths_series[idx].reset_index(drop=True)


# Get dataset number
def get_dataset_num(x: str) -> str:
    """Return dataset number from filename."""
    return int(x.replace(".csv", "").split("dataset_")[1])


# Get variance of CV for each dataset
def get_variance_cv(x: str) -> float:
    """Return variance of CV of dataset corresponding to filename."""
    df = pd.read_csv(x)
    idx = (df.cv < 1)

    return np.var(df.loc[idx].cv, ddof=1)


# Get mode of CV for each dataset
def get_mode_cv(x: str) -> float:
    """Return variance of CV of dataset corresponding to filename."""
    df = pd.read_csv(x)
    return mode(df.cv)


# Get range of CV for each dataset
def get_iqr_cv(x: str) -> list:
    """Return [min, max] of CV of dataset corresponding to filename."""
    df = pd.read_csv(x)
    idx = (df.cv < 1)

    return iqr(df.loc[idx].cv)


# Get CV at CPV > 0.8
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


if __name__ == "__main__":
    # INPUT: Dataset and Type of Model (used for feature extraction)
    dataset_choice = int(
        input("DATASET: ** 1: boneage, 2: psp_plates, 3: cifar\n"))
    if dataset_choice == 1:
        dataset_used = "boneage"
    elif dataset_choice == 3:
        dataset_used = "cifar10"
    else:
        dataset_used = "psp_plates"

    # Get Paths for getting sample size
    paths = []
    for root, dirs, files in os.walk(absolute_dir + data_dir + dataset_used,
                                     topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))

    # Get paths for df_results from iterative clustering
    results_paths = get_paths(contains=dataset_used)

    # Get paths for number of PCs selection methods
    pc_selection_paths = get_paths(dir="pc_selection", contains=dataset_used)

    # Get dataset num based on filename
    dataset_nums = results_paths.map(get_dataset_num).tolist()

    # Get variance of Coefficient of Variation, ignoring CV > 1
    cv_variances = results_paths.map(get_variance_cv).tolist()

    # Get mode of Coefficient of Variation
    cv_mode = results_paths.map(get_mode_cv).tolist()

    # Get IQR of Coefficient of Variation, ignoring CV > 1
    cv_iqr = results_paths.map(get_iqr_cv).tolist()

    # Get results from selection methods
    df_selection = pd.read_csv(pc_selection_paths[0])

    # Get Coefficient of Variation at CPV > 0.8
    cv_at_cpv = results_paths.map(get_cv_cpv).tolist()

    # Get training sample size
    training_sample_sizes = get_sample_sizes(paths, dataset_used)[0]

    # Readjust training sample size for boneage
    if dataset_used == "boneage":
        for i in range(len(training_sample_sizes)):
            if training_sample_sizes[i] == 2127:
                training_sample_sizes[i] = 2128
            elif training_sample_sizes[i] == 4255:
                training_sample_sizes[i] = 4256
    # Concatenate Results
    df = pd.DataFrame()
    df["cv_var"] = cv_variances
    df["cv_mode"] = cv_mode
    df["cv_iqr"] = cv_iqr
    df["cv at CPV > 0.8"] = cv_at_cpv
    df["training_size"] = np.array(training_sample_sizes)[dataset_nums]
    df.sort_values(by="training_size", inplace=True, ignore_index=True)

    print(df.groupby("training_size").mean())
