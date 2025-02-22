from main import *
from clustering import Clustering
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
def get_dataset_num(x: str) -> int:
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
def get_cv_cpv(x: str, percent: float) -> float:
    global model_goal
    # Get dataset number
    dataset_num = get_dataset_num(x)

    # Get number of pcs for CPV > 0.8 and CPV > 0.99
    if percent == 0.99:
        pcs_cpv = df_selection.loc[dataset_num, "Cum. Perc. Var. (0.99)"]
    else:
        pcs_cpv = df_selection.loc[dataset_num, "Cum. Perc. Var. (0.8)"]

    # Get df_results
    df = pd.read_csv(x)
    idx = df.features_kept == pcs_cpv
    try:
        return df.loc[idx].cv.values[0]
    except:
        inputs = Inputs(paths)
        inputs.random_seed = 1969
        inputs.get_df_split(dataset_num)

        pca_model = get_pca_model(inputs)

        cluster_model = Clustering(inputs.num_cluster, 100,
                                   inputs.random_seed)
        cluster_model.fit(pca_model.pcs_train.loc[:, :pcs_cpv-1])
        cluster_prediction = cluster_model.predict(
            pca_model.pcs_test.loc[:, :pcs_cpv-1])
        cluster_performances = cluster_model.get_cluster_performances(
            inputs.df_test.copy(),
            cluster_prediction,
            pcs_cpv,
            inputs.num_cluster,
            model_goal=model_goal)
        return variation(cluster_performances)


if __name__ == "__main__":
    # INPUT: Dataset and Type of Model (used for feature extraction)
    dataset_choice = int(
        input("DATASET: ** 1: boneage, 2: psp_plates, 3: cifar\n"))
    if dataset_choice == 1:
        dataset_used = "boneage"
        model_goal = "regression"
    elif dataset_choice == 3:
        dataset_used = "cifar10"
        model_goal = "classification"
    else:
        dataset_used = "psp_plates"
        model_goal = "classification"

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
    df_cv_mode = pd.DataFrame(results_paths.map(get_mode_cv).tolist())
    df_cv_mode = df_cv_mode.apply(lambda x: np.concatenate(x.to_numpy()))
    cv_mode, cv_mode_freq = df_cv_mode["mode"], df_cv_mode["count"]

    # Get IQR of Coefficient of Variation, ignoring CV > 1
    cv_iqr = results_paths.map(get_iqr_cv).tolist()

    # Get results from selection methods
    df_selection = pd.read_csv(pc_selection_paths[0])

    # Get Coefficient of Variation at CPV > 0.8
    cv_at_cpv_80 = results_paths.apply(get_cv_cpv, args=(0.8,)).tolist()
    cv_at_cpv_99 = results_paths.apply(get_cv_cpv, args=(0.99,)).tolist()

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
    df["cv_iqr"] = cv_iqr
    df["cv_mode"] = cv_mode
    df["cv_mode_freq"] = cv_mode_freq
    df["cv at CPV > 0.8"] = cv_at_cpv_80
    df["cv at CPV > 0.99"] = cv_at_cpv_99
    df["training_size"] = np.array(training_sample_sizes)[dataset_nums]
    df.sort_values(by="training_size", inplace=True, ignore_index=True)

    print(df.groupby("training_size").mean())

    df.to_csv(f"{absolute_dir}results/CVs/{dataset_used}.csv", index=False)


dataset_used = "boneage"

df = pd.read_csv(f"{absolute_dir}results/CVs/{dataset_used}.csv")
df["diff_99"] = abs(df.cv_mode - df["cv at CPV > 0.99"])
df["diff_80"] = abs(df.cv_mode - df["cv at CPV > 0.8"])
df.groupby("training_size").mean()
