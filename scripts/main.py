from __future__ import annotations

import os
import random
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns
from scipy.stats import mode, variation
from sklearn.decomposition import PCA

from clustering import *
from pc_selection import PrincCompSelection
from pca import MyPCA

sns.set_style("white")

# File Paths
absolute_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-" \
               "Project/"
# remote_dir = "/home/stanley_hua/scripts/pca_clustering/"
data_dir = "data/"


# CLASS: Inputs for Analysis
class Inputs:
    paths: list
    model_goal: str
    num_cluster: int
    which_datasets: Union[list, int]
    elbow_bool: bool
    exclude_train: int
    chosen_features: list
    save_bool: str
    random_seed: Optional[int]

    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_train_data: pd.DataFrame
    df_test_data: pd.DataFrame
    col_indices: np.array

    _max_pcs: Optional[int]

    def __init__(self, paths: list, model_goal: str) -> None:
        """
        ==Representational Invariants==:
            num_clusters > 0
            which_dataset in [1, 0]
            elbow_bool in [1, 0]
            exclude_train in [1, 0]
            min(df_test.shape[0], df_test.shape[1]) >= len(chosen_features) > 0
        """
        # GLOBAL: Paths
        self.paths = paths
        self.model_goal = model_goal

        # INPUT: Define number of clusters
        # self.num_cluster = int(input("Number of Clusters: "))
        self.num_cluster = 4

        # INPUT: Choose which datasets to iterate
        which_datasets = int(input("One or All datasets ** 1: One, 0: All\n"))

        if which_datasets == 0:
            self.which_datasets = list(range(len(paths)))
        else:
            self.which_datasets = random.sample(range(len(paths)), 1)

        # INPUT: Get Elbow Plot?
        # elbow_bool = bool(int(input("Include Elbow Plot **1: Yes, 0: No\n")))
        self.elbow_bool = bool(0)

        # INPUT: Use Test Set Only? (So only test set used)
        self.exclude_train = int(input("Only Test Set **1: Yes, 0: No\n"))

        # INPUT: Choose Features
        self.chosen_features = list(range(1, 71))

        # INPUT: Save Results
        self.save_bool = input("Save Results? (Y/N) ")

        # INPUT: Random Seed
        random_seed_bool = input("Random Seed? (Y/N) ")
        if random_seed_bool == "Y":
            self.random_seed = None
        else:
            self.random_seed = int(input("Choose int for random seed: "))

        # ERROR HANDLING
        if (self.elbow_bool not in [1, 0] or
                self.exclude_train not in [1, 0]):
            print("Invalid Input! Restarting...")
            self.__init__(self.paths)

    def get_df_split(self, dataset_num: int) -> Tuple[pd.DataFrame,
                                                      pd.DataFrame]:
        """
        PRECONDITIONS:
            csv files containing total df contain 1) "phase" variable
            corresponding to 'train' and 'val' sets.

        Parameters
        ----------
        dataset_num : int
            Refers to index in global variable <paths>.

        Returns
        -------
        pd.DataFrame
            Contains training data.
        pd.DataFrame
            Contains testing data.

        """
        # Read CSV
        df = pd.read_csv(self.paths[dataset_num], index_col=False)
        try:
            df = df.drop("Unnamed: 0", axis=1)
        except KeyError:
            pass

        # Split Train and Test Data
        try:
            self.df_train = df[df.phase == "train"]
            self.df_test = df[df.phase == "val"]
        except AttributeError:
            # If data does not contain train/val split
            self.df_train = df
            self.df_test = df

        # Get number of feature cols
        total_cols = self.df_test.shape[1]
        while True:
            if str(total_cols) in self.df_test.columns:
                break
            total_cols -= 1

        self.col_indices = np.array([i for i in range(total_cols + 1)])

        # Try converting str column idx to int
        try:
            self.df_train = self.df_train.rename(columns=dict(zip(
                self.col_indices.astype(str), self.col_indices)))
            self.df_test = self.df_test.rename(columns=dict(zip(
                self.col_indices.astype(str), self.col_indices)))
        except BaseException:
            pass
        # Get only features
        self.df_train_data = self.df_train.loc[:, self.col_indices]
        self.df_test_data = self.df_test.loc[:, self.col_indices]

        return self.df_train_data, self.df_test_data

    def get_max_pc_features(self) -> None:
        """Return max number of principal components possible, based on
        <test_data>.

        Update self.chosen_features to contain exponentially
        increasing numbers of <chosen_features>, starting from 1-10."""
        chosen_features = list(range(1, 11))
        self._max_pcs = min(self.df_test_data.shape[0], 512)
        for i in range(100, 0, -1):
            if int(self._max_pcs / i) > 10 and \
                    int(self._max_pcs / i) not in chosen_features:
                chosen_features.append(int(self._max_pcs / i))

        self.chosen_features = chosen_features


# CLASS: Storage of Results
class Results:
    """Results object to store results from iterated clustering.

    ==Attributes==:
        _inputs: Inputs object

        cluster_performances:
            cluster performance based on CNN model predictions
        mean_performance:
            mean model performance of current dataset
        cv_performance:
            Coefficient of Variation of cluster performances
        _cluster_performance_flattened:
            preprocessed CV values
        min_mode_cv:
            Number of PCs to include based on Minimum Mode CV of model
            performances

        centroid_distances:
            Euclidean distance between centroids
        sil_score:
            Silhouette Coefficient
        cal_har_score:
            Calinski-Harabasz Score
        dav_bou_score:
            Davies Bouldin Index
        optimal_ks:
            optional array containing number of clusters suggested by elbow
            plot of distortion score
    """
    _inputs: Inputs
    cluster_performances: np.array
    mean_performance: int
    cv_performance: np.array
    _cluster_performance_flattened: np.array
    min_mode_cv: int
    centroid_distances: np.array
    sil_score: np.array
    cal_har_score: np.array
    dav_bou_score: np.array
    optimal_ks: np.array

    def __init__(self, inputs: Inputs):
        """Initialize Results object."""
        self._inputs = inputs

    def store_cluster_results(self, iterated_cluster_results: tuple):
        """Store results from iterated clustering."""
        # Unpack Results
        self.cluster_performances = iterated_cluster_results[0]
        self.centroid_distances = iterated_cluster_results[1]
        self.optimal_ks = iterated_cluster_results[3]

        clustering_metrics = iterated_cluster_results[2]
        self.sil_score = clustering_metrics[0]
        self.cal_har_score = clustering_metrics[1]
        self.dav_bou_score = clustering_metrics[2]

    def assess_fold_performance(self) -> None:
        """Calculate Coefficient of Variation of model performance of clusters
        for each iteration.

        ==Precondition==:
            - store_cluster_results has been called.
        """
        # Get Fold Mean Accuracy for Plotting [Regression vs. Classification]
        if self._inputs.model_goal == "regression":
            def regression_error(x):
                """Return regression error between prediction and label"""
                return np.sqrt(((x.predictions - x.labels) ** 2))

            self._inputs.df_test[
                "pred_performance"] = self._inputs.df_test.apply(
                regression_error, axis=1)
        else:
            self._inputs.df_test["pred_performance"] = (
                    self._inputs.df_test.predictions ==
                    self._inputs.df_test.labels)

        self.mean_performance = self._inputs.df_test["pred_performance"].mean()

        # PREPROCESSING: Get CV of Cluster Accuracies
        cluster_performances_flattened = np.array([])
        cv_accuracy = np.array([])
        for arr in self.cluster_performances:
            cluster_performances_flattened = np.append(
                cluster_performances_flattened, arr)
            cv_accuracy = np.append(cv_accuracy, variation(arr))

        self.cv_performance = cv_accuracy
        self._cluster_performance_flattened = cluster_performances_flattened

    def get_min_mode_cv(self, round_to: int = None, verbose: bool = False):
        """Calculate minimum mode Coefficient of Variation. If <round_to> is
        not None, round CVs to <round_to> decimals.

        ==Precondition==:
            - assess_fold_performance has been called.
        """
        if round_to is None:
            self.min_mode_cv = self._inputs.chosen_features[
                np.where(self.cv_performance ==
                         mode(self.cv_performance).mode)[0][0]]
        else:
            self.min_mode_cv = self._inputs.chosen_features[
                np.where(self.cv_performance.round(round_to) ==
                         mode(self.cv_performance.round(round_to)).mode)[0][0]]

        if verbose:
            print("CHOOSE TOP: ", self.min_mode_cv)


# FUNCTION: Iterate Methodology
def iterative_clustering(inputs: Inputs,
                         pca_model: PCA,
                         n_iter=300,
                         method: str = "kmeans") -> Tuple[np.array, np.array,
                                                          Tuple[np.array,
                                                                np.array,
                                                                np.array],
                                                          Optional[np.array]]:
    # Accumulators
    cluster_performances = []
    centroid_distance = []
    sil_accumulator = []
    cal_har_accumulator = []
    dav_bou_score_accumulator = []
    optimal_ks = []

    cluster_model = Clustering(inputs.num_cluster, n_iter,
                               inputs.random_seed, method)

    # Control for maximum number of PCs
    max_pcs = pca_model.get_max_pc()
    # if max_pcs < max(inputs.chosen_features):
    inputs.chosen_features = range(1, max_pcs + 1)

    # Iterate between Number of Principal Components Kept
    for num_kept in inputs.chosen_features:
        # Get train and test data
        cluster_train = pca_model.pcs_train.loc[:, :num_kept - 1]
        cluster_val = pca_model.pcs_test.loc[:, :num_kept - 1]

        # Fit clustering
        cluster_model.fit(cluster_train)
        # Predict/assign clustering (for test set)
        cluster_prediction = cluster_model.predict(cluster_val)

        # Get Cluster Performances
        cluster_performances.append(cluster_model.get_cluster_performances(
            inputs.df_test.copy(),
            cluster_prediction,
            num_kept,
            inputs.num_cluster,
            model_goal=inputs.model_goal))

        # Get Intrinsic Clustering Performance
        intrinsic_metrics = cluster_model.evaluate_clustering()
        sil_accumulator.append(intrinsic_metrics[0])
        cal_har_accumulator.append(intrinsic_metrics[1])
        dav_bou_score_accumulator.append(intrinsic_metrics[2])

        # Get Mean Distance between Cluster Centroids
        centroid_distance.append(cluster_model.get_centroid_distance())

        # OPTIONAL: Elbow Plot to determine Optimal Number of K Clusters
        if inputs.elbow_bool:
            optimal_ks.append(cluster_model.elbow_plot())

    return (cluster_performances, centroid_distance,
            (sil_accumulator, cal_har_accumulator, dav_bou_score_accumulator),
            optimal_ks)


# Creating Plots
def create_plots(inputs: Inputs, pca_model: PCA,
                 results: Results, dataset_num: int):
    global paths, absolute_dir, data_dir
    # PREPROCESSING: X values <chosen_features> for Plotting
    chosen_features_repeated = np.array([])
    idx = 1
    for num_acc in np.array([len(indiv_acc) for indiv_acc in
                             results.cluster_performances]):
        chosen_features_repeated = np.append(chosen_features_repeated,
                                             np.array([idx] * num_acc))
        idx += 1

    # PREPROCESSING: Create New Figure Titles
    new_title = paths[dataset_num].replace(absolute_dir +
                                           data_dir, "")
    new_title = new_title.replace(".csv", "").replace("\\", " || ")

    # Matplotlib Settings
    plt.rc('font', family='serif')
    rcParams.update({'figure.autolayout': True})

    # FIGURE: General plots [CV, Test Accuracy/RMSE, % Variance Explained]
    fig = plt.figure()
    fig.suptitle(new_title)
    ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(122)

    # SUBPLOT: CV Accuracy vs. # of Principal Components
    df_cv = pd.DataFrame({"num_features": inputs.chosen_features,
                          "cv": results.cv_performance})
    idx = (df_cv.cv == mode(results.cv_performance).mode[0])

    ax1.set_xlabel('Number of Principal Components')
    ax1.set_ylim((min(0, min(results.cv_performance)), round(np.nan_to_num(
        results.cv_performance).max(), 1) + 0.1))
    ax1.set_ylabel('Coefficient of Variation')
    ax1.scatter(df_cv["num_features"], df_cv["cv"],
                color='black', alpha=.5, s=30)
    ax1.scatter(df_cv["num_features"].loc[idx], df_cv["cv"].loc[idx],
                color="darkred", s=30, alpha=0.5, label="Mode")
    ax1.legend()

    # SUBPLOT: Percent Explained Variance vs. Number of Principal Components
    explained_variance = pca_model.get_cum_variance()

    # ax3.set_title("Percent Variance Explained vs. Number of PCs")
    ax3.set_xlabel('Number of Principal Components')
    ax3.set_ylabel('Percent Explained Variance')
    ax3.set_ylim((0, 1))
    ax3.plot(explained_variance, marker="o", markersize=3, alpha=0.7)

    # FIGURE: Cluster Metrics
    fig2 = plt.figure()
    fig2.suptitle("Intrinsic Clustering Metrics")
    bx1 = fig2.add_subplot(221)
    bx2 = fig2.add_subplot(222)
    bx3 = fig2.add_subplot(223)
    bx4 = fig2.add_subplot(224)

    bx1.set_xticks([])
    bx2.set_xticks([])
    bx3.tick_params(axis='x', labelsize=8)
    bx4.tick_params(axis='x', labelsize=8)
    bx1.tick_params(axis='y', labelsize=8)
    bx2.tick_params(axis='y', labelsize=8)
    bx3.tick_params(axis='y', labelsize=8)
    bx4.tick_params(axis='y', labelsize=8)

    bx1.set_ylabel('Silhouette Coefficient')
    bx2.set_ylabel('Calinski-Harabasz Index')
    bx3.set_xlabel('Number of Principal Components')
    bx3.set_ylabel('Davies-Bouldin Index')
    bx4.set_xlabel('Number of Principal Components')
    bx4.set_ylabel('''Euclidean Distance
    between Centroids''')

    # SUBPLOT: Silhouette Coefficient
    bx1.scatter(inputs.chosen_features, results.sil_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown")

    # SUBPLOT: Calinski-Harabasz Index
    bx2.scatter(inputs.chosen_features, results.cal_har_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown")

    # SUBPLOT: Davies-Bouldin Index
    bx3.scatter(inputs.chosen_features, results.dav_bou_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown")

    # SUBPLOT: Euclidean Distance Between Centroids
    # bx4.scatter(inputs.chosen_features, results.centroid_distances,
    #             marker="o", s=12, alpha=0.7,
    #             c="tab:brown")

    if inputs.save_bool == "Y":
        # Check if Folders Exist
        results_dir = absolute_dir + "results/graphs/" + dataset_used
        general_dir = results_dir + "/general"
        cluster_metrics_dir = results_dir + "/cluster_metrics"
        try:
            os.mkdir(results_dir)
            os.mkdir(general_dir)
            os.mkdir(cluster_metrics_dir)
        except:
            try:
                os.mkdir(general_dir)
                os.mkdir(cluster_metrics_dir)
            except:
                pass
        fig.savefig(general_dir + "/" + dataset_used + "_dataset_" +
                    str(dataset_num) + ".png", bbox_inches='tight')
        fig2.savefig(cluster_metrics_dir + "/" + dataset_used + "_dataset_" +
                     str(dataset_num) + ".png", bbox_inches='tight')

        # FIGURE: Automatic Elbow Method for KMeans Clustering
        if inputs.elbow_bool:
            try:
                os.mkdir(absolute_dir + "results/graphs/" +
                         dataset_used + "/optimal_k")
            except:
                pass
            plt.figure()
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Optimal Number of KMeans Clusters')
            plt.xticks(fontsize=7)
            plt.ylim((0, 10))
            plt.plot(inputs.chosen_features, results.optimal_ks,
                     "^--", markersize=5, alpha=0.7)
            plt.title(new_title, y=1.05)
            plt.savefig(absolute_dir + "results/graphs/" + dataset_used +
                        "/optimal_k/" + new_title.replace("||", "-") + ".png",
                        bbox_inches='tight')


def get_pca_model(inputs: Inputs) -> MyPCA:
    """Return fitted PCA object based on <inputs>."""
    # PCA: Fit & Transform
    pca_model = MyPCA(inputs.random_seed)

    if inputs.exclude_train != 1:
        pca_model.compute(inputs.df_train_data, inputs.df_test_data,
                          whole=False, with_scaler=True, with_std=False)
    else:
        pca_model.compute(inputs.df_test_data, inputs.df_test_data,
                          whole=False, with_scaler=True, with_std=False)

    return pca_model


def get_results(inputs: Inputs, iter_results: tuple) -> Results:
    """Return Results object based on inputs and <iter_results>.
        - stores iterative clustering results
        - calculate and store CNN model performance of each cluster
    """
    # Store Results
    results = Results(inputs)
    results.store_cluster_results(iter_results)
    results.assess_fold_performance()
    results.get_min_mode_cv()

    return results


def main(inputs: Inputs,
         dataset_num: int,
         df_selection_methods: pd.DataFrame = None,
         method: str = "kmeans") -> Tuple[Results,
                                          pd.DataFrame]:
    """Run Principal Components Analysis + Iterated Clustering of PCs for
    <dataset_num>. Return results and dataframe containing recommended number
    of PCs to keep based on selection methods in pc_selection.py.

    If <inputs>.save_bool is True, save dataframe of results with corresponding
    dataset number, and save plots generated.
    """
    if df_selection_methods is None:
        df_selection_methods = pd.DataFrame()

    # Get training and test data & Include only features
    inputs.get_df_split(dataset_num)
    # PCA: Transform Data
    pca_model = get_pca_model(inputs)

    # CLUSTER: PCA-Transformed Data [Iterate over number of features kept.]
    iterated_cluster_results = iterative_clustering(inputs, pca_model,
                                                    100, method=method)

    # Get and store results
    results = get_results(inputs, iterated_cluster_results)

    # PCA: Principal Components Selection Methods
    pc_selection = PrincCompSelection(inputs, pca_model)
    df_selection = pc_selection.select_pcs(idx=paths[dataset_num].replace(
        absolute_dir + data_dir, ""))
    # Add Minimum Mode CV
    df_selection["Minimum Mode CV"] = results.min_mode_cv
    df_selection_methods_new = pd.concat([df_selection_methods, df_selection])

    # Create Plots
    create_plots(inputs, pca_model, results, dataset_num)

    # Save Results
    # if inputs.save_bool == "Y":
    #     df_results = pd.DataFrame([inputs.chosen_features,
    #                                results.cv_performance,
    #                                results.cluster_performances,
    #                                results.centroid_distances]).transpose()
    #     df_results = df_results.rename({0: "features_kept",
    #                                     1: "cv",
    #                                     2: "cluster_performances",
    #                                     3: "mean_centroid_distance",
    #                                     }, axis=1)
    #     df_results.to_csv(absolute_dir + "results/dataset/" +
    #                       dataset_used + "_dataset_" +
    #                       str(dataset_num) + ".csv",
    #                       index=False)

    return results, df_selection_methods_new


# CLIENT CODE
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

    # Get csv file paths
    paths = []
    for root, dirs, files in os.walk(absolute_dir + data_dir + dataset_used,
                                     topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))
    inputs = Inputs(paths, model_goal)

    # Iterate over Seeds
    df_selection_methods = pd.DataFrame()

    # Loop over the datasets (folds)
    for dataset_num in inputs.which_datasets:
        results, df_selection_methods = main(inputs,
                                             dataset_num,
                                             df_selection_methods,
                                             method="kmeans")

    # if inputs.save_bool == "Y":
    #     df_selection_methods.to_csv(absolute_dir + "results/pc_selection/" +
    #                                 dataset_used + f"-{inputs.random_seed}" +
    #                                 ".csv", index=False)
