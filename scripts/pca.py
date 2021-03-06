from __future__ import annotations
from typing import List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import random
import seaborn as sns

from sklearn.decomposition import PCA, SparsePCA, KernelPCA, TruncatedSVD

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster.elbow import kelbow_visualizer

from scipy.stats import variation, mode
from scipy.spatial.distance import cdist

from pc_selection import PrincCompSelection

sns.set_style("white")

# File Paths
absolute_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-"\
               "Project/"
remote_home_dir = "/home/stanley_hua/scripts/pca_clustering/"    #OBSOLETE
data_dir = "data/"

# INPUT: Dataset and Type of Model (used for feature extraction)
dataset_choice = int(input("DATASET: ** 1: boneage, 2: psp_plates, 3: cifar\n"))
if dataset_choice == 1:
    dataset_used = "boneage"
    model_goal = "regression"
elif dataset_choice == 3:
    dataset_used = "cifar10"
    model_goal = "classification"
else:
    dataset_used = "psp_plates"
    model_goal = "classification"

# =============================================================================
# if int(input("DATASET: Model goal is regression
# or classification (1/0) "))==1:
#     model_goal="regression"
# else:
#     model_goal="classification"
# =============================================================================

# Get csv file paths
paths = []
for root, dirs, files in os.walk(absolute_dir + data_dir + dataset_used,
                                 topdown=False):
    for name in files:
        paths.append(os.path.join(root, name))

# Plot Style
sns.set_style("white")

# Random Seeds: [1969, 1974, 2000, 2001]


# CLASS: Inputs for Functions
class Inputs:
    paths: list
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

    def __init__(self, paths: list):
        """
        ==Representational Invariants==:
            num_clusters > 0
            which_dataset in [1, 0]
            elbow_bool in [1, 0]
            exclude_train in [1, 0]
            len(chosen_features) > 0

        Returns
            Tuple[list, int,]
        """
        # GLOBAL: Paths
        self.paths = paths

        #-------------------------------------------------------------------------#
        # INPUT: Define number of clusters
# =============================================================================
#         self.num_cluster = int(input("Number of Clusters: "))
# =============================================================================
        self.num_cluster = 4
        #-------------------------------------------------------------------------#
        # INPUT: Choose which datasets to iterate
# =============================================================================
#         which_datasets = int(input("One or All datasets ** 1: One, 0: All\n"))
# =============================================================================
        which_datasets = 0

        if which_datasets == 0:
            self.which_datasets = list(range(len(paths)))
        else:
            self.which_datasets = random.sample(range(len(paths)), 1)
        #-------------------------------------------------------------------------#
        # INPUT: Get Elbow Plot?
        # elbow_bool = bool(int(input("Include Elbow Plot **1: Yes, 0: No\n")))
        self.elbow_bool = bool(0)
        #-------------------------------------------------------------------------#
        # INPUT: Use Test Set Only? (So only test set used)
        # exclude_train = int(input("Only Test Set **1: Yes, 0: No\n"))
        self.exclude_train = 0

        # INPUT: Choose Features
        # self.chosen_features = [1, 2, 3, 5, 10, 15, 20, 30, 50, 70]
        self.chosen_features = list(range(1, 71))
        # INPUT: Save Results
# =============================================================================
#         self.save_bool = input("Save Results? (Y/N) ")
# =============================================================================
        self.save_bool = "N"
        # INPUT: Random Seed
# =============================================================================
#         random_seed_bool = input("Random Seed? (Y/N) ")
# =============================================================================
        random_seed_bool = "Y"
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
            Contains only training data.
        pd.DataFrame
            Contains only testing data.

        """
        # Read CSV
        df = pd.read_csv(self.paths[dataset_num], index_col=False)
        try:
            df = df.drop("Unnamed: 0", axis=1)
        except BaseException:
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

        self.col_indices = np.array([i for i in range(total_cols+1)])

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
        chosen_features = list(range(1,11))
        self._max_pcs = min(self.df_test_data.shape[0], 512)
        for i in range(100,0,-1):
            if int(self._max_pcs/i) > 10 and \
                    int(self._max_pcs/i) not in chosen_features:
                chosen_features.append(int(self._max_pcs/i))

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
        self.cluster_performances, self.centroid_distances, cluster_metrics, self.optimal_ks = iterated_cluster_results

        (self.sil_score, self.cal_har_score, self.dav_bou_score) = cluster_metrics

    def assess_fold_performance(self):
        """Calculate model performance of clusters.

        ==Precondition==:
            - store_cluster_results has been called.
        """

        global model_goal
        # Get Fold Accuracy [Regression vs. Classification]
        if model_goal == "regression":
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
        #-------------------------------------------------------------------------#
        # PREPROCESSING: Get CV of Cluster Accuracies
        cluster_performances_flattened = np.array([])
        cv_accuracy = np.array([])
        for arr in self.cluster_performances:
            cluster_performances_flattened = np.append(cluster_performances_flattened,
                                                     arr)
            cv_accuracy = np.append(cv_accuracy, variation(arr))

        self.cv_performance = cv_accuracy
        self._cluster_performance_flattened = cluster_performances_flattened

    def get_min_mode_cv(self, round_to: int = None, verbose: bool = False):
        """Calculate minimum mode Coefficient of Variation. If <round_to> is
        not None, round CVs to <round_to> decimals.

        ==Precondition==:
            - assess_fold_performance() has been called.
        """
        if round_to is None:
            self.min_mode_cv = inputs.chosen_features[
                np.where(self.cv_performance ==
                         mode(self.cv_performance).mode)[0][0]]
        else:
            self.min_mode_cv = inputs.chosen_features[
                np.where(self.cv_performance.round(round_to) ==
                         mode(self.cv_performance.round(round_to)).mode)[0][0]]

        if verbose:
            print("CHOOSE TOP: ", self.min_mode_cv)


# CLASS: PCA Subclass
class MyPCA(PCA):
    """Principal Component Analysis Object. Inherits from scikit-learn PCA
    class."""
    _chosen_features: list
    train_scaler: StandardScaler
    test_scaler: StandardScaler
    pcs_train: pd.DataFrame
    pcs_test: pd.DataFrame

    _max_pcs: int

    def __init__(self, chosen_features=None, random_state: int = 2020-12-15) ->\
            None:
        super().__init__(random_state=random_state)
        self._chosen_features = chosen_features

        # if SparsePCA, alpha=0.01
        # if TruncatedSVD, n_components=70
    def compute(self,
                df_train_data, df_test_data,
                whole=False,
                with_scaler=True,
                with_std=False) -> None:
        """
        1) Center/Standardize data.
        2) Runs Principal Component Analysis.
        3) Saves model and principal components (as pd.DataFrame) to object

        Returns
        -------
        None.

        """
        if not whole:
            # Center Data
            if with_scaler:
                self.train_scaler = StandardScaler(with_std=with_std)
                df_train_data = self.train_scaler.fit_transform(df_train_data)
                self.test_scaler = StandardScaler(with_std=with_std)
                df_test_data = self.train_scaler.transform(df_test_data)

            # PCA fit-transform data
            train_pcs = self.fit_transform(df_train_data)
            test_pcs = self.transform(df_test_data)
        else:
            # Concat training and test
            df_whole = pd.concat([df_train_data, df_test_data])

            # Center Data
            if with_scaler:
                self.train_scaler = StandardScaler(with_std=with_std)
                df_whole = self.train_scaler.fit_transform(df_whole)

            # PCA fit then transform data
            self.fit(df_whole)
            train_pcs = self.transform(
                pd.DataFrame(df_whole).loc[:df_train_data.index.tolist()[-1],:])
            test_pcs = self.transform(
                pd.DataFrame(df_whole).loc[df_train_data.index.tolist()[-1]+1:,:])

        # Save instance and principal components
        self.pcs_train = pd.DataFrame(train_pcs)
        self.pcs_test = pd.DataFrame(test_pcs)
        self._max_pcs = len(self.pcs_train.columns) - 1

    def get_cum_variance(self) -> pd.Series:
        """
        Returns
        -------
        Cumulative percent explained variance for all principal components
        """
        return pd.Series(self.explained_variance_ratio_[:]).cumsum()

    def get_total_variance(self, display=False):
        """Return Total Variance."""
        if display:
            print("Sum Eigenvalues/Trace/Total Variance:",
                  self.explained_variance_.sum())
        return self.explained_variance_.sum()

    def get_general_variance(self, display=False) -> np.array:
        """Return General Variance."""
        if not display:
            print("General Variance:",
                  np.linalg.det(self.get_covariance()))
        return np.linalg.det(self.get_covariance())

    def get_noise_variance(self, display=False) -> np.array:
        """Return Noise Variance"""
        if not display:
            print("Noise Variance:",self.noise_variance_)
        return self.noise_variance_

# SVD Helper Functions
    def get_optimal_thresh(self):
        """Return Optimal Singular Value Threshold
        according to Gavish and Donoho 2014."""
        m, n = self.pcs_train.shape
        try:
            isinstance(self.singular_values_, np.ndarray)
        except BaseException:
            raise Exception("SVD Algorithm Not In Use")

        if m == n:
            return 4/math.sqrt(3)
        else:
            B = m/n
            return math.sqrt(2*(B + 1) +
                             (8*B/(B + 1 + math.sqrt(B**2 + 14*B + 1))))


# Clustering
class KMeansCluster:
    def __init__(self,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 num_clusters: int,
                 n_iter: int,
                 random_state_: int = None):
        self._train_data = train_data
        self._test_data = test_data
        self._num_clusters = num_clusters
        self._n_iter = n_iter
        self._random_state = random_state_

        # Initialize KMeans Model
        self.model = KMeans(n_clusters=num_clusters,
                      random_state=random_state_,
                      n_init=n_iter)

        self.model.fit(train_data)

    def predict(self) -> np.array:
        """Return cluster predictions"""
        return self.model.predict(self._test_data)

    def elbow_plot(self,
                   start_num_clusters: int =2,
                   end_num_clusters: int = 10) -> int:
        """Return elbow value. And create elbow plots for KMeans with
        <start_num_clusters> to <end_num_clusters>.
        """
        elbow_model = kelbow_visualizer(
             KMeans(n_clusters=self._num_clusters,
                    random_state=self._random_state,
                    n_init=self._n_iter),
             self._train_data,
             k=(start_num_clusters, end_num_clusters))
        try:
            elbow_k = int(elbow_model.elbow_value_)
        except:
            elbow_k = 0

        return elbow_k

    def evaluate_clustering(self):
        """Return Silouette, Calinski-Harabasz and Davies Bouldin score."""
        sil_score=silhouette_score(self._train_data,
                                   self.model.labels_,
                                   metric='euclidean')

        cal_har_score=calinski_harabasz_score(self._train_data,
                                              self.model.labels_)

        dav_bou_score=davies_bouldin_score(self._train_data,
                                           self.model.labels_)

        return sil_score, cal_har_score, dav_bou_score

    def get_centroid_distance(self):
        """Return mean centroid distances."""
        return np.average(np.min(cdist(self._train_data,
                                       self.model.cluster_centers_,
                                       'euclidean'),
                                 axis=1))



# Main Code for Getting Effects of PCs on Clustering
    def get_cluster_performances(self,
                               df: pd.DataFrame,
                               cluster_prediction: np.array,
                               num_kept: int,
                               num_cluster: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing test samples.
        cluster_prediction : np.array
            Cluster belonging predictions for <df>.

        Returns
        -------
        df_cluster_performances : TYPE
            Contain cluster accuracies.
        """
        global model_goal
        # Getting cluster sizes
        cluster_indiv_num = pd.Series(cluster_prediction)
        cluster_indiv_num = cluster_indiv_num.value_counts().index.to_list()
        cluster_sizes = pd.Series(cluster_prediction).value_counts().values

        # Sorting by cluster sizes
        sorted_cluster_sizes = pd.DataFrame(cluster_sizes,
                                            index=cluster_indiv_num)
        sorted_cluster_sizes = sorted_cluster_sizes.sort_index().values.flatten()

        # Get cluster test accuracies
        df["cluster"] = cluster_prediction
        if model_goal=="regression":
            df["prediction_accuracy"] = df.apply(lambda x: np.sqrt(((x.predictions - x.labels) ** 2)), axis=1)
        else:
            df["prediction_accuracy"] = (df.predictions==df.labels)
        df_cluster_performances = df.groupby(by=["cluster"]).mean()["prediction_accuracy"]

        if (df_cluster_performances==0).sum() > 0:
            zero_idx = np.where(df_cluster_performances==0)
            print("At "+str(num_kept)+ " PCs, the cluster/s "+
                  str(zero_idx[0].tolist())+" with "+
                  str(sorted_cluster_sizes[zero_idx])+" values have 0 accuracy.")

            for cluster_idx in np.where(df_cluster_performances==0)[0]:
                print("Cluster "+str(cluster_idx) +
                      " has images with 0 accuracy at indices: " +
                      str(np.where(cluster_prediction==cluster_idx)[0]))

        #Trying to prevent NA in CV and Test Accuracy
        fix_cluster_performances=[]
        for cluster in range(num_cluster):
            try:
                fix_cluster_performances.append(df_cluster_performances[cluster])
            except:
                fix_cluster_performances.append(-1)

        return fix_cluster_performances


def iterative_clustering(inputs: Inputs,
                         pca_model: PCA,
                         n_iter=1) -> Tuple[np.array, np.array,
                                            Tuple[np.array, np.array, np.array],
                                            Optional[np.array]]:
    global model_goal

    # Accumulators
    cluster_performances = []
    centroid_distance = []
    sil_accumulator = []
    cal_har_accumulator = []
    dav_bou_score_accumulator = []
    optimal_ks = []

    # Iterate between Number of Principal Components Kept
    for num_kept in inputs.chosen_features:
        # Get train and test data
        cluster_train = pca_model.pcs_train.loc[:, :num_kept-1]
        cluster_val = pca_model.pcs_test.loc[:, :num_kept-1]

        # CLUSTER ALGORITHM
        cluster_model = KMeansCluster(cluster_train,
                                      cluster_val,
                                      inputs.num_cluster,
                                      n_iter,
                                      inputs.random_seed)

        # OPTIONAL: Elbow Plot to determine Optimal Number of K Clusters
        if inputs.elbow_bool:
            optimal_ks.append(cluster_model.elbow_plot())

        # Predict Cluster Belonging (for Test Set)
        cluster_prediction = cluster_model.predict()

        # Append Cluster Testing Accuracies
        cluster_performances.append(cluster_model.get_cluster_performances(
            inputs.df_test.copy(),
            cluster_prediction,
            num_kept,
            inputs.num_cluster))

        # CLUSTERING: INTERNAL VALIDATION
        # Get Silhouette, Calinski-Harabasz, Davies Bouldin Metrics (for Test)
        sil_score, cal_har_score, dav_bou_score = cluster_model.evaluate_clustering()
        sil_accumulator.append(sil_score)
        cal_har_accumulator.append(cal_har_score)
        dav_bou_score_accumulator.append(dav_bou_score)

        # Get Mean Distance between Cluster Centroids
        centroid_distance.append(cluster_model.get_centroid_distance())

    return (cluster_performances, centroid_distance,
            (sil_accumulator, cal_har_accumulator, dav_bou_score_accumulator),
            optimal_ks)


# Creating Informative Plots
def create_plots(inputs: Inputs, pca_model: PCA,
                 results: Results, dataset_num: int):
    global paths, absolute_dir, data_dir
    # PREPROCESSING: X values <chosen_features> for Plotting
    chosen_features_repeated = np.array([])
    idx = 1
    for num_acc in np.array([len(indiv_acc) for indiv_acc in
                             results.cluster_performances]):
        chosen_features_repeated = np.append(chosen_features_repeated,
                                             np.array([idx]*num_acc))
        idx += 1

    # PREPROCESSING: Create New Figure Titles
    new_title = paths[dataset_num].replace(absolute_dir +
                                           data_dir,"")
    new_title = new_title.replace(".csv", "").replace("\\", " || ")

    # FIGURE: General plots [CV, Test Accuracy/RMSE, % Variance Explained]
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)

    # Figure Settings
    plt.tight_layout()
    fig.suptitle(new_title, y=1.05)

    # SUBPLOT: CV Accuracy vs. # of Principal Components
    df_cv = pd.DataFrame({"num_features": inputs.chosen_features,
                          "cv": results.cv_performance})
    idx = (df_cv.cv == mode(results.cv_performance).mode[0])

    ax1.scatter(df_cv["num_features"], df_cv["cv"],
                facecolors='gray', alpha=.4, s=50)
    ax1.scatter(df_cv["num_features"].loc[idx], df_cv["cv"].loc[idx],
                color="black", s=50, alpha=0.55)
    ax1.set_ylabel('Coefficient of Variation')
    ax1.set_xlabel('Number of Principal Components')
    ax1.set_ylim((min(0, min(results.cv_performance)), round(np.nan_to_num(
        results.cv_performance).max(), 1)+0.1))
    # ax1.tick_params(axis='x', labelsize=7)

    # SUBPLOT: Boxplot of Cluster Testing Accuracies vs. #
    # of Principal Components
    ax2.tick_params(axis='x', labelsize=7)
    ax2.boxplot(results.cluster_performances, labels=inputs.chosen_features)
    ax2.scatter(chosen_features_repeated,
                results._cluster_performance_flattened,
                s=20,
                c="black",
                alpha=0.4,
                edgecolors="none",
                )
    ax2.axhline(results.mean_performance,
                c="red",
                alpha=0.8)
    if model_goal == "classification":
        ax2.set_ylabel('Testing Accuracy')
        ax2.set_ylim((min(0, min(results._cluster_performance_flattened)), 1))
    else:
        ax2.set_ylabel('Testing RMSE')
        ax2.set_ylim((min(0, min(results._cluster_performance_flattened)), 100))
    ax2.tick_params(axis='x', labelsize=7)
    ax2.set_xlabel('Number of Principal Components')

    # SUBPLOT: Percent Explained Variance vs. Number of Principal Components
    ax3.set_xlabel('Number of Principal Components')
    ax3.set_ylabel('Percent Explained Variance')
    ax3.tick_params(axis='x', labelsize=7)
    ax3.set_ylim((0, 1))
    explained_variance = pca_model.get_cum_variance()
    ax3.plot(explained_variance, marker="o", markersize=3, alpha=0.7)
    fig.align_ylabels()

    # FIGURE: Cluster Metrics
    fig2 = plt.figure()
    bx1 = fig2.add_subplot(221)
    bx2 = fig2.add_subplot(222)
    bx3 = fig2.add_subplot(223)
    bx4 = fig2.add_subplot(224)
    fig2.suptitle("PSP Plates | N=2928 (Fold 4)")

    # SUBPLOT: Silhouette Coefficient
    bx1.set_ylabel('Silhouette Coefficient')
    bx1.scatter(inputs.chosen_features, results.sil_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown")
    bx1.tick_params(axis='x', labelsize=7)

    # SUBPLOT: Calinski-Harabasz Index
    bx2.set_ylabel('Calinski-Harabasz Index')
    bx2.scatter(inputs.chosen_features, results.cal_har_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown")
    bx2.tick_params(axis='x', labelsize=7)

    # SUBPLOT: Davies-Bouldin Index
    bx3.set_xlabel('Number of Principal Components')
    bx3.set_ylabel('Davies-Bouldin Index')
    bx3.scatter(inputs.chosen_features, results.dav_bou_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown")
    bx3.tick_params(axis='x', labelsize=7)

    # SUBPLOT: Euclidean Distance Between Centroids
    bx4.set_xlabel('Number of Principal Components')
    bx4.set_ylabel('''Euclidean Distance
between Centroids''')
    bx4.set_ylim((0, 75))
    bx4.tick_params(axis='x', labelsize=7)
    bx4.scatter(inputs.chosen_features, results.centroid_distances, marker="o",
                s=12, alpha=0.7,
                c="tab:brown")

    plt.tight_layout()

    if inputs.save_bool == "Y":
        # Check if Folders Exist
        results_dir = absolute_dir+"results/graphs/"+dataset_used
        general_dir = results_dir+"/general"
        cluster_metrics_dir = results_dir+"/cluster_metrics"
        try:
            os.mkdir(results_dir)
            os.mkdir(general_dir)
            os.mkdir(cluster_metrics_dir)
        except BaseException:
            try:
                os.mkdir(general_dir)
                os.mkdir(cluster_metrics_dir)
            except BaseException:
                pass
        fig.savefig(general_dir+"/"+dataset_used+"_dataset_" +
                    str(dataset_num)+".png", bbox_inches='tight')
        fig2.savefig(cluster_metrics_dir+"/"+dataset_used+"_dataset_" +
                     str(dataset_num)+".png", bbox_inches='tight')

        # FIGURE: Automatic Elbow Method for KMeans Clustering
        if inputs.elbow_bool:
            try:
                os.mkdir(absolute_dir+"results/graphs/"+
                         dataset_used+"/optimal_k")
            except BaseException:
                pass
            plt.figure()
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Optimal Number of KMeans Clusters')
            plt.xticks(fontsize=7)
            plt.ylim((0, 10))
            plt.plot(inputs.chosen_features, results.optimal_ks,
                     "^--", markersize=5, alpha=0.7)
            plt.title(new_title, y=1.05)
            plt.savefig(absolute_dir+"results/graphs/"+dataset_used +
                        "/optimal_k/"+new_title.replace("||", "-")+".png",
                        bbox_inches='tight')


def get_pca_model(inputs: Inputs) -> PCA:
    """Return fitted PCA object based on <inputs>."""
    # PCA: Fit & Transform
    pca_model = MyPCA(inputs.chosen_features, inputs.random_seed)

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
         df_selection_methods: pd.DataFrame = None) -> Tuple[Results,
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

    pca_model = get_pca_model(inputs)

    # CLUSTER: PCA-Transformed Data [Iterate over number of features kept.]
    iterated_cluster_results = iterative_clustering(inputs,
                                                    pca_model,
                                                    n_iter=100)

    # Get and store results
    results = get_results(inputs, iterated_cluster_results)

    # PCA: Principal Components Selection Methods
    pc_selection = PrincCompSelection(inputs, pca_model)
    df_selection = pc_selection.select_pcs(idx=paths[dataset_num].replace(
        absolute_dir+data_dir, ""))
    df_selection["Minimum Mode CV"] = results.min_mode_cv
    df_selection_methods_new = pd.concat([df_selection_methods, df_selection])

    # Create Plots
    create_plots(inputs, pca_model, results, dataset_num)
    plt.show()

    # Save Results
    if inputs.save_bool == "Y":
        df_results = pd.DataFrame([inputs.chosen_features,
                                   results.cv_performance,
                                   results.cluster_performances,
                                   results.centroid_distances]).transpose()
        df_results = df_results.rename({0: "features_kept",
                                        1: "cv",
                                        2: "cluster_performances",
                                        3: "mean_centroid_distance",
                                        }, axis=1)
        df_results.to_csv(absolute_dir+"results/dataset/" +
                          dataset_used+"_dataset_" +
                          str(dataset_num)+".csv",
                          index=False)

    return results, df_selection_methods_new


# CLIENT CODE
if __name__ == "__main__":

# =============================================================================
#     for dataset_used in ["psp_plates", "boneage"]:
#         paths=[]
#         for root, dirs, files in os.walk(absolute_dir + data_dir + dataset_used, topdown=False):
#            for name in files:
#               paths.append(os.path.join(root, name))
# =============================================================================

    inputs = Inputs(paths)
    inputs.which_datasets = [11]
    # Iterate over Seeds
    # for random_seed in [1969, 1974, 2000, 2001]:
    inputs.random_seed = 1969

    df_selection_methods = pd.DataFrame()

    # Loop over the datasets (folds)
    for dataset_num in inputs.which_datasets:
        _, df_selection_methods = main(inputs,
                                       dataset_num,
                                       df_selection_methods)

    if inputs.save_bool == "Y":
        df_selection_methods.to_csv(absolute_dir+"results/pc_selection/" +
                                    dataset_used+f"-{inputs.random_seed}"+
                                    ".csv", index=False)

