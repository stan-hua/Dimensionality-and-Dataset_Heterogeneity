from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score
from yellowbrick.cluster.elbow import kelbow_visualizer
from scipy.spatial.distance import cdist
import skfuzzy as fuzz

import pandas as pd
import numpy as np


# CLASS Clustering
class Clustering:
    """Clustering Class. Wrapper for K-Means and Fuzzy C-Means.

    ==Attributes==:
        _train_data:
            Training set data
        _test_data:
            Test set data
        model:
            Either of class K-Means or Fuzzy C-Means. Used to cluster training
            and test set
        _num_clusters:
            Chosen number of clusters
        _n_iter:
            Maximum number of iterations
        _random_state:
            Random seed
        _soft_cluster:
            Decision to soft cluster first (using Fuzzy C-Means)
    """
    _train_data: pd.DataFrame
    _test_data: pd.DataFrame
    _num_clusters: int
    _n_iter: int
    _random_state: int
    _soft_cluster: bool

    temp_cluster_sizes: np.array

    def __init__(self,
                 num_clusters: int,
                 n_iter: int,
                 random_state_: int = None,
                 soft_cluster: bool = False):
        self._train_data = None
        self._test_data = None
        self._num_clusters = num_clusters
        self._n_iter = n_iter
        self._random_state = random_state_
        self._soft_cluster = soft_cluster

        # Initialize KMeans Model
        if not soft_cluster:
            self.model = KMeans(n_clusters=num_clusters,
                                random_state=random_state_,
                                n_init=n_iter)
        else:
            self.model = FuzzyCMeans(n_clusters=num_clusters,
                                     random_state=random_state_,
                                     n_iner=n_iter)

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit on training data"""
        self._train_data = train_data
        self.model.fit(train_data)

    def predict(self, test_data: pd.DataFrame) -> np.array:
        """Return cluster predictions."""
        self._test_data = test_data
        return self.model.predict(test_data)

    def elbow_plot(self,
                   start_num_clusters: int = 2,
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
        """Return Silhouette, Calinski-Harabasz and Davies Bouldin score."""
        sil_score = silhouette_score(self._train_data,
                                     self.model.labels_,
                                     metric='euclidean')

        cal_har_score = calinski_harabasz_score(self._train_data,
                                                self.model.labels_)

        dav_bou_score = davies_bouldin_score(self._train_data,
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
                                 num_cluster: int,
                                 model_goal: str) -> np.array:
        """Returns list of cluster accuracies.
        """
        # Getting cluster counts
        cluster_freq = np.unique(cluster_prediction, return_counts=True)

        # Get cluster test accuracies
        df["cluster"] = cluster_prediction
        if model_goal == "regression":
            df["prediction_accuracy"] = df.apply(
                lambda x: np.sqrt(((x.predictions - x.labels) ** 2)), axis=1)
        else:
            df["prediction_accuracy"] = (df.predictions == df.labels)

        # TODO: Adjust Cluster Performance by cluster probabilities
        if self._soft_cluster:
            df["cluster_probs"] = self.model.probs_
            df_cluster_performances = df.groupby(by=["cluster"]).apply(
                lambda x: np.average(
                    x.prediction_accuracy, weights=x.cluster_probs))
        else:
            df_cluster_performances = df.groupby(
                by=["cluster"]).mean()["prediction_accuracy"]

        if (df_cluster_performances == 0).sum() > 0:
            idx = np.where(df_cluster_performances == 0)[0]
            # if multiple clusters with 0 accuracy
            for idx_ in idx:
                print("At "+str(num_kept) + " PCs, cluster " +
                      str(idx_) + " with " +
                      str(cluster_freq[1][idx]) +
                      " values have 0 accuracy.")

            for cluster_idx in np.where(df_cluster_performances == 0)[0]:
                print("Cluster "+str(cluster_idx) +
                      " has images with 0 accuracy at indices: " +
                      str(np.where(cluster_prediction == cluster_idx)[0]))

        # Trying to prevent NA in CV and Test Accuracy
        fix_cluster_performances = []
        for cluster in range(num_cluster):
            try:
                fix_cluster_performances.append(
                    df_cluster_performances[cluster])
            except:
                fix_cluster_performances.append(-1)
        fix_cluster_performances = np.array(fix_cluster_performances) # without anything

        # TODO: Adjust Cluster Performances by Sample Size
        self.temp_cluster_sizes = cluster_freq[1]

        return fix_cluster_performances


# CLASS Fuzzy C-Means
class FuzzyCMeans:
    """Fuzzy C-Means Class.

    ==Attributes==:
        _num_clusters:
            Chosen number of clusters
        _n_iter:
            Maximum number of iterations
        _random_state:
            Random seed
        cluster_centers_:
            Location of cluster centers
        probs_:
            Probability of being in hard cluster (highest prob.)
        labels_:
            Hard cluster predictions
    """
    _num_clusters: int
    _n_iter: int
    _random_state: int
    cluster_centers_: np.array
    probs_: np.array
    labels_: np.array

    def __init__(self,
                 num_clusters: int = 4,
                 n_iter: int = 1000,
                 random_state_: int = None) -> None:
        """Initialize Fuzzy C-Means Class."""
        self._num_clusters = num_clusters
        self._n_iter = n_iter
        self._random_state = random_state_
        self.cluster_centers_ = None
        self.probs_ = None
        self.labels_ = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit Fuzzy C-Means on <train_data>."""
        center, _, _, _, _, num_itered, fpc = fuzz.cluster.cmeans(
            train_data.transpose(), self._num_clusters, 2, error=0.005,
            maxiter=self._n_iter, init=None)
        self.cluster_centers_ = center

    def predict(self, test_data: pd.DataFrame) -> np.array:
        """Return predicted cluster assignment."""
        u, u0, d, jm, p, fpc = fuzz.cmeans_predict(test_data.transpose(),
                                                   self.cluster_centers_, 2,
                                                   error=0.005,
                                                   maxiter=self._n_iter)
        # Hard Clustering
        self.labels_ = np.argmax(u, axis=0)
        # Probabilities
        self.probs_ = np.max(u, axis=0)

        return self.labels_


