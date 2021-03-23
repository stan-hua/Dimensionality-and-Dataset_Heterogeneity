import skfuzzy as fuzz
from pca import *


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
                                 num_cluster: int) -> np.array:
        """Returns list of cluster accuracies.
        """
        global model_goal
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
            except Exception():
                fix_cluster_performances.append(-1)
        fix_cluster_performances = np.array(fix_cluster_performances)

        # TODO: Adjust Cluster Performances by Sample Size
        cluster_adjusted_size = cluster_freq * fix_cluster_performances

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
            Probability of Hard Clustering
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
        """Fit Fuzzy C-Means on Training Data"""
        center, _, _, _, _, num_itered, fpc = fuzz.cluster.cmeans(
            train_data.transpose(), self._num_clusters, 2, error=0.005,
            maxiter=self._n_iter, init=None)
        self.cluster_centers_ = center

    def predict(self, test_data: pd.DataFrame) -> np.array:
        u, u0, d, jm, p, fpc = fuzz.cmeans_predict(test_data.transpose(),
                                                   self.cluster_centers_, 2,
                                                   error=0.005,
                                                   maxiter=self._n_iter)
        cluster_pred = cluster_model.predict()
        # Hard Clustering
        self.labels_ = np.argmax(cluster_pred, axis=0)
        # Probabilities
        self.probs_ = np.max(cluster_pred, axis=0)

        return self.labels_


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
    inputs = Inputs(paths)
    inputs.which_datasets = [13]
    # Iterate over Seeds
    # for random_seed in [1969, 1974, 2000, 2001]:
    inputs.random_seed = 1969

    df_selection_methods = pd.DataFrame()

    # Loop over the datasets (folds)
    # for dataset_num in inputs.which_datasets:
    #     _, df_selection_methods = main(inputs,
    #                                    dataset_num,
    #                                    df_selection_methods)
    #
    # if inputs.save_bool == "Y":
    #     df_selection_methods.to_csv(absolute_dir+"results/pc_selection/" +
    #                                 dataset_used+f"-{inputs.random_seed}"+
    #                                 ".csv", index=False)


    if df_selection_methods is None:
        df_selection_methods = pd.DataFrame()

    inputs.get_df_split(1)
    pca_model = get_pca_model(inputs)
    num_kept = 5
    cluster_train = pca_model.pcs_train.loc[:, :num_kept-1]
    cluster_val = pca_model.pcs_test.loc[:, :num_kept-1]
    cluster_model = FuzzyCMeans(cluster_train,
                                cluster_val,
                                inputs.num_cluster,
                                1000,
                                inputs.random_seed)
