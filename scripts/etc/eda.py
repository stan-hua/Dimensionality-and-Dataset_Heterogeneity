from __future__ import annotations
from typing import Optional, Tuple, Union

import os
import random

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

sns.set_style("white")
plt.rc('font', family='serif')
plt.ioff()
rcParams.update({'figure.autolayout': True})

# File Paths
absolute_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-" \
               "Project/"

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
        which_datasets = 0

        if which_datasets == 0:
            self.which_datasets = list(range(len(paths)))
        else:
            self.which_datasets = random.sample(range(len(paths)), 1)

        # INPUT: Get Elbow Plot?
        # elbow_bool = bool(int(input("Include Elbow Plot **1: Yes, 0: No\n")))
        self.elbow_bool = bool(0)

        # INPUT: Use Test Set Only? (So only test set used)
        self.exclude_train = 0

        # INPUT: Choose Features
        self.chosen_features = list(range(1, 71))

        # INPUT: Save Results
        self.save_bool = "N"

        # INPUT: Random Seed
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

    # t-SNE Plots
    for i in range(len(paths)):
        df_train, df_test = inputs.get_df_split(i)
        train_tsne = TSNE().fit_transform(df_train)
        train_tsne = pd.DataFrame(train_tsne)

        test_tsne = TSNE().fit_transform(df_test)
        test_tsne = pd.DataFrame(test_tsne)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"{dataset_used} | Dataset {i}")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax1.set_title("Training Set")
        ax2.set_title("Testing Set")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax1.plot(train_tsne[0], train_tsne[1], "o")
        ax2.plot(test_tsne[0], test_tsne[1], "o")
        try:
            fig.savefig(f"{absolute_dir}results/graphs/{dataset_used}/eda"
                        f"/tsne/{i}.png")
        except:
            os.mkdir(f"{absolute_dir}results/graphs/{dataset_used}/eda/tsne")
            fig.savefig(f"{absolute_dir}results/graphs/{dataset_used}/eda"
                        f"/tsne/{i}.png")

