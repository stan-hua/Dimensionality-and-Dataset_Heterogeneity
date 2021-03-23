from svcca import get_sample_sizes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
import seaborn as sns
import os
sns.set_style("white")

absolute_dir = "/Users/Stanley/Desktop/Tyrrell Lab/" \
               "ROP Project/PCA-Clustering-Project/results/"


def _helper_get_sample_sizes(dataset_used: str) -> list:
    """Return list of total sample sizes."""
    # Get Paths for getting sample size
    paths = []
    for root, dirs, files in os.walk(absolute_dir.replace("results", "data") +
                                     dataset_used, topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))

    # Get Sample Sizes
    if dataset_used == "cifar10":
        sample_sizes = get_sample_sizes(paths, dataset_used, end=20)
    else:
        sample_sizes = get_sample_sizes(paths, dataset_used)

    return np.array(sample_sizes).sum(axis=0)


def get_pc_selection_dfs(dataset_used: str) -> list:
    """Return list of DataFrames for Selection Methods for Number of Principal
    Components.
    """
    # Get Paths to Selection Dfs
    paths = []
    for root, dirs, files in os.walk(absolute_dir+"pc_selection",
                                     topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))
    paths_series = pd.Series(paths)
    idx = paths_series.str.contains(dataset_used)
    paths_series = paths_series[idx].reset_index(drop=True)

    # Get dataframes for pcs selected
    dfs = []
    for path in paths_series:
        dfs.append(pd.read_csv(path))

    return dfs


# Divide by Max Number of PCs
def divide_by_max_pcs(x):
    return x/min(x["sample_size"], 512)


# PLOT Selection Methods
def plot_selection_methods(dataset_used: str,
                           dfs: list, sample_sizes: list) -> None:
    plt.rcParams["figure.figsize"] = (8, 6)
    """Create Plots for selection methods for number of principal components to
    keep."""
    legend_elements = [Line2D([0], [0], marker='o', label='Cumulative Percent Variance > 0.99',
                              markerfacecolor='g', markersize=7, ls=""),
                       Line2D([0], [0], marker='o', label='Cumulative Percent Variance > 0.8',
                              markerfacecolor='r', markersize=7, ls=""),
                       Line2D([0], [0], marker='^', label='Perc. Var. > 0.1',
                              markerfacecolor='purple', markersize=7, ls=""),
                       Line2D([0], [0], marker='s', label='Eig. 1',
                              markerfacecolor='b', markersize=7, ls=""),
                       Line2D([0], [0], marker='s', label='Eig. Avg.',
                              markerfacecolor='y', markersize=7, ls=""),
                       Line2D([0], [0], marker='^', label='Minimum Mode CV',
                              markerfacecolor='orange', markersize=7, ls="")
                       ]
    x_tick_labels = list(set(sample_sizes))
    x_tick_labels.sort()

    # 1) Plot Percent Variance -based Methods oax1 Selection
    fig, ax1 = plt.subplots()
    ax1.scatter(x=x_ticks, y=dfs[0]["Cum. Perc. Var. (0.8)"],
                c="r",
                alpha=0.5)

    ax1.scatter(x_ticks, dfs[0]["Cum. Perc. Var. (0.99)"],
                c="g",
                alpha=0.5)

    # ax1.scatter(x_ticks, dfs[0]["Perc. Var. (0.1)"],
    #             c="purple",
    #             marker="^",
    #             alpha=0.5)

    ax1.set_title(dataset_used)
    ax1.set_xlabel("Training Sample Size")
    ax1.set_ylabel("Number of PCs Suggested")
    ax1.legend(handles=legend_elements[:3], loc='best',   # legend_elements[:3] including Perc. Var. (0.1)
               bbox_to_anchor=(1.05, 1), frameon=True)
    ax1.set_xticks(x_tick_loc)
    ax1.set_xticklabels(x_tick_labels)
    # ax1.set_ylim([0, 450])
    # ax1.set_ylim([0, 1])
    plt.tight_layout()

    # 2) Plot Eigenvalue-based Methods of Selection
    # fig, ax2 = plt.subplots()
    #
    # ax2.scatter(x_ticks, dfs[0]["Eig. 1"],
    #                  c="b",
    #                  marker="s",
    #                  alpha=0.5)
    #
    # ax2.scatter(x_ticks, dfs[0]["Eig. Avg."],
    #                  c="y",
    #                  marker="s",
    #                  alpha=0.5)
    #
    # ax2.set_title(dataset_used)
    # ax2.set_xlabel("Training Sample Size")
    # ax2.set_ylabel("Number of PCs Suggested")
    # ax2.legend(handles=legend_elements[3:5], loc='best', frameon=True)
    # ax2.set_xticks(x_tick_loc)
    # ax2.set_xticklabels(x_tick_labels)
    # # ax2.set_ylim([0, 450])
    # ax2.set_ylim([0, 1])

    # 3) Plot CV -based Methods of Selection
    # fig, ax3 = plt.subplots()

    ax1.scatter(x_ticks, dfs[0]["Minimum Mode CV"],
                c="orange",
                marker="^",
                alpha=0.5)

    ax1.set_title(dataset_used)
    ax1.set_xlabel("Training Sample Size")
    ax1.set_ylabel("Number of PCs Suggested")
    ax1.legend(handles=legend_elements[5:], loc='best', frameon=True)
    ax1.set_xticks(x_tick_loc)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_ylim([0, 450])
    # ax1.set_ylim([0, 1])


# PLOT CV vs. Random Seed
def plot_cv_random_seed(dataset_used: str,
                        dfs: list, sample_sizes: list) -> None:
    """Create plots for CV vs. Random Seed for each sample size."""
    df_cv_random = pd.DataFrame()
    random_seeds = [1969, 1974, 2000, 2001]
    df_cv_random["sample_size"] = dfs[0]["sample_size"]
    for i in range(len(dfs)):
        df_cv_random[str(random_seeds[i])] = dfs[i]["Minimum Mode CV"]

    for size in np.unique(sample_sizes):
        idx = df_cv_random.sample_size == size
        plt.figure()
        for seed_idx in range(len(random_seeds)):
            plt.scatter([seed_idx]*4,
                        df_cv_random.loc[idx][str(random_seeds[seed_idx])],
                        alpha=0.3, c="midnightblue")

        plt.xticks(list(range(4)), labels=random_seeds)
        plt.ylabel("Number of PCs Suggested")
        plt.xlabel("Random Seed")
        plt.title(f"{dataset_used} | CV-based selection for {size} " + \
                  "training samples")
        # plt.ylim([0, 450])
        plt.ylim([0, 1])
        plt.show()


if __name__ == "__main__":
    # Get Dataset Choice
    dataset_choice = int(
        input("DATASET: ** 1: boneage, 2: psp_plates, 3: cifar\n"))
    if dataset_choice == 1:
        dataset_used = "boneage"
        num_sample_sizes = 4
    elif dataset_choice == 3:
        dataset_used = "cifar10"
        num_sample_sizes = 5
    else:
        dataset_used = "psp_plates"
        num_sample_sizes = 4

    # Concatenate PC Selection + Sample Sizes
    dfs = get_pc_selection_dfs(dataset_used)
    sample_sizes = _helper_get_sample_sizes(dataset_used)
    for df in dfs:
        df["sample_size"] = sample_sizes
        df.sort_values(by=["sample_size"], inplace=True, ignore_index=True)
    sample_sizes.sort()

    # Divide by max_pcs
    # dfs[0] = dfs[0].apply(divide_by_max_pcs, axis=1)

    # Plot Prep.
    x_ticks = np.array(list(range(1, num_sample_sizes+1))*4).reshape(
        num_sample_sizes, 4).transpose().flatten()
    x_tick_loc = list(range(1, num_sample_sizes+1))

    plot_selection_methods(dataset_used, dfs, sample_sizes)




