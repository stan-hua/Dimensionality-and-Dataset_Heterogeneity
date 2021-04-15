from typing import List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
import seaborn as sns
import os

sns.set_style("white")
plt.rc('font', family='serif')

absolute_dir = "/Users/Stanley/Desktop/Tyrrell Lab/" \
               "ROP Project/PCA-Clustering-Project/results/"


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
def plot_selection_methods(dfs: List[List[pd.DataFrame]]) -> None:
    """Create Plots for selection methods for number of principal components to
    keep."""

    # Plot Prep.
    x_tick_labels = list(set(dfs[0][0].sample_size))
    x_tick_labels.sort()
    x_ticks = np.array(list(range(1, len(x_tick_labels)+1))*4).reshape(
        len(x_tick_labels), 4).transpose().flatten()
    x_tick_loc = list(range(1, len(x_tick_labels)+1))

    # 1) Plot Percent Variance -based Methods of Selection
    fig_1 = plt.figure()
    fig_1.subplots_adjust(wspace=0)
    fig_1.set_tight_layout(True)
    ax1 = fig_1.add_subplot(131)
    ax2 = fig_1.add_subplot(132, sharey=ax1)
    ax3 = fig_1.add_subplot(133, sharey=ax1)

    ax1.set_title("Bone Age")
    ax1.set_ylabel("Suggested Number of PCs")
    ax1.set_ylim([0, 450])
    ax1.set_xticks(x_tick_loc)
    ax1.set_xticklabels(x_tick_labels, rotation=315)

    ax1.scatter(x=x_ticks, y=dfs[0][0]["Cum. Perc. Var. (0.8)"],
                c="r", s=15, alpha=0.5, label="CPV >= 0.8")

    ax1.scatter(x_ticks, dfs[0][0]["Cum. Perc. Var. (0.99)"],
                c="g", s=15, alpha=0.5, label="CPV >= 0.99")

    ax1.scatter(x_ticks, dfs[0][0]["Minimum Mode CV"],
                c="orange", s=15, marker="^", alpha=0.5, label="Min. Mode")

    ax2.set_title("PSP Plates")
    ax2.set_xlabel("Sample Size", labelpad=10)
    ax2.set_xticks(x_tick_loc)
    ax2.set_xticklabels(x_tick_labels, rotation=315)

    ax2.scatter(x=x_ticks, y=dfs[1][0]["Cum. Perc. Var. (0.8)"],
                c="r", s=15, alpha=0.5, label="CPV >= 0.8")

    ax2.scatter(x_ticks, dfs[1][0]["Cum. Perc. Var. (0.99)"],
                c="g", s=15, alpha=0.5, label="CPV >= 0.99")

    ax2.scatter(x_ticks, dfs[1][0]["Minimum Mode CV"],
                c="orange", s=15, marker="^", alpha=0.5, label="Min. Mode")

    # Plot Prep.
    x_tick_labels = list(set(dfs[2][0].sample_size))
    x_tick_labels.sort()
    x_ticks = []
    for i in range(len(x_tick_labels)):
        x_ticks += 4 * [i+1]
    x_tick_loc = list(range(1, len(x_tick_labels)+1))

    ax3.set_title("CIFAR 10")
    ax3.set_xticks(x_tick_loc)
    ax3.set_xticklabels(x_tick_labels, rotation=315)

    ax3.scatter(x=x_ticks, y=dfs[2][0]["Cum. Perc. Var. (0.8)"],
                c="r", s=15, alpha=0.5, label="CPV >= 0.8")

    ax3.scatter(x_ticks, dfs[2][0]["Cum. Perc. Var. (0.99)"],
                c="g", s=15, alpha=0.5, label="CPV >= 0.99")

    ax3.scatter(x_ticks, dfs[2][0]["Minimum Mode CV"],
                c="orange", s=15, marker="^", alpha=0.5, label="Min. Mode")

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax1.legend(shadow=True, loc="upper left")


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
    # dataset_choice = int(
    #     input("DATASET: ** 1: boneage, 2: psp_plates, 3: cifar\n"))

    all_dfs = []

    for dataset_choice in [1, 2, 3]:
        if dataset_choice == 1:
            dataset_used = "boneage"
            sample_sizes = [300] * 4
            sample_sizes += [700] * 4
            sample_sizes += [5674] * 4
            sample_sizes += [2837] * 4
        elif dataset_choice == 3:
            dataset_used = "cifar10"
            sample_sizes = [12000] * 4
            sample_sizes += [2000] * 4
            sample_sizes += [400] * 4
            sample_sizes += [8000] * 4
            sample_sizes += [800] * 4
        else:
            dataset_used = "psp_plates"
            sample_sizes = [100] * 4
            sample_sizes += [300] * 4
            sample_sizes += [2928] * 4
            sample_sizes += [1464] * 4

        # Concatenate PC Selection + Sample Sizes
        dfs = get_pc_selection_dfs(dataset_used)

        for df in dfs:
            df["sample_size"] = sample_sizes
            df.sort_values(by=["sample_size"], inplace=True, ignore_index=True)

        all_dfs.append(dfs)
        # Divide by max_pcs
        # dfs[0] = dfs[0].apply(divide_by_max_pcs, axis=1)

    plot_selection_methods(all_dfs)




