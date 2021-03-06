from svcca import get_sample_sizes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
import seaborn as sns
import os
sns.set_style("white")

dataset_used = ["psp_plates", "boneage", "cifar10"]
choice = 2
num_sample_sizes = 5

absolute_dir = "/Users/Stanley/Desktop/Tyrrell Lab/" \
               "ROP Project/PCA-Clustering-Project/results/"
paths = []
for root, dirs, files in os.walk(absolute_dir+"pc_selection", topdown=False):
    for name in files:
        paths.append(os.path.join(root, name))

paths_series = pd.Series(paths)
idx = paths_series.str.contains(dataset_used[choice])

paths_series = paths_series[idx].reset_index(drop=True)
# PREPROCESSING:
    # 1) Add training sample size
    # 2) Sort by sample size
# Get dataframes for pcs selected
dfs = []
for path in paths_series:
    dfs.append(pd.read_csv(path))

training_sample_sizes = get_sample_sizes(end=20)[0]

for df in dfs:
    df["training_size"] = training_sample_sizes
    df.sort_values(by=["training_size"], inplace=True, ignore_index=True)
training_sample_sizes.sort()


# Divide by Max Number of PCs
def divide_by_max_pcs(x):
    return x/min(x["training_size"], 512)
# dfs[0] = dfs[0].apply(divide_by_max_pcs, axis=1)


# PLOT Selection Methods
# Prep.
x_ticks = np.array(list(range(1, num_sample_sizes+1))*4).reshape(
    num_sample_sizes, 4).transpose().flatten()
x_tick_loc = list(range(1, num_sample_sizes+1))


def plot_selection_methods():
    plt.rcParams["figure.figsize"] = (20, 20)
    """Create Plots for selection methods for number of principal components to
    keep."""
    legend_elements = [Line2D([0], [0], marker='o', label='CPV > 0.8',
                              markerfacecolor='r', markersize=7, ls=""),
                       Line2D([0], [0], marker='o', label='CPV > 0.99',
                              markerfacecolor='g', markersize=7, ls=""),
                       Line2D([0], [0], marker='^', label='Perc. Var. > 0.1',
                              markerfacecolor='purple', markersize=7, ls=""),
                       Line2D([0], [0], marker='s', label='Eig. 1',
                              markerfacecolor='b', markersize=7, ls=""),
                       Line2D([0], [0], marker='s', label='Eig. Avg.',
                              markerfacecolor='y', markersize=7, ls=""),
                       Line2D([0], [0], marker='^', label='Minimum Mode CV',
                              markerfacecolor='orange', markersize=7, ls="")
                       ]
    x_tick_labels = list(set(training_sample_sizes))
    x_tick_labels.sort()

    # 1) Plot Percent Variance -based Methods oax1 Selection
    fig, ax1 = plt.subplots()
    ax1.scatter(x=x_ticks, y=dfs[0]["Cum. Perc. Var. (0.8)"],
                c="r",
                alpha=0.5)

    ax1.scatter(x_ticks, dfs[0]["Cum. Perc. Var. (0.99)"],
                c="g",
                alpha=0.5)

    ax1.scatter(x_ticks, dfs[0]["Perc. Var. (0.1)"],
                c="purple",
                marker="^",
                alpha=0.5)

    ax1.set_title(dataset_used[choice])
    ax1.set_xlabel("Training Sample Size")
    ax1.set_ylabel("Number of PCs Suggested")
    ax1.legend(handles=legend_elements[:3], loc='best',
               bbox_to_anchor=(1.05, 1), frameon=True)
    ax1.set_xticks(x_tick_loc)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_ylim([0, 450])
    plt.tight_layout()

    # 2) Plot Eigenvalue-based Methods of Selection
    fig, ax2 = plt.subplots()

    ax2.scatter(x_ticks, dfs[0]["Eig. 1"],
                     c="b",
                     marker="s",
                     alpha=0.5)

    ax2.scatter(x_ticks, dfs[0]["Eig. Avg."],
                     c="y",
                     marker="s",
                     alpha=0.5)

    ax2.set_title(dataset_used[choice])
    ax2.set_xlabel("Training Sample Size")
    ax2.set_ylabel("Number of PCs Suggested")
    ax2.legend(handles=legend_elements[3:5], loc='best', frameon=True)
    ax2.set_xticks(x_tick_loc)
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_ylim([0, 450])

    # 3) Plot CV -based Methods of Selection
    fig, ax3 = plt.subplots()

    ax3.scatter(x_ticks, dfs[0]["Minimum Mode CV"],
                c="orange",
                marker="^",
                alpha=0.5)

    ax3.set_title(dataset_used[choice])
    ax3.set_xlabel("Training Sample Size")
    ax3.set_ylabel("Number of PCs Suggested")
    ax3.legend(handles=legend_elements[5:], loc='best', frameon=True)
    ax3.set_xticks(x_tick_loc)
    ax3.set_xticklabels(x_tick_labels)

    ax3.set_ylim([0, 450])


# PLOT CV vs. Random Seed
def plot_cv_random_seed():
    """Create plots for CV vs. Random Seed for each sample size."""
    df_cv_random = pd.DataFrame()
    random_seeds = [1969, 1974, 2000, 2001]
    df_cv_random["training_size"] = dfs[0]["training_size"]
    for i in range(len(dfs)):
        df_cv_random[str(random_seeds[i])] = dfs[i]["Minimum Mode CV"]

    for size in np.unique(training_sample_sizes):
        idx = df_cv_random.training_size == size
        plt.figure()
        for seed_idx in range(len(random_seeds)):
            plt.scatter([seed_idx]*4,
                        df_cv_random.loc[idx][str(random_seeds[seed_idx])],
                        alpha=0.3, c="midnightblue")

        plt.xticks(list(range(4)), labels=random_seeds)
        plt.ylabel("Number of PCs Suggested")
        plt.xlabel("Random Seed")
        plt.title(f"{dataset_used[choice]} | CV-based selection for {size} " + \
                  "training samples")
        plt.ylim([0, 450])
        plt.show()


if __name__ == "__main__":
    pass
    # paths = []
    # for root, dirs, files in os.walk(absolute_dir+"dataset", topdown=False):
    #     for name in files:
    #         paths.append(os.path.join(root, name))
    #
    # paths_series = pd.Series(paths)
    # idx = paths_series.str.contains(dataset_used[choice])
    # paths_series = paths_series[idx].reset_index(drop=True)
    # a = pd.read_csv(paths_series[0])



