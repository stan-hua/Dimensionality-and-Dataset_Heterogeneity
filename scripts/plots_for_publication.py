from main import *
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import mode

# PATHS
orig_data_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/" \
                "PCA-Clustering-Project/data/"
results_dir = "/Users/Stanley/Desktop/Tyrrell Lab/" \
              "ROP Project/PCA-Clustering-Project/results/"
data_dir = f"{results_dir}/dataset/"
features_dir = '/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/' \
               'PCA-Clustering-Project/data/'
final_dir = f"{results_dir}/graphs/presentation_graphs/"

# Plot Settings
sns.set_style("white")
plt.rc('font', family='serif')


# FIGURE 1: Intrinsic Cluster Metrics
# HELPER: Get cluster metrics
def get_cluster_metrics(path: str, model_goal: str) -> Tuple[Inputs, Results]:
    """Return Inputs and Results from running methodlogy.
    Cluster Analysis methodology is done on features in <path>."""
    # Get dataset
    inputs = Inputs([path], model_goal)
    inputs.random_seed = None
    inputs.get_df_split(0)
    # PCA
    pca_model = get_pca_model(inputs)
    # Iterate clustering over the PCs
    iterated_cluster_results = iterative_clustering(inputs,
                                                    pca_model,
                                                    n_iter=100)
    # Get results
    return inputs, get_results(inputs, iterated_cluster_results)


def plot_fig_1():
    """Plot Figure 1: the Coefficient of Variation vs. Number of Principal
    Components (PC)."""
    # PATHS
    small = "psp_plates/teeth_features_100_red_whole_fold_0.csv"
    big = "psp_plates/teeth_features_all_red_whole_fold_0.csv"
    model_goal = "classification"
    # PRE-PLOT: Get metrics
    inputs_small, results_small = get_cluster_metrics(features_dir+small,
                                                      model_goal)
    inputs_big, results_big = get_cluster_metrics(features_dir+big,
                                                  model_goal)

    # Create Figure
    fig_1 = plt.figure()
    fig_1.set_tight_layout(True)
    ax1 = fig_1.add_subplot(221)
    ax2 = fig_1.add_subplot(222)
    ax3 = fig_1.add_subplot(223)
    ax4 = fig_1.add_subplot(224)

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.tick_params(axis='x', labelsize=8)
    ax4.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)

    ax1.set_ylabel('Silhouette Coefficient')
    ax2.set_ylabel('Calinski-Harabasz Index')
    ax3.set_xlabel('Number of Principal Components')
    ax3.set_ylabel('Davies-Bouldin Index')
    ax4.set_xlabel('Number of Principal Components')
    ax4.set_ylabel('''Euclidean Distance
    between Centroids''')

    # SUBPLOT: Silhouette Coefficient
    ax1.scatter(inputs_small.chosen_features, results_small.sil_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown", label="N = 100")
    ax1.scatter(inputs_big.chosen_features, results_big.sil_score,
                marker="o", s=12, alpha=0.7,
                c="tab:blue", label="N = 2928")
    # SUBPLOT: Calinski-Harabasz Index
    ax2.scatter(inputs_small.chosen_features, results_small.cal_har_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown", label="N = 100")
    ax2.scatter(inputs_big.chosen_features, results_big.cal_har_score,
                marker="o", s=12, alpha=0.7,
                c="tab:blue", label="N = 2928")

    # SUBPLOT: Davies-Bouldin Index
    ax3.scatter(inputs_small.chosen_features, results_small.dav_bou_score,
                marker="o", s=12, alpha=0.7,
                c="tab:brown", label="N = 100")
    ax3.scatter(inputs_big.chosen_features, results_big.dav_bou_score,
                marker="o", s=12, alpha=0.7,
                c="tab:blue", label="N = 2928")

    # SUBPLOT: Euclidean Distance Between Centroids
    ax4.scatter(inputs_small.chosen_features, results_small.centroid_distances,
                marker="o", s=12, alpha=0.7,
                c="tab:brown", label="N = 100")
    ax4.scatter(inputs_big.chosen_features, results_big.centroid_distances,
                marker="o", s=12, alpha=0.7,
                c="tab:blue", label="N = 2928")

    # Legend
    legend_elements = [Line2D([0], [0], marker='o',
                              label='N = 100',
                              markerfacecolor='tab:brown',
                              markersize=9, ls=""),
                       Line2D([0], [0], marker='o',
                              label='N = 2928',
                              markerfacecolor='tab:blue',
                              markersize=9, ls="")]

    fig_1.legend(handles=legend_elements, frameon=True, shadow=True)

    fig_1.savefig(f"{final_dir}cluster_metrics vs. num_pcs.png",
                  dpi=1200)


# FIGURE 2: Instability of CV over Number of PCs
def plot_fig_2():
    """Plot Figure 2: the Coefficient of Variation vs. Number of Principal
    Components (PC)."""
    # Sample Size 300. Fold 1 of 4.
    df_small = pd.read_csv(f"{data_dir}psp_plates_dataset_4.csv")
    # Sample Size: 2928. Fold 1 of 4.
    df_big = pd.read_csv(f"{data_dir}psp_plates_dataset_12.csv")

    # PRE-PLOT: Get indices for mode CV (for small and big sample size)
    small_idx = df_small.cv == df_small.cv.mode()[0]
    big_idx = df_big.cv == df_big.cv.mode()[0]

    # Create Figure
    fig_1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig_1.set_tight_layout(True)
    plt.ylabel('Coefficient of Variation')
    plt.xlabel('Number of Principal Components')
    ax1.set_ylim((0, max(
        round(df_small.cv.max(), 1)+0.1,
        round(df_big.cv.max(), 1)+0.1)))
    ax2.set_ylim((0, max(
        round(df_small.cv.max(), 1)+0.1,
        round(df_big.cv.max(), 1)+0.1)))
    ax1.set_title("PSP Plates | N = 300")
    ax2.set_title("PSP Plates | N = 2928")

    # Scatter Plot
    ax1.scatter(df_small["features_kept"].loc[~small_idx],
                df_small["cv"].loc[~small_idx],
                color='black', alpha=.5, s=30)
    ax1.scatter(df_small["features_kept"].loc[small_idx],
                df_small["cv"].loc[small_idx],
                color="darkred", s=30, alpha=0.5, label="Mode")

    ax2.scatter(df_big["features_kept"].loc[~big_idx],
                df_big["cv"].loc[~big_idx],
                color='black', alpha=.5, s=30)
    ax2.scatter(df_big["features_kept"].loc[big_idx],
                df_big["cv"].loc[big_idx],
                color="darkred", s=30, alpha=0.5, label="Mode")
    ax1.legend()
    ax2.legend()

    fig_1.savefig(f"{final_dir}cv vs. num_pcs (PSP Plates).png",
                  dpi=1200)


# FIGURE 3: Cumulative Percent Variances Explained
# Helper Method: Get Cumulative Percent Variances
def _get_percent_variance(path: str, model_goal: str) -> pd.Series:
    """Return cumulative percent variance for all PCs at path."""
    inputs = Inputs([orig_data_dir + path], model_goal)
    inputs.random_seed = None
    inputs.get_df_split(0)
    pca_model = get_pca_model(inputs)
    return pca_model.get_cum_variance()


def plot_fig_3():
    """Plot Figure 3: Scree Plots for each dataset, as an example.
    """
    bone = "boneage/boneage_features_red_whole_all_fold_0.csv"
    psp = "psp_plates/teeth_features_all_red_whole_fold_0.csv"
    cifar = "cifar10/cifar10_features_12000_fold_0.csv"

    bone_pv = _get_percent_variance(bone, "regression")
    psp_pv = _get_percent_variance(psp, "classification")
    cifar_pv = _get_percent_variance(cifar, "classification")

    bone_pv.index += 1

    # Create Figure
    fig_1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                          constrained_layout=False)
    fig_1.set_tight_layout(True)
    # fig_1.suptitle("Scree Plots for Each Dataset")
    ax1.set_title("Bone Age")
    ax2.set_title("PSP Plates")
    ax3.set_title("Modified CIFAR10")
    ax2.set_ylabel('Cumulative Percent Variance Explained', labelpad=10)
    ax3.set_xlabel('Number of Principal Components', labelpad=10)
    ax3.set_xlim(0, 512)
    ax3.set_ylim(0, 1)
    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.scatter(bone_pv.index, bone_pv,
                color='teal', alpha=.8, s=4)
    ax2.scatter(bone_pv.index, cifar_pv,
                color='teal', alpha=.8, s=4)
    ax3.scatter(bone_pv.index, psp_pv,
                color='teal', alpha=.8, s=4)

    fig_1.savefig(f"{final_dir}scree_plots.png",
                  dpi=1200)


# SUP. FIGURE 1: Random Seed on CV
# HELPER FUNCTIONS
def get_iterated_cvs(path: str, model_goal: str,
            num_pcs: int, num_iter: int,
            weighted: bool = False, method: str = "kmeans") -> int:
    """For <path>, use <num_pcs> number of Principal Components in calculating
    Coefficient of Variation. Iterate this <num_iter> times.

    NOTE: Random Seed is randomly set, so running this function may return
    differing values of CV.
    """
    inputs = Inputs([path], model_goal)
    inputs.random_seed = None
    inputs.get_df_split(0)

    pca_model = get_pca_model(inputs)

    cv_accum = []
    for i in range(num_iter):
        cluster_model = Clustering(inputs.num_cluster, 300,
                                   inputs.random_seed, method=method)
        cluster_model.fit(pca_model.pcs_train.loc[:, :num_pcs-1])
        cv_within_iter = []
        for g in range(10):
            cluster_prediction = cluster_model.predict(
                pca_model.pcs_test.loc[:, :num_pcs-1])
            cluster_performances = cluster_model.get_cluster_performances(
                inputs.df_test.copy(),
                cluster_prediction,
                num_pcs,
                inputs.num_cluster,
                model_goal=model_goal)
            if weighted:
                weighted_stats = DescrStatsW(
                    cluster_performances,
                    weights=cluster_model.temp_cluster_sizes,
                    ddof=0)
                cv_within_iter.append(weighted_stats.std / weighted_stats.mean)
            else:
                cv_within_iter.append(np.std(cluster_performances) /
                                      np.mean(cluster_performances))
        cv_accum.append(mode(cv_within_iter, axis=None).mode[0])
    return cv_accum


def get_datasets_cvs(small: bool, weighted: bool = False) -> None:
    """Get and save Coefficient of Variation values iterated for each dataset.
    """
    # PATHS to data
    if not small:  # for largest sample size
        bone = "boneage/boneage_features_red_whole_all_fold_0.csv"
        psp = "psp_plates/teeth_features_all_red_whole_fold_0.csv"
        cifar = "cifar10/cifar10_features_12000_fold_0.csv"
        size = "greatest"
    else:
        bone = "boneage/boneage_features_red_whole_300_fold_0.csv"
        psp = "psp_plates/teeth_features_100_red_whole_fold_0.csv"
        cifar = "cifar10/cifar10_features_400_fold_0.csv"
        size = "smallest"

    # Get CV iterated with randomly chosen seed
    cvs_bone = get_iterated_cvs(orig_data_dir+bone, "regression",
                                num_pcs=9,
                                num_iter=100,
                                weighted=weighted)
    cvs_cifar = get_iterated_cvs(orig_data_dir+cifar, "classification",
                                 num_pcs=46,
                                 num_iter=100,
                                 weighted=weighted)
    cvs_psp = get_iterated_cvs(orig_data_dir+psp, "classification",
                               num_pcs=43,
                               num_iter=100,
                               weighted=weighted)

    df = pd.DataFrame()
    df["boneage"] = cvs_bone
    df["cifar10"] = cvs_cifar
    df["psp_plates"] = cvs_psp

    weighted_str = ""
    if weighted:
        weighted_str = ", weighted by size"

    df.to_csv(
        f"{results_dir}CVs/random_seeds ({size} size{weighted_str}).csv",
        index=False)


def plot_sup_fig_1(weighted: bool = False):
    """Plot Supplementary Figure 1: Bar Plots of CV vs. Random Seed.

    Minimum Mode CV (random seed = 1969) is used to determine number of
    principal components to select.

    In addition, the maximum sample size for each dataset is used, more
    specifically fold 1 of 4.
    """
    weight_str = ""
    if weighted:
        weight_str = ", weighted by size"
    df_small = pd.read_csv(
        f"{results_dir}CVs/random_seeds "
        f"(smallest size{weight_str}).csv")
    df_big = pd.read_csv(
        f"{results_dir}CVs/random_seeds "
        f"(greatest size{weight_str}).csv")
    # PRE: Prepare for creating plot legend
    legend_elements = [Line2D([0], [0], marker='o',
                              label='Smallest Dataset Size',
                              markerfacecolor='darkmagenta',
                              markersize=7, ls=""),
                       Line2D([0], [0], marker='o',
                              label='Greatest Dataset Size',
                              markerfacecolor='goldenrod',
                              markersize=7, ls="")]

    # Create Figure
    plt.figure()
    plt.title("Effect of Random Seed on CV")
    plt.ylabel("Coefficient of Variation", labelpad=3)
    plt.xlabel("Dataset Used", labelpad=3)
    plt.ylim((0, max(0.2, round(df_small.max().max(), 2)+0.05)))
    sns.stripplot(data=df_small,
                  order=["boneage", "cifar10", "psp_plates"],
                  color="darkmagenta", alpha=0.15)
    plt.boxplot(df_small, positions=[0, 1, 2],
                labels=["Bone Age", "Modified CIFAR10", "PSP Plates"])

    sns.stripplot(data=df_big,
                  order=["boneage", "cifar10", "psp_plates"],
                  color="goldenrod", alpha=0.15)
    plt.boxplot(df_big, positions=[0, 1, 2],
                labels=["Bone Age", "Modified CIFAR10", "PSP Plates"])

    plt.legend(handles=legend_elements, loc='best', shadow=True)
    plt.tight_layout()

    plt.savefig(f"{final_dir}cv vs. random_seed{weight_str}.png",
                dpi=1200)

    # Extra Violin Plot
    # sns.violinplot(data=df_small, scale="width", label="Small")
    # sns.violinplot(data=df_big, scale="width", label="Big")
    # plt.ylim([0, 0.3])
    # plt.ylabel("Coefficient of Variation")
    # plt.legend()


# SUP. FIGURE 2: PCs on CV
def plot_sup_fig_2():
    """Plot : Bar Plots of CV vs. Number of Principal Components.

    In addition, the maximum sample size for each dataset is used, more
    specifically fold 1 of 4.
    """
    big_paths = ["boneage_dataset_8.csv", "psp_plates_dataset_8.csv",
                 "cifar10_dataset_0.csv"]
    small_paths = ["boneage_dataset_0.csv", "psp_plates_dataset_0.csv",
                   "cifar10_dataset_8.csv"]
    df_big = pd.DataFrame()
    for path in big_paths:
        df = pd.read_csv(data_dir+path)
        df_big = pd.concat([df_big, df.cv], axis=1)
    df_small = pd.DataFrame()
    for path in small_paths:
        df = pd.read_csv(data_dir+path)
        df_small = pd.concat([df_small, df.cv], axis=1)

    df_big.columns = ["boneage", "cifar10", "psp_plates"]
    df_small.columns = ["boneage", "cifar10", "psp_plates"]

    # PRE: Prepare for creating plot legend
    legend_elements = [Line2D([0], [0], marker='o',
                              label='Smallest Dataset Size',
                              markerfacecolor='darkmagenta',
                              markersize=7, ls=""),
                       Line2D([0], [0], marker='o',
                              label='Greatest Dataset Size',
                              markerfacecolor='goldenrod',
                              markersize=7, ls="")]

    # Create Figure
    plt.figure()
    plt.title("Effect of Number of Principal Components on CV")
    plt.ylabel("Coefficient of Variation", labelpad=3)
    plt.xlabel("Dataset Used", labelpad=3)
    # plt.ylim((0, max(0.2, round(df_small.max().max(), 2)+0.05)))
    plt.ylim((0, 0.25))
    sns.stripplot(data=df_small,
                  order=["boneage", "cifar10", "psp_plates"],
                  color="darkmagenta", alpha=0.15)
    plt.boxplot(df_small, positions=[0, 1, 2],
                labels=["Bone Age", "Modified CIFAR10", "PSP Plates"])

    sns.stripplot(data=df_big,
                  order=["boneage", "cifar10", "psp_plates"],
                  color="goldenrod", alpha=0.15)
    plt.boxplot(df_big, positions=[0, 1, 2],
                labels=["Bone Age", "Modified CIFAR10", "PSP Plates"])

    plt.legend(handles=legend_elements, loc='best', shadow=True)
    plt.tight_layout()

#%% Get CV (mode from 10 iterations) for each number of PCs
def get_cv_against_pc(path: str, model_goal: str,
            weighted: bool = False, method: str = "kmeans") -> int:
    """For <path>, use <num_pcs> number of Principal Components in calculating
    Coefficient of Variation. Iterate this <num_iter> times.

    NOTE: Random Seed is randomly set, so running this function may return
    differing values of CV.
    """
    # Get train/test set
    inputs = Inputs([path], model_goal)
    inputs.random_seed = None
    inputs.get_df_split(0)
    
    # PCA
    pca_model = get_pca_model(inputs)
    max_pcs = pca_model.get_max_pc()
    
    cv_accum = []
    for num_pcs in range(1, max_pcs):
        cluster_model = Clustering(inputs.num_cluster, 300,
                                   inputs.random_seed, method=method)
        cluster_model.fit(pca_model.pcs_train.loc[:, :num_pcs-1])
        cv_within_iter = []
        for g in range(10):  # clustering iterated for one instance
            cluster_prediction = cluster_model.predict(
                pca_model.pcs_test.loc[:, :num_pcs-1])
            cluster_performances = cluster_model.get_cluster_performances(
                inputs.df_test.copy(),
                cluster_prediction,
                num_pcs,
                inputs.num_cluster,
                model_goal=model_goal)
            if weighted:
                weighted_stats = DescrStatsW(
                    cluster_performances,
                    weights=cluster_model.temp_cluster_sizes,
                    ddof=0)
                cv_within_iter.append(weighted_stats.std / weighted_stats.mean)
            else:
                cv_within_iter.append(np.std(cluster_performances) /
                                      np.mean(cluster_performances))
        cv_accum.append(mode(cv_within_iter, axis=None).mode[0])
    return cv_accum


if __name__ == "__main__":
    small = False
    if not small:  # for largest sample size
        bone = "boneage/boneage_features_red_whole_all_fold_0.csv"
        psp = "psp_plates/teeth_features_all_red_whole_fold_0.csv"
        cifar = "cifar10/cifar10_features_12000_fold_0.csv"
        size = "greatest"
    else:
        bone = "boneage/boneage_features_red_whole_300_fold_0.csv"
        psp = "psp_plates/teeth_features_100_red_whole_fold_0.csv"
        cifar = "cifar10/cifar10_features_400_fold_0.csv"
        size = "smallest"

    # Get CV iterated with randomly chosen seed
    cv_performance = get_cv_against_pc(orig_data_dir+bone, "regression")
    # cv_performance = get_cv_against_pc(orig_data_dir+cifar, "classification")
    # cv_performance = get_cv_against_pc(orig_data_dir+psp, "classification")


    new_title = bone.replace(".csv", "").replace("/", " || ")

    # FIGURE: General plots [CV, Test Accuracy/RMSE, % Variance Explained]
    plt.title(new_title)

    # SUBPLOT: CV Accuracy vs. # of Principal Components
    df_cv = pd.DataFrame({"num_features": list(range(len(cv_performance))),
                          "cv": cv_performance})
    idx = (df_cv.cv == mode(cv_performance).mode[0])

    plt.xlabel('Number of Principal Components')
    plt.ylim((min(0, min(cv_performance)), round(np.nan_to_num(
        cv_performance).max(), 1) + 0.1))
    plt.ylabel('Coefficient of Variation')
    plt.scatter(df_cv["num_features"], df_cv["cv"],
                color='black', alpha=.5, s=30)
    plt.scatter(df_cv["num_features"].loc[idx], df_cv["cv"].loc[idx],
                color="darkred", s=30, alpha=0.5, label="Mode")
    plt.legend()
