from pca import *
from random_tests import *

from sklearn.model_selection import train_test_split

#%% Update File Paths
if dataset_used == "boneage":
    dataset_used = "boneage_new_1"
elif dataset_used == "psp_plates":
    dataset_used = "psp_plates_new_1"

paths=[]
for root, dirs, files in os.walk(absolute_dir + data_dir + dataset_used, topdown=False):
   for name in files:
      paths.append(os.path.join(root, name))
#%% SVCCA Similarity between same sample size vs. different sample size
def get_feature_col_indices(df: pd.DataFrame) -> List[int]:
    total_cols = df.shape[1]
    while True:
        if total_cols in df.columns:
            break
        total_cols -= 1
        
    col_indices = [i for i in range(total_cols+1)]
    
    return col_indices

def get_train_test(pca_option=False, start=0, end=4, num_dims: int=20,
                   cpv=False):
    """
    Parameters
    ----------
    pca_option : bool, optional
        True; use Principal Component Analysis. False; use Singular Value 
            Decomposition. The default is True.
    start : int, optional
        The starting index of datasets in global variable <paths>. The default 
        is 0.
    end : int, optional
        The ending index of datasets in global variable <paths>. The default 
        is 4.
    num_dims : int, optional
        The number of reduced dimensions to keep.
    cpv: bool, optional
        Use cumulative percent variance > 0.99 to decide number of dims to 
            keep?

    Returns
    -------
    train_dfs : pd.DataFrame
        Dimensionality-reduced training data.
    test_dfs : TYPE
        Dimensionality-reduced testing data.

    """
    global paths
    
    train_dfs = []
    test_dfs = []
    
    # If dataframes are with the same name in different folders
    if "new" in dataset_used:
        # First dataset with the same name
        inputs = Inputs(paths)
        inputs.get_df_split(start)
        train, test = process_train_test(inputs, pca_option, num_dims, cpv)
        
        # Accumulate processed train/test dataframes
        train_dfs.append(train)
        test_dfs.append(test)
        
        # Get second dataset with the same name
        paths_2 = paths[start].split("\\")
        
        if paths_2[0][-1] == "1":
            paths_2[0] = paths_2[0].replace("1", "2")
        else:
            paths_2[0] = paths_2[0].replace("2", "1")
        second_dataset = "\\".join(paths_2)
        
        # Second dataset with the same name
        inputs = Inputs([second_dataset])
        inputs.get_df_split(0)
        train, test = process_train_test(inputs, pca_option, num_dims, cpv)
        
        # Accumulate processed train/test dataframes
        train_dfs.append(train)
        test_dfs.append(test)
    
    # If all dfs are in the same folder
    else:
        inputs = Inputs(paths)
        for dataset_num in range(start, end):
            inputs.get_df_split(dataset_num)
            
            train, test = process_train_test(inputs, pca_option, num_dims, cpv)
            
            # Accumulate processed train/test dataframes
            train_dfs.append(train)
            test_dfs.append(test)
        
    
    return train_dfs, test_dfs

# HELPER FUNCTION: Get PCA / SVD / Not Changed Dataframes
def process_train_test(inputs: Inputs, pca_option: bool, 
                       num_dims: int, cpv: bool) -> Tuple[pd.DataFrame, 
                                                          pd.DataFrame]:
    """Transform train/test in <inputs> via PCA/SVD/None"""
    if num_dims > inputs.df_test_data.shape[0]:
        error_str = "num_dims must be within"+ \
            f" [1, {inputs.df_test_data.shape[0]})"
        print(dataset_num)
        raise Exception(error_str)
        

    if pca_option:
        # PRINCIPAL COMPONENT ANALYSIS
        pca_model = pca(inputs.chosen_features)
        if inputs.exclude_train != 1:
            pca_model.compute(inputs.df_train_data, inputs.df_test_data, 
                              whole=False, with_scaler=True, with_std=False)
        else:
            pca_model.compute(inputs.df_test_data, inputs.df_test_data, 
                              whole=False, with_scaler=True, with_std=False)
        
        # Get CPV >= 0.99
        if cpv:
            num_dims = np.where(
                pca_model.get_cum_variance() > 0.99)[0][0]
        
        if num_dims > inputs.df_test_data.shape[0]:
            num_dims = inputs.df_test_data.shape[0] - 1 
        
        # Return PCA-transformed data with selected 'num_dims' dimensions
        return pca_model.pcs_train.loc[:, :num_dims].transpose(
            ).to_numpy(), pca_model.pcs_test.loc[:, :num_dims].transpose(
            ).to_numpy()
    elif not pca_option:
        # SINGULAR VALUE DECOMPOSITION
        # Center Data
        train_centered = inputs.df_train_data.to_numpy() - \
            np.mean(inputs.df_train_data.to_numpy(), axis=0, keepdims=True)
        test_centered = inputs.df_test_data.to_numpy() - \
            np.mean(inputs.df_test_data.to_numpy(), axis=0, keepdims=True)
        
        # Perform SVD
        Ub1, sb1, Vb1 = np.linalg.svd(train_centered.transpose(), 
                                      full_matrices=False)
        Ub2, sb2, Vb2 = np.linalg.svd(test_centered.transpose(), 
                                      full_matrices=False)
        
        # Get CPV >= 0.99
        cum_perc = np.cumsum(sb1)/sum(sb1)
        num_dims = np.where(cum_perc >= 0.99)[0][0]
        
        if num_dims > inputs.df_test_data.shape[0]:
            num_dims = inputs.df_test_data.shape[0] - 1 
        # Get SVD Projections
        svb1 = np.dot(sb1[:num_dims]*np.eye(num_dims), Vb1[:num_dims])
        svb2 = np.dot(sb2[:num_dims]*np.eye(num_dims), Vb2[:num_dims])
        
        # Save SVD-transformed data with selected 'num_dims' dimensions
        return svb1, svb2
    else:
        return inputs.df_train_data, inputs.df_test_data


#%% Get Training Sample Sizes
def get_sample_sizes(start: int = 0, end: int = None, 
                     unique:bool=False) -> Tuple[list, list]:
    """Return training and test sample sizes for datasets from 
    idx <start> to <end> in <paths>."""
    global paths
    
    if end is None:
        end = len(paths)
    train_dfs, test_dfs = get_train_test(pca_option=True, start=start, 
                                         end=end, num_dims=1)
    
    train_sizes = []
    test_sizes = []
    
    for i in range(len(train_dfs)):
        if unique and train_dfs[i].shape[1] not in train_sizes:
            train_sizes.append(train_dfs[i].shape[1])
            test_sizes.append(test_dfs[i].shape[1])
        elif not unique:
            train_sizes.append(train_dfs[i].shape[1])
            test_sizes.append(test_dfs[i].shape[1])
            
    return train_sizes, test_sizes

#%%
def _get_cca(dfs: List[pd.DataFrame], 
             within_model: bool) -> Tuple[np.array,
                                          List[int],
                                          List[int]]:
    """Helper Function. Return array of pairwise mean CCA for all subsets of 
    <dfs> of size 2, and two lists corresponding to the datasets compared
    pairwise. 
    
    If within_model, each df from <dfs> is subset 50% and SVCCA is calculated 
    between subsets of one df."""
    mean_cca = []
    compared_x = []
    compared_y = []
    
    if within_model:
        i = 0
        for df in dfs:
            if df.shape[0] % 2 != 0:
                idx = np.random.choice(inputs.df_train.index.tolist())
                df.drop(index=idx, inplace=True)
            
            df_a, df_b = train_test_split(df,
                                          train_size=0.5,
                                          shuffle=True)
            
            results = cca_core.get_cca_similarity(df_a, 
                                                  df_b, 
                                                  epsilon=1e-10,
                                                  verbose=False)
            plt.plot(results["cca_coef1"], lw=2.0, label="%d - %d" % (i,i))
            
            # For Plotting
            compared_x.append(i)
            compared_y.append(i)
            
            mean_cca.append(results["cca_coef1"].mean())
            i += 1
            del results
    else:
        for f in range(len(dfs)):
            for g in range(len(dfs)):
                if f!=g:
                    results = cca_core.get_cca_similarity(dfs[f], 
                                                          dfs[g], 
                                                          epsilon=1e-10,
                                                          verbose=False)
                    plt.plot(results["cca_coef1"], lw=2.0, label="%d - %d" % 
                             (f,g))
                    
                    # For Plotting
                    compared_x.append(f)
                    compared_y.append(g)
                    
                    mean_cca.append(results["cca_coef1"].mean())
                    del results
                    
    return mean_cca, compared_x, compared_y


def compare_svcca(start=0, end=4, 
                  pca_option=False, 
                  num_dims=10,
                  within_model: bool=False,
                  save=False) -> Tuple[np.array, np.array,
                                       Optional[Tuple[int, int]]]:
    """
    Parameters
    ----------
    start : int, optional
        The starting index of datasets in global variable <paths>. The default 
        is 0.
    end : int, optional
        The ending index of datasets in global variable <paths>. The default 
        is 4.
    pca_option : bool, optional
        True; use Principal Component Analysis. False; use Singular Value 
        Decomposition. The default is True.
    num_dims : int, optional
        The number of reduced dimensions to keep. The default is 20.
    save : bool, optional
        If yes, save plots in <absolute_dir> + "results/graphs/svcca/". 
        The default is False.

    Returns
    -------
    Mean Training CCA, Mean Test CCA
    """
    # Get list of train & test dataframes
    train_dfs, test_dfs = get_train_test(pca_option=pca_option, start=start, 
                                         end=end, num_dims=num_dims, cpv=True) # NOTE: CPV is True.
    
    # Get CCA Similarity for TRAINING SETS
    train_mean_cca, compared_x, compared_y = _get_cca(train_dfs, within_model)
    print("Train Mean CCA:", np.mean(np.array(train_mean_cca)))
    
    # Create Plot for Training Data
    plt.title("%s | Train Feature Sets %d to %d with %d dimensions" % (
        dataset_used, start, end, num_dims))
    plt.grid()
    plt.xlabel("Sorted CCA Correlation Coeff Idx")
    plt.ylabel("CCA Correlation Coefficient Value")
    plt.legend()
    if save==True:
        plt.savefig(absolute_dir+"results/graphs/svcca/%s/train_%d_%d(%d).png" % (
            dataset_used, start, end, num_dims))
    plt.show()
    
    # Get CCA Similarity for TEST SETS
    test_mean_cca, compared_x, compared_y = _get_cca(test_dfs, within_model)
    print("Test Mean CCA:", np.mean(np.array(test_mean_cca)))
    
    # Create Plot for Test Data
    plt.title("%s | Test Feature Sets %d to %d with %d dimensions" % (
        dataset_used, start, end, num_dims))
    plt.xlabel("Sorted CCA Correlation Coeff Idx")
    plt.ylabel("CCA Correlation Coefficient Value")
    plt.grid()
    plt.legend()
    if save==True:
        plt.savefig(absolute_dir+"results/graphs/svcca/%s/test_%d_%d(%d).png" % (
            dataset_used, start, end, num_dims))
    plt.show()
    
    return np.array(train_mean_cca), np.array(test_mean_cca), (compared_x, 
                                                               compared_y)


# Get SVCCA vs. Num Dimensions
def svcca_dim():
    try:
        os.mkdir(absolute_dir+"results/graphs/svcca")
    except:
        pass
    try:
        os.mkdir(absolute_dir+"results/graphs/svcca/"+dataset_used)
    except:
        pass
    
    start=int(input("Start: "))
    end=int(input("End: "))
    dims=[1,2,3,5,10,15,20]
    train_svcca=[]
    test_svcca=[]
    for i in dims:
        train_corr,test_corr=compare_svcca(start=start,end=end,pca_option=True, num_dims=i, save=True)
        train_svcca.append(train_corr)
        test_svcca.append(test_corr)
    
    fig=plt.figure()
    fig.suptitle("%s | Train & Test Feature Sets (%d-%d)" % (dataset_used, start, end))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(dims, train_svcca)
    ax1.set_xlabel("Dimensions Kept")
    ax1.set_ylabel("Mean SVCCA (Train Set)")
    ax1.set_ylim(0,1)
    ax2.plot(dims, test_svcca)
    ax2.set_xlabel("Dimensions Kept")
    ax2.set_ylabel("Mean SVCCA (Test Set)")
    ax2.set_ylim(0,1)
    fig.tight_layout(pad=1.0)
    plt.savefig(absolute_dir+"results/graphs/svcca/%s/train-test_%d_%d.png" % (dataset_used, start, end))
    plt.show()


# WRAPPER FUNCTION: Get SVCCA at all sample sizes.
def get_svcca_all(start: int = 0, end: int=16, num_dims: int = 20, groups: int=4):
    """Returns average training and test SVCCA at each sample size 
            in datasets <start> to <end> in <paths>. Assumes datasets with same
            number of samples are grouped in <groups> datasets.
    
    ==Preconditions==:
        - KFolds multiples of 4. e.g. Dataset 0, 1, 2 and 3 are of the same 
            fold.
    """
    train_svcca=[]
    test_svcca=[]
    for dataset_num in range(start, end, groups):
        train_corr, test_corr, points = compare_svcca(start=dataset_num, 
                                              end=dataset_num+groups,
                                              pca_option=False, 
                                              num_dims=num_dims,
                                              save=False)
        train_svcca.append(train_corr)
        test_svcca.append(test_corr)
    
    return train_svcca, test_svcca, points
#%% CV Accuracy
# FUNCTION: Get CV at one sample size for 4 folds.
def get_cv_folds(start: int = 0, end: int = 4, num_dims: int = 20) -> float:
    """Return Average CV from <start> to <end> datasets in <paths>.
    
    ==Preconditions==:
        - datasets must be of the same sample size
        - assumes that datasets are k folds of the same original dataset.
    """
    global absolute_dir, dataset_used
    
    result_files = []
    for some_path in os.listdir(absolute_dir+"results/dataset/"):
        if some_path.find(dataset_used) == 0:
            result_files.append(some_path)
    
    cv_performance = []
    for i in range(start, end):
        df = pd.read_csv(absolute_dir + "results/dataset/" + dataset_used +
                         "_dataset_" + str(i)+".csv")
        cv_performance.append(df.loc[df.features_kept == num_dims].cv)
    
    return np.mean(cv_performance)


# WRAPPER FUNCTION: Get CV at all sample sizes.
def get_cv_all(start: int = 0, end: int = 16, num_dims: int = 20) -> list:
    """Returns average CV at each sample size 
            in datasets <start> to <end> in <paths>. 
    
    ==Preconditions==:
        - KFolds multiples of 4. e.g. Dataset 0, 1, 2 and 3 are of the same 
            fold.
    """
    cv_performances = []
    for dataset_num in range(start, end, 4):
        cv_performances.append(get_cv_folds(dataset_num, 
                                            dataset_num+4, 
                                            num_dims=num_dims))
    
    return cv_performances
#%% SVCCA vs. CV
def compare_cv_svcca_all(start: int = 0, end: int=16, num_dims: int = 20):
    sample_sizes = get_sample_sizes(start=start,end=end, unique=True)
    cv = get_cv_all(start=start, end=end, num_dims=num_dims)
    svcca = get_svcca_all(start=start, end=end, num_dims=num_dims)
    
    train_sample_sizes = np.array(sample_sizes[0])[np.argsort(sample_sizes[0])]
    cv = np.array(cv)[np.argsort(sample_sizes[0])]
    train_svcca = np.array(svcca[0]).mean(axis=1)[np.argsort(sample_sizes[0])]
    test_svcca = np.array(svcca[1]).mean(axis=1)[np.argsort(sample_sizes[0])]
    
    plt.plot(cv, test_svcca, "o")
    plt.title("CV vs. Test SVCCA")
    plt.show()
    
    plt.plot(range(len(sample_sizes[0])), cv, "s-", label="CV")
    plt.plot(range(len(sample_sizes[0])), train_svcca, "o--", 
             label="SVCCA (Training)")
    plt.plot(range(len(sample_sizes[0])), test_svcca, "o--", 
             label="SVCCA (Test)")
    plt.xlabel("Training Sample Sizes")
    plt.xticks(range(len(sample_sizes[0])), train_sample_sizes)
    plt.title(dataset_used)
    plt.legend()
    plt.show()
    
    plt.plot(range(len(sample_sizes[0])), 
             np.array(test_svcca)-np.array(train_svcca), 
             "o--", label="SVCCA (Difference)")
    plt.plot(range(len(sample_sizes[0])), cv, "s-", label="CV")
    plt.xlabel("Training Sample Sizes")
    plt.xticks(range(len(sample_sizes[0])), train_sample_sizes)
    plt.title(dataset_used)
    plt.legend()
    plt.show()

#%%
if __name__ == "__main__":
    # train_svcca, test_svcca, _ = get_svcca_all(end=16, groups=4)
    print(get_sample_sizes())
    _, _, points = get_svcca_all(end=5, groups=1)
    
    # Plot Line
    df = pd.DataFrame({"paths":paths, "svcca": [i[0] for i in _]})
    df.paths = df["paths"].str.replace('.csv', '')
    df["dataset_proportion"] = df["paths"].str.split("_").map(lambda x: int(x[-1]))
    
    df.sort_values("dataset_proportion", inplace=True, ignore_index=True)
    
    ax = df.plot("dataset_proportion", "svcca", kind="scatter", 
                 c="brown")
    ax.set_xlabel("Percentage of Dataset Used")
    ax.set_ylabel("SVCCA Within Model")
    ax.set_ylim([0,1])
    plt.show()
    
    # Create Heatmap
    df = pd.DataFrame()
    for i in range(len(_[0])):
        x = points[0][i]
        y = points[1][i]
        df.loc[x, y] = _[0][i]
        
    df = df[[0,1,2,3,4]]
    
    mask = np.triu(np.ones_like(df, dtype=np.bool))
    sns.heatmap(df,
                cmap="cubehelix",
                mask=mask,
                vmin=0, 
                vmax=1,
                annot=True)
    plt.title(f"{dataset_used} | {num_dims} Principal Components")
