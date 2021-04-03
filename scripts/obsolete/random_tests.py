from pca import *
from scipy.stats import ttest_ind_from_stats
from scipy.stats import bartlett
import cca_core

#%% Similarity between Feature Vectors
#FUNCTION: Cosine Similarity
def cosine_similarity(input1, input2):
    """Calculating the cosine similarity of two inputs.
    The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
    `1` denotes they are the most similar.
    Args:
        input1, input2: two input numpy arrays.
    Returns:
        Element-wise cosine similarity of two inputs.
    """
    similarity = 1 - scipy.spatial.distance.cosine(input1, input2)
    rounded_similarity = int((similarity * 10000)) / 10000.0
    return rounded_similarity


def check_train_similarity(df_train_data):
    train_similarities=np.array([])
    for i,j in itertools.combinations(df_train_data.index.tolist(), 2):
        train_similarities=np.append(train_similarities, cosine_similarity(df_train_data.loc[i,:], df_train_data.loc[j,:]))
    return train_similarities


def check_test_similarity(df_test_data):
    test_similarities=np.array([])
    for i,j in itertools.combinations(df_test_data.index.tolist(), 2):
        test_similarities=np.append(test_similarities, cosine_similarity(df_test_data.loc[i,:], df_test_data.loc[j,:]))
    return test_similarities


def check_train_test_similarity(df_train_data, df_test_data, n_iter=100):
    train_test_similarities=np.array([])
    for i in range(n_iter):
        train_idx=random.choice(df_train_data.index.tolist())
        test_idx=random.choice(df_test_data.index.tolist())
        train_test_similarities=np.append(train_test_similarities,
                                    cosine_similarity(df_train_data.loc[train_idx,:], df_test_data.loc[test_idx,:]))
    return train_test_similarities


#%%FUNCTION: Return Mean Squared Error between Original Features and PCA-Transformed Features for Each Observation
def compare_pca_inverse(df_train_data, df_test_data, pca_model: PCA, num_features: int, split=False):
    if split==False:
        df_untrans=pd.concat([df_train_data, df_test_data])
        pca_transformed_df=pd.concat([pca_train, pca_test])
        df_inverse_trans=pca_model.train_scaler.inverse_transform(pca_model.inverse_transform(pca_transformed_df))
        df_diff=pd.DataFrame(df_untrans.to_numpy()-df_inverse_trans)
        return df_diff.apply(abs).mean(axis=1)
    else:
        df_train_inverse_trans=get_pca_inverse(pca_model, pca_model.pcs_train, num_features)
        df_test_inverse_trans=get_pca_inverse(pca_model, pca_model.pcs_test, num_features)
        df_diff_train=pd.DataFrame(df_train_data-df_train_inverse_trans)
        df_diff_test=pd.DataFrame(df_test_data-df_test_inverse_trans)
        return (df_diff_train**2).mean(axis=1), (df_diff_test**2).mean(axis=1)


def get_pca_inverse(pca_model: PCA, df_transformed: pd.DataFrame,
                    num_features: int) -> pd.DataFrame:
    """
    Parameters
    ----------
    pca_model : PCA
        Fitted PCA object.
    df_loadings : pd.DataFrame
        Dataframe containing projected loadings from PCA transformation.
    num_features : int
        The number of principal components to keep.

    Returns
    -------
    pd.DataFrame
        Return df_loadings PCA inverse transformed back to original space,
        given <num_features> principal components kept.

        NOTE: Dropped principal components are replaced with sparse columns.
    """
    sparse_df = pd.DataFrame(0.0, index=np.arange(len(df_transformed)),
                             columns=range(num_features,
                                           df_transformed.shape[1]))
    pcs_filled = pd.concat([df_transformed.loc[:,:num_features-1],
                            sparse_df], axis=1)

    return pd.DataFrame(pca_model.train_scaler.inverse_transform(
        pca_model.inverse_transform(pcs_filled)))


def get_reconstruction_errors(pca_model: PCA,
                              inputs: Inputs,
                              train_or_test: str="train",
                              num_features: int=10) -> pd.DataFrame:
    """
    Parameters
    ----------
    pca_model : PCA
        Fitted PCA object.
    inputs: Inputs
        Contains attributes
    train_or_test: str
        Must be "train" or "test"
    num_features : int
        The number of principal components to keep.

    Returns
    -------
    pd.DataFrame
        Contains the residual reconstruction errors.
    """
    if train_or_test == "train":
        df_transformed = pca_model.pcs_test
        df_original = inputs.df_test_data
    else:
        df_transformed = pca_model.pcs_train
        df_original = inputs.df_train_data

    df_reconstructed = get_pca_inverse(pca_model,
                                       df_transformed, num_features)
    df_errors = pd.DataFrame(df_original.reset_index(drop=True)
                             - df_reconstructed)
    return df_errors


def variances_of_the_reconstruction_error(pca: PCA,
                                         inputs: Inputs,
                                         train_or_test: str="train") -> List:
    """Return list of mean variances of the reconstruction errors.

    Precondition:
        PCA object's compute method has been called.
    """
    vre = []

    for num_features in range(0, pca_model._max_pcs):
        reconstruction_errors = get_reconstruction_errors(pca,
                                         inputs,
                                         train_or_test,
                                         num_features)
        vre.append(np.mean(reconstruction_errors.var()))

    return vre



#%% RUN Variance of the Reconstruction Error
def run_vre():
    col_indices = [i for i in range(512)]
    df = pd.read_csv(paths[12], index_col=False)
    try:
        df = df.drop("Unnamed: 0", axis=1)
    except:
        pass

    #GET TRAINING DATA & VAL & TEST DATA
    df_train = df[df.phase=="train"]
    df_test = df[df.phase=="val"]

    df_train_data = df_train.loc[:,col_indices]
    df_test_data = df_test.loc[:,col_indices]

    #Get, Plot and Save results
    pca_model=pca()
    pca_model.compute(df_train_data,df_test_data)

    train_var_ = variances_of_the_reconstruction_error(pca_model,
                                                      pca_model.pcs_train,
                                                      df_train_data,
                                                      (1, 100))

    test_var_ = variances_of_the_reconstruction_error(pca_model,
                                                      pca_model.pcs_test,
                                                      df_test_data,
                                                      (1, 100))

    plt.plot(list(range(1, 100)), train_var_, c="Hello")

    plt.plot(list(range(1, 100)), test_var_)

#%% HELPER FUNCTIONS: Get vector component of observation on Principal Component
def get_vector_component(base_vector, some_vector):
    return ((base_vector.dot(some_vector))/(base_vector.dot(base_vector)))*base_vector


# WRAPPER FUNCTION: Apply get_vector_component() on PCA-transformed dataframe
def get_df_vector_comp(x, l):
    global pca_model
    return get_vector_component(pca_model.train_scaler.inverse_transform(pca_model.components_[l]), x)        #NOTE: Inverse Transformation to de-normalize Principal Component Vectors


#FUNCTION: Compare Vector Component (of Training/Testing Observations) on Principal Components
def compare_vcomp_pcs(num_pc=10, display=False):
    global df_train_data, df_test_data, pca_model

    #PCA: Compare Vector Components of PCs between Training and Test Data
    norm_vcomp_train=[]
    norm_vcomp_test=[]
    #Compare for top num_pc components
    for l in range(num_pc):
        # sum_vcomp_train.append(df_train_data.apply(lambda x: get_df_vector_comp(x, l), axis=1).abs().sum())
        # sum_vcomp_test.append(df_test_data.apply(lambda x: get_df_vector_comp(x, l), axis=1).abs().sum())
        norm_vcomp_train.append(np.median(np.linalg.norm(df_train_data.apply(lambda x: get_df_vector_comp(x, l), axis=1))))
        norm_vcomp_test.append(np.median(np.linalg.norm(df_test_data.apply(lambda x: get_df_vector_comp(x, l), axis=1))))
    if display!=False:
        print(cosine_similarity(norm_vcomp_train, norm_vcomp_test))


#%% FUNCTION: Get Cosine Similarity Within and Between Clusters
def compare_cluster_similarity():
    global pca_model, num_cluster, dataset_num, chosen_features
    include_elbow=False
    n_iter=100
    random_state=None
    df_data=df_test.copy()

    #Get train and test data
    cluster_train=pca_model.pcs_train.loc[:,:chosen_features-1]
    cluster_val=pca_model.pcs_test.loc[:,:chosen_features-1]

    #Fit Training, Predict Cluster
    cluster_prediction,cluster_distances,metrics=cluster_kmeans(cluster_train,
                                                                 cluster_val,
                                                                 num_clusters=num_cluster,
                                                                 n_iter=n_iter,
                                                                 r_state=random_state,
                                                                 include_elbow=include_elbow)
    #Getting cluster sizes
    cluster_indiv_num=pd.Series(cluster_prediction).value_counts().index.to_list()
    cluster_sizes=pd.Series(cluster_prediction).value_counts().values

    sorted_cluster_sizes=pd.DataFrame(cluster_sizes, index=cluster_indiv_num).sort_index().values.flatten()

    #Get cluster test accuracies
    df_data["cluster"]=cluster_prediction
    if model_goal=="regression":
        df_data["prediction_accuracy"]=df_data.apply(lambda x: np.sqrt(((x.predictions - x.labels) ** 2)), axis=1)
    else:
        df_data["prediction_accuracy"]=(df_data.predictions==df_data.labels)
    df_cluster_accuracies=df_data.groupby(by=["cluster"]).mean()["prediction_accuracy"]

    print(df_cluster_accuracies)
    for i in range(4):
        cluster_num=i
        print("Cluster", i)
        cluster_idx=df_data[df_data.cluster==cluster_num].index.tolist()
        sim_test_transformed=check_test_similarity(pca_model.pcs_test.loc[:,:chosen_features-1].loc[np.array(cluster_idx)-len(pca_model.pcs_train),:])
        sim_test_untransformed=check_test_similarity(df_data[df_data.cluster==cluster_num].loc[:, col_indices])

        print("Within Test Cluster Similarity [Original]:",sim_test_untransformed.mean(), sim_test_untransformed.std())
        print("Within Test Cluster Similarity [PCA-transformed]:",sim_test_transformed.mean(), sim_test_transformed.std())

        sim_train_test_transformed=check_train_test_similarity(pca_model.pcs_train.loc[:,:chosen_features-1],
                                    pca_model.pcs_test.loc[:,:chosen_features-1].loc[np.array(cluster_idx)-len(pca_model.pcs_train),:],
                                    n_iter=1000)
        sim_train_test_untransformed=check_train_test_similarity(df_train_data.loc[:, col_indices],
                                                                 df_test_data.loc[cluster_idx, col_indices],
                                                                 n_iter=1000)

        print("Between Train-Test Similarity [Original]:",sim_train_test_untransformed.mean(), sim_train_test_untransformed.std())
        print("Between Train-Test Similarity [PCA-transformed]:",sim_train_test_transformed.mean(), sim_train_test_transformed.std())

# NOTE
        #[Original]
            #cluster prediction based on chosen PCA-transformed features
                #but cosine similarity used on original 512-dimensional features

#%% Checking Stability of First PC / Singular Vector
def check_pc_stability(pc):
    pc_accum=[]
    for i in range(5):
        pca_model=pca()
        pca_model.compute(df_train_data, df_test_data, whole=False, with_scaler=True, with_std=False)
        pc_accum.append(pca_model.pcs_train.loc[:,pc].to_numpy())
        del pca_model
    print(check_train_similarity(pd.DataFrame(pc_accum)))


#%% TEST: Significant Difference between PCA Inverse Transforms (with sparsely filled in PCs) and Original Features
def test_inverse_difference():
    col_indices=[str(i) for i in range(512)]
    num_cluster=4
    dataset_num=10
    chosen_features=5
    for dataset_num in range(16):
        print(paths[dataset_num])

        df=pd.read_csv(paths[dataset_num], index_col=False)
        try:
            df=df.drop("Unnamed: 0", axis=1)
        except:
            pass

        #GET TRAINING DATA & VAL & TEST DATA
        df_train=df[df.phase=="train"]
        df_test=df[df.phase=="val"]

        df_train_data=df_train.loc[:,col_indices]
        df_test_data=df_test.loc[:,col_indices]

        #Get, Plot and Save results
        pca_model=pca()
        pca_model.compute(df_train_data,df_test_data)

        #Assessing PCA-Inverse
        def ttest_inverse_diff(num_features):
            inverse_diffs=compare_pca_inverse(df_train_data=df_train_data, df_test_data=df_test_data, pca_model=pca_model, num_features=num_features, split=True)
            if bartlett(inverse_diffs[0], inverse_diffs[1]).pvalue < 0.05:
                equal_var=False
            else:
                equal_var=True
            return ttest_ind_from_stats(inverse_diffs[0].mean(), inverse_diffs[0].std(), len(inverse_diffs[0]),
                                             inverse_diffs[1].mean(), inverse_diffs[1].std(), len(inverse_diffs[1]),
                                             equal_var=equal_var)

        p_value=[]
        t_statistic=[]
        effect_sizes=[]
        for num_features in range(1,50,1):
            test=ttest_inverse_diff(num_features)
            print("Dimensions: %d, p-value: %f" % (num_features, test.pvalue))
            print("T-Statistic: %f" % (test.statistic))

            inverse_diffs=compare_pca_inverse(df_train_data=df_train_data, df_test_data=df_test_data, pca_model=pca_model, num_features=num_features, split=True)
            pooled_std=math.sqrt((inverse_diffs[0].std()**2+inverse_diffs[1].std()**2)/2)
            effect_size=(inverse_diffs[0].mean()-inverse_diffs[1].mean())/pooled_std
            print("Effect Size: %f" % (effect_size))

            p_value.append(test.pvalue)
            t_statistic.append(test.statistic)
            effect_sizes.append(effect_size)

        fig = plt.figure()
        fig.suptitle("%s | Dataset %d" % (dataset_used,dataset_num))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        plt.tight_layout(pad=1.5)
        ax1.axhline(0.05,
                    ms=2,
                    color="black",
                    linestyle="--")
        ax1.plot(p_value)
        ax1.set_ylabel('p-value')

        ax2.plot(t_statistic)
        ax2.set_ylabel('T-Statistic')
        ax3.plot(effect_sizes)
        ax3.set_ylabel("Effect Size")
        plt.show()
        del pca_model,df_train_data, df_test_data

    pooled_var=math.sqrt(((len(inverse_diffs[0])-1)*inverse_diffs[0].std()**2)+((len(inverse_diffs[1])-1)*inverse_diffs[1].std()**2)
                         /(len(inverse_diffs[0])+len(inverse_diffs[1])-2))
    pooled_std=math.sqrt((inverse_diffs[0].std()**2+inverse_diffs[1].std()**2)/2)

#%% PROJECTION PURSUIT: Robust PCA

# from direpack import ppdire, capi, dicomo
#
# def proj_pursuit(df_train_data):
#     pca = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'var', 'center': 'median'}, n_components=4, optimizer='grid',optimizer_options={'ndir':1000,'maxiter':1000}, center_data=True)
#     pca.fit(df_train_data,ndir=200)
#     return pca.x_loadings_

#%% Plot Top 2 Principal Components


#%%
if __name__ == "__main__":
    inputs = Inputs(paths)

    df_selection_methods = pd.DataFrame()

    for dataset_num in inputs.which_datasets:
        # Get training and test data & Include only features
        inputs.get_df_split(dataset_num)

        # PCA: Fit & Transform
        pca_model = pca(inputs.chosen_features)

        if inputs.exclude_train != 1:
            pca_model.compute(inputs.df_train_data, inputs.df_test_data,
                              whole=False, with_scaler=True, with_std=False)
        else:
            pca_model.compute(inputs.df_test_data, inputs.df_test_data,
                              whole=False, with_scaler=True, with_std=False)

        # Get top 2 PCA-transformed training features and testing features
        a = pca_model.pcs_train.iloc[:, :2]
        b = pca_model.pcs_test.iloc[:, :2]

        # Plot 2D reduced features by training and test set
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)

        # Training Set
        a.plot(x=0, y=1, kind="scatter",
               marker="o", c=inputs.df_train.labels,
               colormap="Spectral", alpha=0.2, colorbar=False, grid=False,
               ax=ax1)
        a.plot(x=0, y=1, kind="scatter",
               marker="o", c=inputs.df_train.predictions,
               colormap="Spectral", alpha=0.2, colorbar=False, grid=False,
               ax=ax1,
               ylabel = "Training Set")
        a.loc[(inputs.df_train.labels != inputs.df_train.predictions
               ).values].plot(x=0, y=1,
                              kind="scatter", color="red", ax=ax1,
                              grid=False, alpha=0.8)

        # Testing Set
        b.plot(x=0, y=1, kind="scatter",
                     marker="o", c=inputs.df_test.labels,
                     colormap="BrBG", alpha=0.2, colorbar=False, grid=False,
                     ax=ax2)
        b.plot(x=0, y=1, kind="scatter",
                     marker="o", c=inputs.df_test.predictions,
                     colormap="BrBG", alpha=0.2, colorbar=False, grid=False,
                     ax=ax2,
                     ylabel = "Testing Set")
        b.loc[(inputs.df_test.labels != inputs.df_test.predictions
               ).values].plot(x=0, y=1, kind="scatter",
                              color="red", ax=ax2, grid=False, alpha=0.8)

        fig.suptitle(paths[dataset_num].replace(absolute_dir+data_dir,
                                                "").replace(".csv",
                                                            "").replace("\\",
                                                                        "||"))
        plt.tight_layout()
        plt.show()


