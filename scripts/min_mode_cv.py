from __future__ import annotations
from typing import Optional

import pandas as pd
import numpy as np
from scipy.stats import mode, variation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


def get_min_mode_cv(X: pd.DataFrame, y: np.array, y_pred: np.array,
                    X_train: Optional[pd.DataFrame] = None) -> int:
    """Return the number of principal components suggested by Minimum Mode CV.

    ==Parameters==
        X: pd.DataFrame
            Contains extracted features from testing set
        y: np.array
            true labels of the testing set. Each value must correspond to each
            row in X.
        y_pred: np.array
            Convolutional Neural Network predictions of the testing set. Each
            value must correspond to each row in X.
        X_train: OPTIONAL pd.DataFrame
            Contains extracted features from training set

    DISCLAIMER: Methodology should not be performed on testing sets without
                labels.
    """
    # Principal Component Analysis
    if X_train:
        pca = PCA()
        X_train_pcs = pca.fit_transform(X_train)
        X_pcs = pca.transform(X)
    else:
        pca = PCA()
        X_train_pcs = None
        X_pcs = pca.fit_transform(X)

    # Infer CNN Model Task
    task = get_model_task(y)
    # Get CNN prediction performances
    if task == "classification":  # Accuracy
        pred_performance = (y == y_pred)
    else:  # Mean Squared Error
        pred_performance = mean_squared_error(y, y_pred)

    # Accumulator for CVs over 1 to 70 PCs
    cv_accum = []
    for num_pcs in range(1, 71):
        # Get Cluster Assignment
        kmeans = KMeans(num_clusters=4)
        if X_train_pcs is not None:  # if X_train is declared, fit on training
            kmeans.fit(X_train_pcs.loc[:, :num_pcs - 1])
        else:
            kmeans.fit(X.loc[:, :num_pcs - 1])
        cluster_pred = kmeans.predict(X.loc[:, :num_pcs - 1])

        # Get Mean CNN Performances for Each Cluster
        df = pd.DataFrame({"cluster": cluster_pred,
                           "performance": pred_performance})
        df = df.groupby(by=["cluster"]).mean()["performance"]

        # Empty clusters are assigned -1
        mean_performances = []
        for cluster in range(4):
            try:
                mean_performances.append(
                    df.loc[cluster])
            except KeyError:  # if empty cluster
                mean_performances.append(-1)
        mean_performances = np.array(mean_performances)

        # Accumulate Coefficient of Variation
        cv_accum.append(variation(mean_performances))
    cv_accum = np.array(cv_accum)

    # Return the Minimum Number of PCs that Result in the Mode CV
    return np.where(cv_accum == mode(cv_accum).mode)[0][0]


def get_model_task(y: np.array) -> str:
    """Return model task based on <y>.

    If <y> contains categorical data (only integers), return "classification",
    else return "regression".
    """
    if all(y == np.round(y)):
        return "classification"
    return "regression"

