import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# CLASS: PCA Subclass
class MyPCA(PCA):
    """Principal Component Analysis Object. Inherits from scikit-learn PCA
    class."""
    train_scaler: StandardScaler
    test_scaler: StandardScaler
    pcs_train: pd.DataFrame
    pcs_test: pd.DataFrame
    _max_pcs: int

    def __init__(self, random_state: int = 2020-12-15) ->\
            None:
        super().__init__(random_state=random_state)

        # if SparsePCA, alpha=0.01
        # if TruncatedSVD, n_components=70
    def compute(self,
                df_train_data, df_test_data,
                whole=False,
                with_scaler=True,
                with_std=False) -> None:
        """
        1) Center/Standardize data.
        2) Runs Principal Component Analysis.
        3) Saves model and principal components (as pd.DataFrame) to object

        Returns
        -------
        None.

        """
        if not whole:
            # Center Data
            if with_scaler:
                self.train_scaler = StandardScaler(with_std=with_std)
                df_train_data = self.train_scaler.fit_transform(df_train_data)
                self.test_scaler = StandardScaler(with_std=with_std)
                df_test_data = self.train_scaler.transform(df_test_data)

            # PCA fit-transform data
            train_pcs = self.fit_transform(df_train_data)
            test_pcs = self.transform(df_test_data)
        else:
            # Concat training and test
            df_whole = pd.concat([df_train_data, df_test_data])

            # Center Data
            if with_scaler:
                self.train_scaler = StandardScaler(with_std=with_std)
                df_whole = self.train_scaler.fit_transform(df_whole)

            # PCA fit then transform data
            self.fit(df_whole)
            train_pcs = self.transform(
                pd.DataFrame(
                    df_whole).loc[:df_train_data.index.tolist()[-1], :])
            test_pcs = self.transform(
                pd.DataFrame(
                    df_whole).loc[df_train_data.index.tolist()[-1]+1:, :])

        # Save instance and principal components
        self.pcs_train = pd.DataFrame(train_pcs)
        self.pcs_test = pd.DataFrame(test_pcs)
        self._max_pcs = len(self.pcs_train.columns) - 1

    def get_cum_variance(self) -> pd.Series:
        """Returns cumulative percent explained variance for all principal
        components.
        """
        return pd.Series(self.explained_variance_ratio_[:]).cumsum()

    def get_total_variance(self, display=False):
        """Return Total Variance."""
        if display:
            print("Sum Eigenvalues/Trace/Total Variance:",
                  self.explained_variance_.sum())
        return self.explained_variance_.sum()

    def get_general_variance(self, display=False) -> np.array:
        """Return General Variance."""
        if not display:
            print("General Variance:",
                  np.linalg.det(self.get_covariance()))
        return np.linalg.det(self.get_covariance())

    def get_noise_variance(self, display=False) -> np.array:
        """Return Noise Variance"""
        if not display:
            print("Noise Variance:", self.noise_variance_)
        return self.noise_variance_

    def get_max_pc(self) -> int:
        """Return maximum number of principal components.
        ==Precondition==:
            - compute is called.
        """
        return self.pcs_test.shape[1] + 1
