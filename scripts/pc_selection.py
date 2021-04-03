from __future__ import annotations
import pandas as pd
import numpy as np

class PrincCompSelection:
    _pca_model: PCA

    def __init__(self, inputs: Inputs, pca: PCA) -> None:
        """Initializer for Principal Component Selection object.

        ==Precondition==:
            - 'compute' method has been called for <pca>
        """
        self._pca = pca
        self._inputs = inputs

    def cpv(self, percent: float) -> int:
        """Return number of principal components to select, such that the
        Cumulative Percent Variance (CPV) is greater than <percent>.

        <percent> must be in decimal form (e.g. 80% --> 0.8)

        Example:
        >>> pca = PCA()
        >>> pca.compute(df_train, df_test)
        >>> pc_selection = PrincCompSelection(pca)
        >>> print(pc_selection.cpv(0.70))
        <int>
        """
        return np.where(self._pca.get_cum_variance() >= percent)[0][0] + 1

    def pv_10(self) -> int:
        """Return number of principal components to select, such that the
        percent variance explained is at least 0.1.

        From Rand R. Wilcox 2017, Chapter 6: Some Multivariate Methods
        """
        try:
            return np.where(
                self._pca.explained_variance_ratio_ >= 0.1)[0][-1] + 1
        except:
            return 0

    def eig_1(self) -> int:
        """Return number of principal components to select, such that remaining
        pcs have eigenvalues greater than 1.

        If no singular_values_ are > 1, return 0.
        """
        try:
            return np.where(self._pca.explained_variance_ > 1)[0][-1] + 1
        except:
            return 0

    def eig_avg(self) -> int:
        """Return number of principal components to select, such that remaining
        pcs have eigenvalues greater than the average eigenvalue.
        """
        return np.where(self._pca.singular_values_ >
                        self._pca.singular_values_.mean())[0][-1] + 1

    def vre(self) -> int:
        """Return number of principal components to select, based on Variance
        of the Reconstruction Error."""
        raise NotImplementedError

    def pa(self) -> int:
        """Return number of principal components to select, based on Horn's
        Parallel Analysis. Used for Factor Analysis"""
        raise NotImplementedError

    def ac(self) -> int:
        """Return number of principal components to select, based on Auto-
        Correlation."""
        raise NotImplementedError

    def select_pcs(self, idx=1) -> pd.DataFrame:
        """Return a dataframe containing number of principal components, based
        on multiple methods.
        """
        df_pc_select = pd.DataFrame({
            "Cum. Perc. Var. (0.8)": self.cpv(0.8),
            "Cum. Perc. Var. (0.99)": self.cpv(0.99),
            "Perc. Var. (0.1)": self.pv_10(),
            "Eig. 1": self.eig_1(),
            "Eig. Avg.": self.eig_avg()
            },
            index=[idx])

        return df_pc_select

