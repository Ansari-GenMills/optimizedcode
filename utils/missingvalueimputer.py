import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.interpolate import UnivariateSpline

class TimeSeriesMissingValueHandler:
    def __init__(self, project_id: str, logger, method='linear', k_neighbors=5):
        self.project_id = project_id
        self.logger = logger  # Logger instance for logging operations
        self.method = method
        self.k_neighbors = k_neighbors

    def _mean_imputation(self, df, columns):
        imputer = SimpleImputer(strategy='mean')
        before_imputation = df[columns].isna().sum()
        df[columns] = imputer.fit_transform(df[columns])
        after_imputation = df[columns].isna().sum()
        return df, before_imputation, after_imputation

    def _median_imputation(self, df, columns):
        imputer = SimpleImputer(strategy='median')
        before_imputation = df[columns].isna().sum()
        df[columns] = imputer.fit_transform(df[columns])
        after_imputation = df[columns].isna().sum()
        return df, before_imputation, after_imputation

    def _ffill_bfill(self, df, columns, method='ffill'):
        if method not in ['ffill', 'bfill']:
            raise ValueError("Method should be 'ffill' or 'bfill'.")
        before_imputation = df[columns].isna().sum()
        df[columns] = df[columns].fillna(method=method)
        after_imputation = df[columns].isna().sum()
        return df, before_imputation, after_imputation

    def _knn_imputation(self, df, columns):
        imputer = KNNImputer(n_neighbors=self.k_neighbors)
        before_imputation = df[columns].isna().sum()
        df[columns] = imputer.fit_transform(df[columns])
        after_imputation = df[columns].isna().sum()
        return df, before_imputation, after_imputation

    def _spline_interpolation(self, df, columns):
        before_imputation = df[columns].isna().sum()
        for col in columns:
            if df[col].dtype.kind in 'fi' and df[col].notna().sum() > 0:
                x = df.index.astype(np.int64)
                y = df[col].values
                valid_indices = ~np.isnan(y)
                spline = UnivariateSpline(x[valid_indices], y[valid_indices], k=3, s=0)
                df[col] = spline(x)
        after_imputation = df[columns].isna().sum()
        return df, before_imputation, after_imputation

    def _linear_interpolation(self, df, columns):
        before_imputation = df[columns].isna().sum()
        df[columns] = df[columns].interpolate(method='linear')
        after_imputation = df[columns].isna().sum()
        return df, before_imputation, after_imputation

    def impute_missing_values(self, df, columns, group_by=None, method=None):
        """
        Impute missing values in the DataFrame based on the specified method and group.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the time series data.
        columns : list
            List of column names to be processed for missing value imputation.
        group_by : list, optional
            List of column names to group by before applying the imputation.
        method : str, optional
            Method to use for imputation. If None, uses the method specified during initialization.
            Options are 'mean', 'median', 'ffill', 'bfill', 'knn', 'spline', 'linear'.

        Returns:
        --------
        df : pandas.DataFrame
            The DataFrame with missing values imputed.
        """
        if method is None:
            method = self.method

        if group_by:
            # Create a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            # Group by specified columns
            grouped = df.groupby(group_by)

            for name, group in grouped:
                # Apply imputation method on each group
                if method == 'mean':
                    group, before, after = self._mean_imputation(group, columns)
                elif method == 'median':
                    group, before, after = self._median_imputation(group, columns)
                elif method in ['ffill', 'bfill']:
                    group, before, after = self._ffill_bfill(group, columns, method=method)
                elif method == 'knn':
                    group, before, after = self._knn_imputation(group, columns)
                elif method == 'spline':
                    group, before, after = self._spline_interpolation(group, columns)
                elif method == 'linear':
                    group, before, after = self._linear_interpolation(group, columns)
                else:
                    raise ValueError("Invalid method. Choose from 'mean', 'median', 'ffill', 'bfill', 'knn', 'spline', 'linear'.")

                # Update the original DataFrame with imputed values
                df.loc[group.index, columns] = group[columns]

        else:
            # No grouping, apply directly to the DataFrame
            if method == 'mean':
                df, before, after = self._mean_imputation(df, columns)
            elif method == 'median':
                df, before, after = self._median_imputation(df, columns)
            elif method in ['ffill', 'bfill']:
                df, before, after = self._ffill_bfill(df, columns, method=method)
            elif method == 'knn':
                df, before, after = self._knn_imputation(df, columns)
            elif method == 'spline':
                df, before, after = self._spline_interpolation(df, columns)
            elif method == 'linear':
                df, before, after = self._linear_interpolation(df, columns)
            else:
                raise ValueError("Invalid method. Choose from 'mean', 'median', 'ffill', 'bfill', 'knn', 'spline', 'linear'.")

        return df
