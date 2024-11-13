import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

class TimeSeriesOutlierHandler:
    def __init__(self, project_id, logger=None, method='zscore', threshold=3.0):
        self.project_id = project_id
        self.logger = logger
        self.method = method
        self.threshold = threshold

    def _zscore_outlier_detection(self, df, columns):
        z_scores = np.abs(stats.zscore(df[columns], nan_policy='omit'))
        return (z_scores > self.threshold)

    def _iqr_outlier_detection(self, df, columns):
        outliers = pd.DataFrame(index=df.index)
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        return outliers

    def _rolling_outlier_detection(self, df, columns, window=5, sigma=3.0):
        outliers = pd.DataFrame(index=df.index, columns=columns)
        for col in columns:
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            outliers[col] = np.abs(df[col] - rolling_mean) > (sigma * rolling_std)
        return outliers

    def _robust_scaler_outlier_detection(self, df, columns):
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df[columns])
        median = np.median(scaled_data, axis=0)
        mad = np.median(np.abs(scaled_data - median), axis=0)
        outliers = np.abs(scaled_data - median) > (3 * mad)
        return pd.DataFrame(outliers, index=df.index, columns=columns)

    def _outlier_treatment(self, df, outliers, columns, method, group_by=None):
        if method == 'zscore' or method == 'robust_scaler':
            if group_by:
                for name, group in df.groupby(group_by):
                    for col in columns:
                        median_value = group[col].median()
                        df.loc[outliers[col] & (df.index.isin(group.index)), col] = median_value
            else:
                for col in columns:
                    median_value = df[col].median()
                    df.loc[outliers[col], col] = median_value

        elif method == 'iqr':
            if group_by:
                for name, group in df.groupby(group_by):
                    for col in columns:
                        Q1 = group[col].quantile(0.25)
                        Q3 = group[col].quantile(0.75)
                        median_value = (Q1 + Q3) / 2
                        df.loc[outliers[col] & (df.index.isin(group.index)), col] = median_value
            else:
                for col in columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    median_value = (Q1 + Q3) / 2
                    df.loc[outliers[col], col] = median_value

        elif method == 'rolling':
            for col in columns:
                rolling_median = df[col].rolling(window=5, min_periods=1).median()
                df.loc[outliers[col], col] = rolling_median[outliers[col]]
                
        return df

    def handle_outliers(self, df, columns, group_by=None, method=None, **kwargs):
        if method is None:
            method = self.method

        if group_by:
            df = df.copy()
            grouped = df.groupby(group_by)

            for name, group in grouped:
                if method == 'zscore':
                    outliers = self._zscore_outlier_detection(group, columns)
                elif method == 'iqr':
                    outliers = self._iqr_outlier_detection(group, columns)
                elif method == 'rolling':
                    window = kwargs.get('window', 5)
                    sigma = kwargs.get('sigma', 3.0)
                    outliers = self._rolling_outlier_detection(group, columns, window=window, sigma=sigma)
                elif method == 'robust_scaler':
                    outliers = self._robust_scaler_outlier_detection(group, columns)
                else:
                    raise ValueError("Invalid method. Choose from 'zscore', 'iqr', 'rolling', 'robust_scaler'.")

                df.loc[group.index, columns] = self._outlier_treatment(group, outliers, columns, method, group_by)
        else:
            if method == 'zscore':
                outliers = self._zscore_outlier_detection(df, columns)
            elif method == 'iqr':
                outliers = self._iqr_outlier_detection(df, columns)
            elif method == 'rolling':
                window = kwargs.get('window', 5)
                sigma = kwargs.get('sigma', 3.0)
                outliers = self._rolling_outlier_detection(df, columns, window=window, sigma=sigma)
            elif method == 'robust_scaler':
                outliers = self._robust_scaler_outlier_detection(df, columns)
            else:
                raise ValueError("Invalid method. Choose from 'zscore', 'iqr', 'rolling', 'robust_scaler'.")

            df = self._outlier_treatment(df, outliers, columns, method)

        return df
