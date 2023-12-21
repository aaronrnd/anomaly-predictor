from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.anomaly_predictor.utils import resample_hourly, split_long_nans


class DataCleaner:
    """Cleans data in preparation for feature engineering."""

    def __init__(self):
        pass

    def clean_data(
        self,
        data: pd.DataFrame,
        features_to_standardize: list,
        standardization_method: str,
        scaler_args: dict,
        outlier_threshold: float,
        outlier_window: int,
        nan_range_threshold: int,
        impute_nan_window: int,
        impute_nan_period: int,
        impute_nan_center: bool,
        bearing_cond_fill: int,
        convert_to_int: list,
    ) -> list:
        """Wrapper script to run all CleanData functions:
            1. Keeps only features of interest and "Anomaly" column if it exists.
            2. Replaces negative variables with NaNs.
            3. Resamples to hourly data.
            4. Standardizes variables.
            5. Replaces outliers within specified window and defined threshold with NaNs.
            6. Splits the data on long range of NaNs.
            7. Replaces 'Bearing Condition' value as NaN when 'Speed' is 0.
            8. Utilizes rolling mean to impute missing values.
            9. Imputes any missing value in 'Bearing Condition' column with fill_value.
            10. Corrects floats to integers in 'Bearing Condition' and 'Anomaly'.
            11. Drop any rows with NaNs.

        Args:
            data (pd.DataFrame): DataFrame to be cleaned.
            features_to_standardize (list): Columns to be standardized.
            standardization_method (str): Standardization method to be used.
            scaler_args (dict): Parameters for standardization scaler.
            outlier_threshold (float,): Threshold for values to be considered as outliers.
            outlier_window (int): Windowing for threshold for values.
            nan_range_threshold (int): Threshold of min range of NaN length for the data to be split on.
            impute_nan_window (int): Window size via which the rolling mean is derived.
            impute_nan_period (int): Minimum number of observations in window required to have a value.
            impute_nan_center (bool): Whether to set window labels as center of window.
            bearing_cond_fill (int): Value to be imputed for bearing condition where speed is 0.
            convert_to_int (list): List of column names to convert values to integer.

        Returns:
            list: List containing the split and cleaned DataFrame(s).
        """

        data = self._keep_features(data, features_to_standardize)
        data = self._check_negatives(data)
        data = resample_hourly(data)

        standardized_df = self._standardization(
            data, features_to_standardize, standardization_method, scaler_args
        )
        data = self._check_outliers(
            standardized_df,
            data,
            features_to_standardize,
            outlier_threshold,
            outlier_window,
        )

        data_list = split_long_nans(data, nan_range_threshold)
        if "Bearing Condition" in features_to_standardize:
            data_list = self._preprocess_bearing_cond(data_list)

        data_list = self._impute_nans(
            data_list, impute_nan_window, impute_nan_period, impute_nan_center
        )
        if "Bearing Condition" in features_to_standardize:
            data_list = self._impute_bearing_cond(data_list, bearing_cond_fill)
        data_list = self._float_to_int(data_list, convert_to_int)
        data_list = [data.dropna() for data in data_list]
        return data_list

    def _keep_features(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Keeps only specified column names from a given DataFrame. If "Anomaly"
        is within columns, it is also retained.

        Args:
            data (pd.DataFrame): DataFrame with only essential columns kept.
            columns (list): List of columns to be retained.

        Returns:
            data (pd.DataFrame): Processed DataFrame with columns of interest.
        """
        if "Anomaly" in data.columns:
            columns = columns + ["Anomaly"]
        data = data[columns]
        return data

    def _check_negatives(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check negative variables in each feature and replaces them with NaNs.

        Args:
            data (pd.DataFrame): DataFrame to be checked for negative values.

        Returns:
            data (pd.DataFrame): Processed DataFrame with replaced negative values.
        """
        for col in data.columns:
            condition = (data[col] < 0).values
            data.loc[condition, col] = np.NaN
        return data

    def _standardization(
        self,
        data: pd.DataFrame,
        columns: list,
        method: str = "RobustScaler",
        scaler_args: dict = None,
    ) -> pd.DataFrame:
        """Performs standardization for a given dataframe.

        Args:
            data (pd.DataFrame): DataFrame to be standardized.
            columns (list): Columns to be standardized.
            method (str, optional): Choice of standardization method. Defaults to "RobustScaler".
            scaler_args (dict): Parameters for standardization scaler.

        Returns:
            pd.DataFrame: Standardized DataFrame.
        """

        if method == "MinMaxScaler":
            scaler = MinMaxScaler()

        elif method == "StandardScaler":
            scaler = StandardScaler()

        elif method == "RobustScaler":
            scaler = RobustScaler(**scaler_args)

        standardized_array = scaler.fit_transform(data[columns])
        standardized_df = data.copy()
        standardized_df[columns] = standardized_array
        return standardized_df

    def _check_outliers(
        self,
        standardized_df: pd.DataFrame,
        data: pd.DataFrame,
        columns: list,
        outlier_threshold: float,
        outlier_window: int,
    ) -> pd.DataFrame:
        """Check outlier variables in each feature and replaces them with NaNs.

        Args:
            standardized_df (pd.DataFrame): Standardized DataFrame to be checked for outlier values.
            data (pd.DataFrame): Unstandardized DataFrame with outliers to be replaced with NaNs.
            columns (list): List of columns to be checked for outlier values.
            outlier_threshold (float, optional): Threshold for values to be considered as outliers.
                Defaults to None.
            outlier_window (int, optional): Windowing for threshold for values. Defaults to None.

        Returns:
            pd.DataFrame: Processed DataFrame with replaced outlier values.
        """

        for col in columns:
            condition = (
                abs(
                    standardized_df[col]
                    - (standardized_df[col].rolling(window=outlier_window).median())
                )
                >= outlier_threshold
            )
            data.loc[condition, col] = np.NaN

        return data

    @staticmethod
    def _preprocess_bearing_cond(data_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Replaces 'Bearing Condition' value as NaN when 'Speed' is 0.

        Args:
            data_list (list[pd.DataFrame]): list of dataframes to be preprocessed.
                This should be the output of split_long_nans.

        Returns:
            list[pd.DataFrame]: Processed list of dataframes.
        """
        for data in data_list:
            data.loc[data["Speed"] == 0, "Bearing Condition"] = np.nan

        return data_list

    @staticmethod
    def _impute_nans(
        data_list: List[pd.DataFrame],
        lookback_period: int = 5,
        min_periods: int = 1,
        center: bool = True,
    ) -> List[pd.DataFrame]:
        """Utilizes rolling mean to impute missing values. If feature is
        'Bearing Condition', forward fill is used instead.

        Args:
            data_list (List[pd.DataFrame]): list of dataframes to be preprocessed.
            lookback_period (int, optional): window size via which the rolling mean
                is derived. Defaults to 5.
            min_periods (int, optional): Minimum number of observations in
                window required to have a value. Defaults to 1.
            center (bool, optional): If False, set the window labels as the
                right edge of the window index. If True, set the window labels
                as the center of the window index. Defaults to False.

        Returns:
            List[pd.DataFrame]: Processed list of dataframes.
        """
        for i, data in enumerate(data_list):
            if "Bearing Condition" in data.columns:
                data_list[i]["Bearing Condition"] = data["Bearing Condition"].fillna(
                    method="ffill"
                )

            rolling_mean = data.rolling(lookback_period, min_periods, center, axis=0).mean()
            if not center:
                rolling_mean = rolling_mean.shift()
            data_list[i] = data.fillna(rolling_mean)

        return data_list

    @staticmethod
    def _impute_bearing_cond(
        data_list: List[pd.DataFrame], fill_value: int = -1
    ) -> List[pd.DataFrame]:
        """Imputes any missing value in 'Bearing Condition' column with fill_value.

        In the event where speed is 0 throughout the asset, no values would be
        imputed in prior function _impute_nans and so this function would then impute
        with a specified fill_value throughout.

        Args:
            data_list (List[pd.DataFrame]): list of dataframes to be preprocessed.
            fill_value (int, optional): Value to be imputed. Defaults to -1.

        Returns:
            List[pd.DataFrame]: Processed list of dataframes.
        """
        for i, data in enumerate(data_list):
            data_list[i]["Bearing Condition"] = data["Bearing Condition"].fillna(
                value=fill_value
            )
        return data_list

    def _float_to_int(
        self, data_list: List[pd.DataFrame], columns: list
    ) -> List[pd.DataFrame]:
        """Converts floats in a specified column to integers.

        Args:
            data_list (List[pd.DataFrame]): list of dataframes to be preprocessed.
            columns (list): Columns with unwanted floats.

        Returns:
            List[pd.DataFrame]: Processed list of dataframes.
        """

        for i, data in enumerate(data_list):
            for column in columns:
                if column not in data.columns:
                    continue
                data_list[i][column] = round(data[column])

        return data_list
