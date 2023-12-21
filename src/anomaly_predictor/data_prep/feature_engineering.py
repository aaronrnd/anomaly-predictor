import re
from pathlib import Path

import pandas as pd
from joblib import load
from pandas import DataFrame, Series
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

from src.anomaly_predictor.modeling.utils import (
    floor_predictions_to_zero_for_nonoperating,
)
from src.anomaly_predictor.utils import export_artifact

pd.options.mode.chained_assignment = None

class FeatureEngineer:
    def __init__(self):
        self.encoder = None

    def fit_encoder(self, application_set: set) -> None:
        """Instantiates self.encoder as OneHotEncoder and fits encoder on application_set.

        Args:
            application_set (set): set of application type
        """
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.encoder.fit(pd.DataFrame(list(application_set), columns=["Application"]))

    def save_encoder(self, path: Path, file_name: str) -> None:
        """Export encoder and save them to path.

        Args:
            path (Path): Directory to place artifact details.
            file_name (str): File name of stored artifact.
        """
        export_artifact(self.encoder, path, file_name)

    def load_encoder(self, path: Path) -> None:
        """Instantiates encoder from given path.

        Args:
            path (Path): path where fitted encoder is stored.
        """
        self.encoder = load(path)

    def engineer_features(
        self,
        data: DataFrame,
        file_name: Path,
        motor_supply_freq: str = "Motor Supply Frequency",
        min_data_points:int=24,
    ) -> DataFrame:
        """Performs the following Feature Engineering:
        1. Creates "Variable_Speed" column based on  Motor Supply Frequency col.
        2. Creates "Asset_Operating" column based on Motor Supply Frequency col.
        3. Creates "Application_<application_type>" column based on file name of dataset.
        4. Floor ground truth to 0 when asset is not operating.

        Args:
            data (DataFrame): dataset.
            filepath (Path): filepath of dataset to extract application type.
            motor_supply_freq (str): column name for motor supply frequency.
            min_data_points (int): minimum data points for determining variable speed boolean.
        Returns:
            data (DataFrame): Data with newly created columns.
        """
        data["Variable_speed"] = self._determine_variable_speed_bool(
            data[motor_supply_freq], min_data_points=min_data_points
        )
        data["Asset_Operating"] = data[motor_supply_freq].apply(
            lambda x: 1 if x != 0 else 0
        )
        data = self.create_one_hot_encode_application(data, file_name)
        if "Anomaly" in data.columns:
            data["Anomaly"] = floor_predictions_to_zero_for_nonoperating(data, data["Anomaly"])
        return data

    @staticmethod
    def _determine_variable_speed_bool(
        speed_array: Series, min_data_points: int = 24
    ) -> int:
        """Infers motor speed type from array.

        If array contains 2 or lesser unique value, 0 will be returned and
        if array contains more than 2 unique value, 1 will be returned.

        Args:
            speed_array (Series): Array containing the speed values.
            min_data_points(int): Number of minimum data points to determine asset
                as non_operating motor.

        Returns:
            variable_bool (int): Boolean Value of variable Speed.
        """
        unique_values = speed_array.value_counts() > min_data_points
        unique_count = unique_values.sum()
        if unique_count <= 2:
            variable_bool = 0
        else:
            variable_bool = 1
        return variable_bool

    def create_one_hot_encode_application(
        self, data: DataFrame, filepath: str
    ) -> DataFrame:
        """Creates one hot encoded feature columns of application type.

        Args:
            data (DataFrame): dataset.
            filepath (str): filepath of data.

        Returns:
            DataFrame: data with one hot encoded application.
        """

        data["Application"] = self.extract_application(filepath)
        encoded_result = self.encoder.transform(
            pd.DataFrame(data["Application"])
        ).toarray()
        column_names = self.encoder.get_feature_names_out(["Application"])
        encoded_df = pd.DataFrame(
            encoded_result, columns=column_names, index=data.index
        )
        data.drop("Application", axis=1, inplace=True)
        return pd.concat([data, encoded_df], axis=1)

    @staticmethod
    def extract_application(filepath: Path) -> str:
        """Infers application type from a filepath.

        The Application Name should be at the start of the file name for this
        function to infer correctly. For example 'Fan3001_20210701-20211122.csv"
        would work. 'Cooling Fan -Fan3001_20210701-20211122.csv" would not work.

        Args:
            filepath (Path): filepath containing asset name at start of filename.

        Returns:
            str: Application type.
        """
        return re.findall(re.compile("[a-zA-Z]+"), filepath.stem)[0]

    @staticmethod
    def extract_abs_median_diff(
        data: DataFrame, statistical_window: int, features_to_extract: list
    ) -> DataFrame:
        """Extract rolling absolute median difference with dataframe median
        according to statistical_window. Currently not in use.

        Args:
            data (DataFrame): original Dataframe.
            statistical_window (int): window size for rolling median.
            features_to_extract (list): List of features to extract absolute median difference.

        Returns:
            DataFrame: dataframe with newly extracted absolute median difference.
        """
        new_column_names = [
            name + "_rolling_median_diff" for name in features_to_extract
        ]
        selected_data = data[features_to_extract]
        diff_df = abs(
            selected_data.median() - selected_data.rolling(statistical_window).median()
        )
        diff_df.columns = new_column_names
        combined_df = pd.merge(data, diff_df, on=data.index, left_index=True).drop(
            ["key_0"], axis=1
        )
        return combined_df

    @staticmethod
    def extract_abs_iqr_diff(
        data: DataFrame, statistical_window: int, features_to_extract: list
    ) -> DataFrame:
        """Extract rolling absolute IQR difference with dataframe IQR according to 
        statistical_window. Currently not in use.

        Args:
            data (DataFrame): original Dataframe.
            statistical_window (int): window size for rolling IQR.
            features_to_extract (list): List of features to extract absolute IQR difference.

        Returns:
            DataFrame: dataframe with newly extracted absolute IQR difference.
        """
        new_column_names = [name + "_rolling_iqr_diff" for name in features_to_extract]
        selected_data = data[features_to_extract]
        iqr = lambda x: stats.iqr(x, interpolation="midpoint")
        diff_df = abs(
            selected_data.apply(iqr)
            - selected_data.rolling(statistical_window).apply(iqr)
        )
        diff_df.columns = new_column_names
        combined_df = pd.merge(data, diff_df, on=data.index, left_index=True).drop(
            ["key_0"], axis=1
        )
        return combined_df

    def create_statistical_features(self,
        data: DataFrame,
        statistical_window: int,
        features_to_extract: list) -> DataFrame:
        """Creates statistical features for the features listed in features_to_extract.
        Currently not in use.

        Args:
            data (DataFrame): Dataframe to extract statistical_window on.
            statistical_window (int): Size of rolling window.
            features_to_extract (list): A list of original features to extract statistical information
                from.

        Returns:
            DataFrame: Dataframe containing statistical features.
        """
        data = self.extract_abs_median_diff(data,statistical_window,features_to_extract)
        data = self.extract_abs_iqr_diff(data,statistical_window,features_to_extract)
        return data[statistical_window - 1:]
