import glob
from pathlib import Path
from typing import Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.anomaly_predictor.data_prep.feature_engineering import FeatureEngineer
from src.anomaly_predictor.utils import get_asset_name


class DataSplitter:
    """Splits assets into train, val, test sets."""

    def __init__(self):
        pass

    def split_files(
        self,
        asset_dir: Path,
        test_size: float,
        random_state: int = 86421,
    ) -> Tuple[dict, Set[str]]:
        """Given a directory of asset csv files, splits assets into train, val,
        test sets and returns a dictionary with values being a list of assets'
        respective filepaths.

        Args:
            asset_dir (Path): directory containing csv files of assets.
            test_size (float): proportion of assets to be used as test data.
            random_state (int, optional): seed used for sklearn's train_test_split
                function. Defaults to 86421.

        Returns:
            A tuple (dict, Set[str]) where dict has keys 'train', 'val', 'test' and
            value being list of filepaths and set contains the unique values of
            Applications seen when splitting files.
        """
        asset_df = self.read_asset_dir(asset_dir)

        full_train, test = self._stratified_split_assets(
            asset_df, test_size, min_rows=3, random_state=random_state
        )
        train, val = self._stratified_split_assets(
            full_train, test_size / (1 - test_size), random_state=random_state
        )

        paths = {"train": train, "val": val, "test": test}
        destination_paths = {key: df["Filepath"].tolist() for key, df in paths.items()}

        return destination_paths, set(asset_df["Application"])

    @staticmethod
    def read_asset_dir(asset_dir: Path) -> pd.DataFrame:
        """Reads directory of asset csv files.

        Args:
            asset_dir (Path): directory path of csv files after processing with
                ingest_data module.

        Returns:
            pd.DataFrame: a dataframe with columns 'Filepath' and 'Application'
                indicating filepaths and application type e.g. Fan, Blower.
        """

        files = glob.glob(str(Path(asset_dir / "*.csv")), recursive=False)
        return pd.DataFrame(
            {
                "Filepath": files,
                "Application": [
                    FeatureEngineer.extract_application(Path(fp)) for fp in files
                ],
            }
        )

    @staticmethod
    def _stratified_split_assets(
        data: pd.DataFrame,
        test_size: float,
        min_rows: int = 2,
        random_state: int = 86421,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits assets into train and test sets in a stratified fashion

        Args:
            data (pd.DataFrame): dataframe to be split. It should contain column
                'Application' as that will be used for stratification.
            test_size (float): proportion of assets to be used as test data.
            min_row (int, optional): Assets with counts below min_row would not
                be considered in train test split and would directly be included
                in the train set. Defaults to 2.
            random_state (int, optional): seed used for sklearn's train_test_split
                function. Defaults to 86421.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: dataframes containing the original
                filepaths of assets split into train and test set respectively.
        """
        app_counts = data["Application"].value_counts()
        low_count_assets = app_counts[app_counts < min_rows].index
        low_count_df = data[data["Application"].isin(low_count_assets)]
        norm_count_df = data[~data.index.isin(low_count_df.index)]

        train, test = train_test_split(
            norm_count_df,
            test_size=test_size,
            random_state=random_state,
            stratify=norm_count_df["Application"],
        )
        train = pd.concat([train, low_count_df])
        return train, test

    def split_by_time(
        self,
        asset_dir: Path,
        test_size: float,
        time_col: str = "MEASUREMENT_TAKEN_ON(UTC)",
    ) -> Tuple[dict, Set[str]]:
        """Given a directory of asset csv files, splits all assets into train, val,
        test portion where train data is chronologically ordered before val data
        and val data before test data. Returns a dictionary with values being a
        list of assets' respective filepaths

        Args:
            asset_dir (Path): directory containing csv files of assets
            test_size (float): proportion of the asset data to be used as test data
            time_col (str): Name of column containing timestamp of data. Used for
                data indexing.

        Returns:
            A tuple (dict, Set[str]) where dict has keys 'train', 'val', 'test' and
            value being list of filepaths and set contains the unique values of
            Applications seen when splitting files
        """
        asset_dir = Path(asset_dir)
        asset_df = self.read_asset_dir(asset_dir)
        files = list(asset_dir.glob("*.csv"))
        partitions = ("train", "val", "test")

        # make directories to store split files
        for key in partitions:
            Path(asset_dir / key).mkdir(parents=True, exist_ok=True)

        for file in files:
            data = pd.read_csv(
                file,
                parse_dates=[time_col],
                infer_datetime_format=True,
                index_col=0,
            ).sort_index()

            full_train, test = train_test_split(
                data,
                test_size=test_size,
                shuffle=False,
            )

            train, val = train_test_split(
                full_train,
                test_size=test_size / (1 - test_size),
                shuffle=False,
            )

            for partition, dataframe in zip(partitions, (train, val, test)):
                start = str(dataframe.index.min().date()).replace("-", "")
                end = str(dataframe.index.max().date()).replace("-", "")
                asset_name = get_asset_name(file)
                output_dir = Path(asset_dir / partition)
                output_path = Path(output_dir / f"{asset_name}_{start}-{end}.csv")
                dataframe.to_csv(output_path)

        destination_paths = {
            partition: list(map(str, Path(asset_dir / partition).glob("*.csv")))
            for partition in partitions
        }

        return destination_paths, set(asset_df["Application"])
