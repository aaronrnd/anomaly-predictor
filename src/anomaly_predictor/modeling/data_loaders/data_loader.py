from glob import glob
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.preprocessing import RobustScaler

from src.anomaly_predictor.modeling.data_loaders.timeseries_dataset import (
    concat_datasets,
)
from src.anomaly_predictor.modeling.utils import curve_shift


class DataLoader:
    """Loads data for train, eval and inference purposes."""

    def __init__(
        self,
        return_mode: str = "sklearn",
        batch_size: int = None,
        lookback_period: int = None,
        lookahead_period: int = None,
        statistical_window: int = None,
        pin_memory: bool = None,
        num_workers: int = None,
        args: dict = None,
        scaler_dir: Path = None,
        remove_non_operating: bool = None,
    ):
        """Initializes parameters of DataLoader and creates a dictionary mapping
        asset name to scaler filepath if scaler_dir is not None.

        Args:
            return_mode (str, optional): Loads feature data as torch DataLoader
                if return_mode is "pytorch" else loads as DataFrame if return_mode
                is "sklearn". Defaults to "sklearn".
            batch_size (int, optional): Number of windows per batch to be loaded.
                Defaults to None.
            lookback_period (int, optional): Size of lookback window. Defaults to None.
            lookahead_period (int, optional): Size of lookahead window, equivalently
                number of periods for data to be shifted by. Defaults to None.
            statistical_window (int, optional): window size for rolling statistical
                features. Defaults to None.
            pin_memory (bool, optional): If True, the data loader will copy Tensors
            into CUDA pinned memory before returning them. Defaults to None.
            num_workers (int, optional): How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Defaults to None.
            args (dict, optional): parameters for initializing Robust Scaler. Initializes
                Robust scaler with default settings if None. Defaults to None.
            scaler_dir (Path, optional): If path to directory containing scalers are
                provided. Creates an attribute scaler_dict, a dictionary that maps
                asset name to filepath of its scaler. Defaults to None.
            remove_non_operating (bool, optional): Whether to remove non-operating data,
                i.e. when feature Asset_Operating is 0. Defaults to None.
        """
        self.return_mode = return_mode
        self.batch_size = batch_size
        self.lookback_period = lookback_period
        self.lookahead_period = lookahead_period
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.statistical_window = statistical_window
        self.remove_non_operating = remove_non_operating
        if args:
            self.scaler = RobustScaler(**args)
        else:
            self.scaler = RobustScaler()

        if scaler_dir is not None:
            self.scaler_dict = {
                Path(fp).stem: fp for fp in Path(scaler_dir).glob("*.joblib")
            }
        else:
            self.scaler_dict = None

    def load_train_data(
        self,
        train_dir: Path,
        feature_list: list,
        label_column: str = "Anomaly",
        time_col: str = "MEASUREMENT_TAKEN_ON(UTC)",
        fit_scaler: bool = True,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> Union[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[torch.utils.data.DataLoader, None],
    ]:
        """Performs further data preprocessing on processed training data.
        1. Read each csv fle in given directory and perform curve_shifting if required.
        2. Concatenate all data into a single DataFrame or torch Dataloader.
        3. Fit scaler on data and transform data.
        4. Split data into features and labels.

        Args:
            train_dir (Path): Path to processed train directory.
            feature_list (list): List of features to be scaled.
            label_column (str, optional): Column containing Anomaly labels.
                Defaults to "Anomaly".
            time_col (str, optional): Column containing date time index. Defaults
                to "MEASUREMENT_TAKEN_ON(UTC)".
            fit_scaler (bool, optional): Indicator for whether scaler will be fit.
                This is only relevant when self.return_mode is pytorch because we
                might use load_train_data on val partition for calculating val loss
                during model training. Defaults to True.
            shuffle (bool, optional): To reshuffle data at every epoch. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if size
                of last batch is smaller than specified batch_size. Defaults to False.

        Returns:
            Union[
                Tuple[pd.DataFrame, pd.Series],
                Tuple[torch.utils.data.DataLoader, list]
            ]: Tuple of feature_data, label_data.
        """
        if self.return_mode == "sklearn":
            main_data = self._concat_data_fit_scaler(
                train_dir, feature_list, label_column, self.lookahead_period
            )
            main_data[feature_list] = self.scaler.transform(main_data[feature_list])
            feature_data = main_data.drop(label_column, axis=1)
            label_data = main_data[label_column]
            return feature_data, label_data

        if self.return_mode == "pytorch":
            if fit_scaler:
                self._concat_data_fit_scaler(
                    train_dir, feature_list, label_column, self.lookahead_period
                )
            files = glob(str(Path(Path(train_dir) / "*.csv")))
            train_dataloader, _ = concat_datasets(
                files,
                feature_list,
                self.scaler,
                "train",
                self.lookback_period,
                self.lookahead_period,
                self.statistical_window,
                label_column,
                self.batch_size,
                shuffle,
                drop_last,
                self.pin_memory,
                self.num_workers,
                self.scaler_dict,
                self.remove_non_operating,
            )
            return train_dataloader, None

    def load_eval_data(
        self,
        test_file: Path,
        feature_list: list,
        label_column: str = "Anomaly",
        time_col: str = "MEASUREMENT_TAKEN_ON(UTC)",
        drop_last: bool = False,
    ) -> Union[
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame],
        Tuple[torch.utils.data.DataLoader, np.ndarray, pd.DataFrame],
    ]:
        """Performs further data preprocessing on processed eval data.
        1. Read csv file specified in test_file and performs curve_shifting if required.
        2. Use fitted scaler to transform data.
        3. Split data into features and labels.

        Args:
            test_file (Path): File path to processed csv file.
            feature_list (list): List of features to be scaled.
            label_column (str, optional): Column containing Anomaly labels.
                Defaults to "Anomaly".
            time_col (str, optional): Column containing date time index. Defaults
                to "MEASUREMENT_TAKEN_ON(UTC)".
            drop_last (bool, optional): Whether to drop the last batch if size
                of last batch is smaller than specified batch_size. Defaults to False.

        Returns:
            Union[\
                Tuple[pd.DataFrame, pd.Series, pd.DataFrame], \
                Tuple[torch.utils.data.DataLoader, np.ndarray, pd.DataFrame]\
            ]: Tuple of feature_data as dataframe or torch Dataloader, label_data \
                and original data.
        """
        data = pd.read_csv(
            test_file,
            parse_dates=[time_col],
            infer_datetime_format=True,
            index_col=0,
        )
        data = curve_shift(
            data,
            label_column,
            self.lookahead_period,
            False,
        )
        if self.return_mode == "sklearn":
            data[feature_list] = self.scaler.transform(data[feature_list])
            feature_data = data.drop(label_column, axis=1)
            label_data = data[label_column]

            return feature_data, label_data, data

        if self.return_mode == "pytorch":
            eval_dataloader, label_data = concat_datasets(
                [test_file],
                feature_list,
                self.scaler,
                "eval",
                self.lookback_period,
                self.lookahead_period,
                self.statistical_window,
                label_column,
                self.batch_size,
                False,
                drop_last,
                self.pin_memory,
                self.num_workers,
                self.scaler_dict,
            )
            return eval_dataloader, np.array(label_data), data

    def load_inference_data(
        self,
        inference_file: Path,
        feature_list: list,
        time_col: str = "MEASUREMENT_TAKEN_ON(UTC)",
        drop_last: bool = False,
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[torch.utils.data.DataLoader, pd.DataFrame],
    ]:
        """Performs further data preprocessing on inference data.
        1. Read csv file specified in inference_file.
        2. Use loaded scaler, which has already been fitted, to transform data.

        Args:
            inference_file (Path): File path to processed csv file.
            feature_list (list): List of features to be scaled.
            time_col (str, optional): Column containing date time index.
                Defaults to "MEASUREMENT_TAKEN_ON(UTC)".
            drop_last (bool, optional): Whether to drop the last batch if size
                of last batch is smaller than specified batch_size. Defaults to False.

        Returns:
            Tuple[Union[pd.DataFrame, torch.utils.data.DataLoader], pd.DataFrame]:
                Dataframe or torch DataLoader containing inference data, along
                with original data.
        """
        data = pd.read_csv(
            inference_file,
            parse_dates=[time_col],
            infer_datetime_format=True,
            index_col=0,
        )

        if self.return_mode == "sklearn":

            feature_data = data.copy()
            feature_data[feature_list] = self.scaler.transform(
                feature_data[feature_list]
            )
            return feature_data, data

        if self.return_mode == "pytorch":
            inference_dataloader, _ = concat_datasets(
                [inference_file],
                feature_list,
                self.scaler,
                "inference",
                self.lookback_period,
                self.lookahead_period,
                self.statistical_window,
                None,
                self.batch_size,
                False,
                drop_last,
                self.pin_memory,
                self.num_workers,
                self.scaler_dict,
            )
            return inference_dataloader, data

    @staticmethod
    def _concat_data(
        file_dir: Path,
        label_column: str = "Anomaly",
        lookahead_period: int = None,
        drop_anomalous: bool = False,
        time_col: str = "MEASUREMENT_TAKEN_ON(UTC)",
    ) -> pd.DataFrame:
        """Iterates through csv files in a given directory and performs the
        following:
        1. curve_shifting for data if required.
        2. concatenates all data into a single DataFrame.

        Args:
            file_dir (Path): Path to folder containing csv files
            label_column (str, optional): Column containing Anomaly labels.
                Defaults to "Anomaly".
            lookahead_period (int, optional): Number of periods for data to be
                shifted by. Defaults to None.
            drop_anomalous (bool, optional): To drop rows with anomalous labels.
                Defaults to False.
            time_col (str, optional): Column containing date time index.
                Defaults to "MEASUREMENT_TAKEN_ON(UTC)".

        Returns:
            pd.DataFrame: Concatenated DataFrame.
        """
        files = glob(str(Path(Path(file_dir) / "*.csv")))
        main_data = None
        for file in files:
            data = pd.read_csv(
                file,
                parse_dates=[time_col],
                infer_datetime_format=True,
                index_col=0,
            )
            if lookahead_period:
                data = curve_shift(data, label_column, lookahead_period, drop_anomalous)
            # apply curveshift then append.
            if main_data is not None:
                main_data = main_data.append(data)
            else:
                main_data = data.copy()
        return main_data

    def _concat_data_fit_scaler(
        self,
        file_dir: Path,
        feature_list: list,
        label_column: str = "Anomaly",
        lookahead_period: int = None,
        drop_anomalous: bool = False,
    ) -> pd.DataFrame:
        """Scales all csv files in a given path
        1. Concatenate all data into a single DataFrame.
        2. Fit scaler on concatenated DataFrame.

        Args:
            file_dir (Path): Path to folder containing csv files.
            feature_list (list): List of features to be scaled.
            label_column (str, optional): Column containing Anomaly labels.
                Defaults to "Anomaly".
            lookahead_period (int, optional): Number of periods for data to be
                shifted by. Defaults to None.
            drop_anomalous (bool, optional): To drop rows with anomalous labels.
                Defaults to False.

        Returns:
            pd.DataFrame: Concatenated and scaled DataFrame.
        """
        main_data = self._concat_data(
            file_dir, label_column, lookahead_period, drop_anomalous=drop_anomalous
        )

        # Fit scaler & transform main_data features
        normal_data = main_data[main_data[label_column] == 0][feature_list]
        self.scaler.fit(normal_data)

        return main_data

    def load_scaler(self, path: Path) -> None:
        """Instantiates scaler from given path.

        Args:
            path (Path): path where fitted scaler is stored.
        """
        self.scaler = load(Path(path))
