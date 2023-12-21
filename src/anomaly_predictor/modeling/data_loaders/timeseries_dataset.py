import logging
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch
from joblib import load
from sklearn.preprocessing import RobustScaler

from src.anomaly_predictor.data_prep.feature_engineering import FeatureEngineer
from src.anomaly_predictor.modeling.utils import curve_shift, split_training_timeseries
from src.anomaly_predictor.utils import get_asset_name

logger = logging.getLogger(__name__)


class TimeSeriesDataset:
    """Creates a custom dataset for a given Dataframe."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_list: list,
        scaler: RobustScaler,
        lookback_period: int = 100,
        lookahead_period: int = 24,
        statistical_window: int = 48,
    ):
        """Initializes parameters of TimeSeriesDataset and creates rolling
        statistical features if statistical_window > 0.

        Args:
            data (pd.DataFrame): DataFrame containing features to be windowed.
            feature_list (list): List of features to be scaled.
            scaler (RobustScaler): Scaler to be used.
            lookback_period (int, optional): Size of window. Defaults to 100.
            lookahead_period (int, optional): Number of periods for data to be
                shifted by. Defaults to 24.
            statistical_window (int): window size for rolling statistical features.
                Defaults to 48.
        """
        # Init dependencies
        self.scaler = scaler
        self.lookback_period = lookback_period
        self.lookahead_period = lookahead_period
        self.df_timeseries = data
        self.df_timeseries[feature_list] = self.scaler.transform(
            self.df_timeseries[feature_list]
        )
        if statistical_window > 0:
            self.df_timeseries = FeatureEngineer().create_statistical_features(
                self.df_timeseries, statistical_window, feature_list
            )

        # Store dataframe properties
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index: int) -> torch.Tensor:
        """Loads and returns a sample from the dataset at the given index.

        Args:
            index (int): Index of sample in a dataset.

        Returns:
            torch.Tensor: Tensor containing windowed feature data.
        """
        feature_data = self.df_timeseries[index : index + self.lookback_period]
        return torch.Tensor(feature_data.values)

    def __len__(self) -> int:
        """Returns the number of sliding windows that can be generated by the
        current dataframe.

        Returns:
            int: Number of sliding windows that can be generated.
        """
        return len(self.df_timeseries) - self.lookback_period + 1

    def __getshape__(self) -> Tuple[int, int, int]:
        """Returns the shape of feature data.

        Returns:
            Tuple[int, int, int]: Shape of feature data (number of windows,
                self.lookback_period, number of features).
        """
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self) -> int:
        """Returns the number of sliding windows that can be generated by the
        current dataframe.

        Returns:
            int: Number of sliding windows that can be generated.
        """
        return self.__len__()


def concat_datasets(
    list_filepath: list,
    feature_list: list,
    scaler: RobustScaler,
    return_mode: str = "train",
    lookback_period: int = 100,
    lookahead_period: int = 24,
    statistical_window: int = 48,
    label_col: str = "Anomaly",
    batch_size: int = 32,
    shuffle: bool = False,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    scaler_dict: dict = None,
    remove_non_operating: bool = None,
) -> Union[Tuple[torch.utils.data.DataLoader, list], Tuple[None, None]]:
    """Concatenates multiple TimeSeriesDataset objects to be passed into torch dataloader.

    Args:
        list_filepath (list): List of csv filepaths.
        feature_list (list): List of features to be scaled.
        scaler (RobustScaler): Scaler to be used.
        return_mode (str, optional): Expects values "train", "eval" and "inference".
            Defaults to "train".
        lookback_period (int, optional): Size of window. Defaults to 100.
        lookahead_period (int, optional): Number of periods for data to be shifted
            by. Defaults to 24.
        statistical_window (int): window size for rolling statistical features.
            Defaults to 48.
        label_col (str, optional): Column containing label data. Defaults to "Anomaly".
        batch_size (int, optional): Number of windows per batch to be loaded.
            Defaults to 32.
        shuffle (bool, optional): Whether to reshuffle data at every epoch.
            Defaults to False.
        drop_last (bool, optional): Whether to drop the last batch if size of last
            batch is smaller than specified batch_size. Defaults to False.
        pin_memory (bool, optional): If True, the data loader will copy Tensors
            into CUDA pinned memory before returning them. Defaults to False.
        num_workers (int, optional): How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Defaults to 0.
        scaler_dict (dict, optional): Dictionary of {asset_name:scaler_filepath}.
            Defaults to None.
        remove_non_operating (bool, optional): Whether to remove non-operating data,
            i.e. when feature Asset_Operating is 0. Defaults to None.

    Returns:
        Union[Tuple[torch.utils.data.DataLoader, list], Tuple[None, None]]: If data
            available is longer than lookback_period, returns tuple of Torch dataloader
            containing batched feature_data and a list. When return mode is "eval",
            list contains ground truth, else list is empty. When data available is shorter
            than lookback_period, a tuple of None is returned.

    """
    dataset_list = []
    label_list = []
    min_sequence_len = lookback_period + statistical_window
    for file in list_filepath:

        # instantiate scaler to individual scaler if possible, log when global scaler used
        asset_name = get_asset_name(file)
        if scaler_dict is not None and asset_name in scaler_dict:
            scaler = load(Path(scaler_dict[asset_name]))
        else:
            logger.info(f"Individual scaler not found for {file}. Global scaler used.")

        if return_mode == "train":
            data = pd.read_csv(
                file, parse_dates=True, infer_datetime_format=True, index_col=0
            )
            data = curve_shift(data, label_col, lookahead_period, False)
            data = data.drop(data[data[label_col] == 1].index)
            data = data.drop(label_col, axis=1)
            if remove_non_operating:
                data = data[data["Asset_Operating"] == 1]
            data_list = split_training_timeseries(data)
            for item in data_list:
                if len(item) >= min_sequence_len:
                    dataset_list.append(
                        TimeSeriesDataset(
                            item,
                            feature_list,
                            scaler,
                            lookback_period,
                            lookahead_period,
                            statistical_window,
                        )
                    )
                else:
                    logger.warning(
                        f"{Path(file)} of sequence length {len(item)} skipped as it is shorter than required minimum length of {min_sequence_len}"
                    )
        else:
            data = pd.read_csv(
                file, parse_dates=True, infer_datetime_format=True, index_col=0
            )
            if return_mode == "eval":
                # Anomalous datapoints to be kept for evaluation
                data = curve_shift(data, label_col, lookahead_period, False)
                label_list.extend(data[label_col][lookback_period - 1 :])
                data = data.drop(label_col, axis=1)
            if len(data) >= min_sequence_len:
                dataset_list.append(
                    TimeSeriesDataset(
                        data,
                        feature_list,
                        scaler,
                        lookback_period,
                        lookahead_period,
                        statistical_window,
                    )
                )
            else:
                logger.warning(
                    f"{Path(file)} of sequence length {len(data)} skipped as it is shorter than required minimum length of {min_sequence_len}"
                )
    if len(dataset_list) > 0:
        concat_datasets = torch.utils.data.ConcatDataset(dataset_list)
        return (
            torch.utils.data.DataLoader(
                concat_datasets,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                pin_memory=pin_memory,
                num_workers=num_workers,
            ),
            label_list,
        )
    else:
        return (None, None)