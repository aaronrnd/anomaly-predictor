# pylint: skip-file
import os
from glob import glob
from math import floor
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch
from joblib import dump, load

from src.anomaly_predictor.modeling.data_loaders.data_loader import DataLoader
from src.anomaly_predictor.modeling.data_loaders.timeseries_dataset import (
    TimeSeriesDataset,
    concat_datasets,
)
from src.anomaly_predictor.modeling.utils import curve_shift, split_training_timeseries

features = [
    "Bearing Condition",
    "Motor Supply Frequency",
]
time_col = "MEASUREMENT_TAKEN_ON(UTC)"
label_column = "Anomaly"

lookahead_period = 24
batch_size = 32
lookback_period = 100
statistical_window = 2
remove_non_operating = True
lstm_loader = DataLoader(
    "pytorch",
    batch_size,
    lookback_period,
    statistical_window,
    remove_non_operating=remove_non_operating,
)


@pytest.fixture(scope="session")
def process_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("process"))


@pytest.fixture(scope="session")
def timestamp_dir(tmpdir_factory, process_dir):
    timestamp_dir = Path(os.sep.join([process_dir, "20220222_000000"]))
    timestamp_dir.mkdir()
    return str(timestamp_dir)


@pytest.fixture(scope="session")
def dummy_csv(tmpdir_factory, process_dir, timestamp_dir):
    np.random.seed(8642)

    date_range = pd.date_range("2021-07-01", periods=500, freq="H")
    test_col = [
        "Bearing Condition",
        "Motor Supply Frequency",
        "Anomaly",
    ]

    for partition in ["train", "val", "inference"]:
        directory = Path(os.sep.join([timestamp_dir, partition]))
        directory.mkdir()
        for i in range(3):
            if partition == "inference":
                test_data = pd.DataFrame(
                    np.random.randint(100, size=(500, 2)),
                    columns=test_col[:2],
                )
            else:
                test_data = pd.DataFrame(
                    np.random.randint(100, size=(500, 3)),
                    columns=test_col,
                )
            test_data["Asset_Operating"] = 1
            test_data.set_index(date_range, inplace=True)
            test_data.index.name = time_col
            if partition != "inference":
                test_data[label_column] = 0
                test_data.loc[130:140, label_column] = 1
                test_data.loc[180:184, label_column] = 1
            test_data.to_csv(str(directory) + "/" + str(i) + ".csv")

    return str(timestamp_dir)


@pytest.fixture(scope="session")
def models_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("models"))


@pytest.fixture(scope="session")
def forecast_dir(tmpdir_factory, models_dir):
    forecast_dir = Path(os.sep.join([models_dir, "forecast"]))
    forecast_dir.mkdir()
    return str(forecast_dir)


@pytest.fixture(scope="session")
def forecast_timestamp_dir(tmpdir_factory, forecast_dir):
    forecast_timestamp_dir = Path(os.sep.join([forecast_dir, "20220222_000000"]))
    forecast_timestamp_dir.mkdir()
    return str(forecast_timestamp_dir)


@pytest.fixture(scope="session")
def create_scaler(dummy_csv, forecast_timestamp_dir):
    train_filepaths = glob(str(Path(Path(dummy_csv) / "train" / "*.csv")))
    loader = DataLoader()
    loader._concat_data_fit_scaler(Path(Path(dummy_csv) / "train"), features)
    dump(loader.scaler, Path(Path(forecast_timestamp_dir) / "scaler.joblib"))
    return loader.scaler, Path(Path(forecast_timestamp_dir) / "scaler.joblib")


def test_custom_dataset(dummy_csv, create_scaler):
    train_filepaths = glob(str(Path(Path(dummy_csv) / "train" / "*.csv")))
    for file in train_filepaths:
        test_csv = pd.read_csv(
            file, parse_dates=True, infer_datetime_format=True, index_col=0
        )
        test_csv = curve_shift(test_csv, label_column, lookahead_period, True)
        test_csv = test_csv.drop(test_csv[test_csv[label_column] == 1].index)
        test_csv = test_csv.drop(test_csv[test_csv["Asset_Operating"] != 1].index)
        data_list = split_training_timeseries(test_csv)
        scaler, _ = create_scaler
        for data in data_list:
            if len(data) >= lookback_period:
                data = data.drop(label_column, axis=1)
                custom_dataset = TimeSeriesDataset(
                    data,
                    features,
                    scaler,
                    lookback_period=lookback_period,
                    lookahead_period=lookahead_period,
                    statistical_window=2,
                )
                print(custom_dataset.__getshape__)
                # Check that the # of total complete windows are correct & that output is a TimeSeriesDataset
                assert custom_dataset.__len__() == (
                    (len(data) - lookback_period + 1) - statistical_window + 1
                )
                assert isinstance(custom_dataset, TimeSeriesDataset)

                # # Check that last possible window is a full window set
                x = custom_dataset[custom_dataset.__len__() - 1]
                assert isinstance(x, torch.Tensor)

                assert list(x.shape) == [
                    lookback_period,
                    (len(test_csv.columns) - 1 + len(features) * 2),
                ]


def test_train_concat_datasets(dummy_csv, create_scaler):
    train_filepaths = glob(str(Path(Path(dummy_csv) / "train" / "*.csv")))
    total_possible_windows = []
    for file in train_filepaths:
        test_csv = pd.read_csv(
            file, parse_dates=True, infer_datetime_format=True, index_col=0
        )
        test_csv = curve_shift(test_csv, label_column, lookahead_period, True)
        test_csv = test_csv.drop(test_csv[test_csv[label_column] == 1].index)
        test_csv = test_csv.drop(test_csv[test_csv["Asset_Operating"] != 1].index)
        data_list = split_training_timeseries(test_csv)
        for data in data_list:
            if len(data) >= lookback_period:
                total_possible_windows.append(len(data) - lookback_period + 1)

    total_possible_batches = floor(sum(total_possible_windows) / batch_size)
    scaler, _ = create_scaler
    train_dataloader, _ = concat_datasets(
        train_filepaths,
        features,
        scaler,
        "train",
        lookback_period,
        lookahead_period,
        2,
        label_column,
        batch_size,
        shuffle=False,
        drop_last=True,
    )
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)

    # Check output types & shapes are correct
    iterate_loader = iter(train_dataloader)
    x = next(iterate_loader)
    assert isinstance(x, torch.Tensor)

    if (sum(total_possible_windows) % batch_size) != 0:
        # Check that last batch is full batch
        for i in range(total_possible_batches - 1):
            x = next(iterate_loader)

        assert isinstance(x, torch.Tensor)

        assert list(x.shape) == [
            batch_size,
            lookback_period,
            (len(test_csv.columns) - 1 + len(features) * 2),
        ]


def test_eval_concat_datasets(dummy_csv, create_scaler):
    val_filepaths = glob(str(Path(Path(dummy_csv) / "val" / "*.csv")))
    total_possible_windows = []
    for file in val_filepaths:
        test_csv = pd.read_csv(
            file, parse_dates=True, infer_datetime_format=True, index_col=0
        )
        test_csv = curve_shift(test_csv, label_column, lookahead_period, False)
        data_list = split_training_timeseries(test_csv)
        for data in data_list:
            if len(data) >= lookback_period:
                total_possible_windows.append(len(data) - lookback_period + 1)
    # total_possible_batches = floor((total_possible_windows) / batch_size)
    total_possible_batches = [floor(x / batch_size) for x in total_possible_windows]

    scaler, _ = create_scaler
    for file, i in zip(val_filepaths, range(len(total_possible_windows))):
        eval_dataloader, labels = concat_datasets(
            [file],
            features,
            scaler,
            "eval",
            lookback_period,
            lookahead_period,
            2,
            label_column,
            batch_size,
            shuffle=False,
            drop_last=False,
        )
        assert total_possible_windows[i] == len(labels)
        assert isinstance(labels, list)

        assert isinstance(eval_dataloader, torch.utils.data.DataLoader)

        # Check output types & shapes are correct
        iterate_loader = iter(eval_dataloader)
        x = next(iterate_loader)

        assert isinstance(x, torch.Tensor)
        assert list(x.shape) == [
            batch_size,
            lookback_period,
            (len(test_csv.columns) - 1 + len(features) * 2),
        ]

        if (total_possible_windows[i] % batch_size) != 0:
            # Check that last possible batch is not a full batch
            for i in range(total_possible_batches[i]):
                x = next(iterate_loader)

            assert isinstance(x, torch.Tensor)
            assert list(x.shape) != [
                batch_size,
                lookback_period,
                (len(test_csv.columns) - 1 + len(features) * 2),
            ]


def test_inference_concat_datasets(dummy_csv, create_scaler):
    inference_filepaths = glob(str(Path(Path(dummy_csv) / "inference" / "*.csv")))
    total_possible_windows = []
    for file in inference_filepaths:
        test_csv = pd.read_csv(
            file, parse_dates=True, infer_datetime_format=True, index_col=0
        )
        if len(test_csv) >= lookback_period:
            total_possible_windows.append(len(test_csv) - lookback_period)
    total_possible_batches = floor(sum(total_possible_windows) / batch_size)

    scaler, _ = create_scaler
    inference_dataloader, _ = concat_datasets(
        inference_filepaths,
        features,
        scaler,
        "inference",
        lookback_period,
        None,
        statistical_window,
        None,
        batch_size,
        shuffle=False,
        drop_last=True,
    )
    assert isinstance(inference_dataloader, torch.utils.data.DataLoader)

    # Check output types & shapes are correct
    iterate_loader = iter(inference_dataloader)
    x = next(iterate_loader)

    assert isinstance(x, torch.Tensor)
    assert list(x.shape) == [
        batch_size,
        lookback_period,
        (len(test_csv.columns) + len(features) * 2),
    ]

    if (sum(total_possible_windows) % batch_size) != 0:
        # Check that end of 1 dataset continues to next dataset
        for i in range(total_possible_batches - 1):
            x = next(iterate_loader)

        assert isinstance(x, torch.Tensor)

        assert list(x.shape) == [
            batch_size,
            lookback_period,
            (len(test_csv.columns) + len(features) * 2),
        ]
