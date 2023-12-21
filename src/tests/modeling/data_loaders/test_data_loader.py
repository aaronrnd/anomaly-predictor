# pylint: skip-file
import os
from glob import glob
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from joblib import dump, load

from src.anomaly_predictor.modeling.data_loaders.data_loader import DataLoader
from src.anomaly_predictor.modeling.utils import curve_shift, split_training_timeseries

features = [
    "Bearing Condition",
    "Motor Supply Frequency",
]

label_column = "Anomaly"

time_col = "MEASUREMENT_TAKEN_ON(UTC)"

lookahead_period = 24
batch_size = 32
lookback_period = 100
statistical_window = 2
pin_memory = False
num_workers = 0
remove_non_operating = True


isoforest_loader = DataLoader(
    lookahead_period=lookahead_period,
    statistical_window=statistical_window,
    remove_non_operating=remove_non_operating,
)
lstm_loader = DataLoader(
    "pytorch",
    batch_size,
    lookback_period,
    lookahead_period,
    statistical_window,
    pin_memory,
    num_workers,
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

    for partition in ["train", "val", "test"]:
        directory = Path(os.sep.join([timestamp_dir, partition]))
        directory.mkdir()
        for i in range(3):
            if partition == "test":
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
            if partition != "test":
                test_data[label_column] = 0
                test_data.iloc[130:140, -1] = 1
                test_data.iloc[180:184, -1] = 1
            test_data.to_csv(str(directory) + "/" + str(i) + ".csv")

    return str(timestamp_dir)


@pytest.fixture(scope="session")
def models_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("models"))


@pytest.fixture(scope="session")
def detection_dir(tmpdir_factory, models_dir):
    detection_dir = Path(os.sep.join([models_dir, "detection"]))
    detection_dir.mkdir()
    return str(detection_dir)


@pytest.fixture(scope="session")
def detection_timestamp_dir(tmpdir_factory, detection_dir):
    detection_timestamp_dir = Path(os.sep.join([detection_dir, "20220222_000000"]))
    detection_timestamp_dir.mkdir()
    return detection_timestamp_dir


@pytest.fixture(scope="session")
def dummy_concat_csv(tmpdir_factory, process_dir, timestamp_dir):
    np.random.seed(8642)

    date_range = pd.date_range("2021-07-01", periods=5, freq="H")
    test_col = [
        "Bearing Condition",
        "Motor Supply Frequency",
        "Anomaly",
    ]

    directory = Path(os.sep.join([timestamp_dir, "dummy_concat"]))
    directory.mkdir()
    for i in range(3):
        test_data = pd.DataFrame(
            np.random.randint(5, size=(5, 3)),
            columns=test_col,
        )
        test_data["Asset_Operating"] = 1
        test_data.set_index(date_range, inplace=True)
        test_data.index.name = "MEASUREMENT_TAKEN_ON(UTC)"
        test_data["Anomaly"] = 0
        test_data.iloc[4:, -2] = 1
        test_data.to_csv(str(directory) + "/" + str(i) + ".csv")

    return str(timestamp_dir)


def test_load_train_data(dummy_csv, detection_timestamp_dir):
    train_path = Path(Path(dummy_csv) / "train")
    results = isoforest_loader.load_train_data(
        train_path, ["Bearing Condition", "Motor Supply Frequency"]
    )
    assert isinstance(results, tuple)
    assert isinstance(results[0], pd.DataFrame)
    assert isinstance(results[1], pd.Series)
    assert len(results) == 2

    # Save out a copy of the fitted scaler to be used for load_inference_data
    dump(
        isoforest_loader.scaler,
        Path(detection_timestamp_dir / "fitted_scaler.joblib"),
    )

    # Testing of LSTM dataloader starts here
    train_filepaths = glob(str(Path(train_path / "*.csv")))
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

    feature_data, _ = lstm_loader.load_train_data(train_path, features)
    assert isinstance(feature_data, torch.utils.data.DataLoader)

    iterate_loader = iter(feature_data)
    X = next(iterate_loader)
    assert list(X.shape) == [
        batch_size,
        lookback_period,
        (len(test_csv.columns) - 1 + len(features) * 2),
    ]
    assert isinstance(X, torch.Tensor)

    if (sum(total_possible_windows) % batch_size) != 0:
        # Check that last batch is full batch
        for i in range(total_possible_batches - 1):
            X = next(iterate_loader)
            assert isinstance(X, torch.Tensor)
            assert list(X.shape) == [
                batch_size,
                lookback_period,
                (len(test_csv.columns) - 1 + len(features) * 2),
            ]

    # Save out a copy of the fitted scaler to be used for load_inference_data
    dump(
        lstm_loader.scaler,
        Path(detection_timestamp_dir / "lstm_fitted_scaler.joblib"),
    )


def test_load_eval_data(dummy_csv):
    val_path = Path(Path(dummy_csv) / "val" / "*.csv")

    # tests for return mode == sklearn
    for file in glob(str(val_path)):
        results = isoforest_loader.load_eval_data(
            file, ["Bearing Condition", "Motor Supply Frequency"]
        )
        assert isinstance(results, tuple)
        assert isinstance(results[0], pd.DataFrame)
        assert isinstance(results[1], pd.Series)
        assert len(results) == 3

    # # Testing of LSTM dataloader starts here
    val_filepaths = glob(str(val_path))
    total_possible_windows = []
    total_possible_batches = []
    for file in val_filepaths:
        test_csv = pd.read_csv(
            file, parse_dates=True, infer_datetime_format=True, index_col=0
        )
        test_csv = curve_shift(test_csv, label_column, lookahead_period, False)
        if len(test_csv) >= lookback_period:
            total_possible_windows.append(len(test_csv) - lookback_period + 1)
            total_possible_batches.append(
                floor((len(test_csv) - lookback_period + 1) / batch_size)
            )

    for index in range(len(total_possible_windows)):
        for file in val_filepaths:
            feature_data, label_data, original_data = lstm_loader.load_eval_data(
                file, features
            )
            assert isinstance(feature_data, torch.utils.data.DataLoader)
            assert isinstance(label_data, np.ndarray)
            assert label_data.shape[0] == total_possible_windows[index]

            iterate_loader = iter(feature_data)
            X = next(iterate_loader)
            assert list(X.shape) == [
                batch_size,
                lookback_period,
                (len(test_csv.columns) - 1 + len(features) * 2),
            ]
            assert isinstance(X, torch.Tensor)

            if (total_possible_windows[index] % batch_size) != 0:
                for i in range(total_possible_batches[index] - 1):
                    X = next(iterate_loader)
                    assert list(X.shape) == [
                        batch_size,
                        lookback_period,
                        (len(test_csv.columns) - 1 + len(features) * 2),
                    ]
                    assert isinstance(X, torch.Tensor)


def test_concat_data(dummy_concat_csv):
    dummy_concat_path = Path(Path(dummy_concat_csv) / "dummy_concat")
    loader = DataLoader()
    main_data = loader._concat_data(dummy_concat_path)
    assert len(main_data) == 15
    assert len(main_data.columns) == 4
    assert isinstance(main_data, pd.DataFrame)

    shifted_data = loader._concat_data(
        dummy_concat_path, lookahead_period=3, drop_anomalous=True
    )
    assert len(shifted_data) == 12
    assert len(shifted_data.columns) == 4
    assert isinstance(shifted_data, pd.DataFrame)


def test_concat_data_fit_scaler(dummy_concat_csv):
    dummy_concat_path = Path(Path(dummy_concat_csv) / "dummy_concat")
    loader = DataLoader()
    scaled_df = loader._concat_data_fit_scaler(
        dummy_concat_path,
        features,
    )
    assert len(scaled_df) == 15
    assert len(scaled_df.columns) == 4
    assert isinstance(scaled_df, pd.DataFrame)

    shifted_data = loader._concat_data_fit_scaler(
        dummy_concat_path, features, lookahead_period=3, drop_anomalous=True
    )
    assert len(shifted_data) == 12
    assert len(shifted_data.columns) == 4
    assert isinstance(shifted_data, pd.DataFrame)


def test_load_scaler(detection_timestamp_dir):
    scaler_file = (glob(str(Path(detection_timestamp_dir / "*.joblib"))))[0]
    inference_loader = DataLoader()
    inference_loader.load_scaler(scaler_file)
    assert set(inference_loader.scaler.get_feature_names_out()) == set(features)


def test_load_inference_data(dummy_csv, detection_timestamp_dir):

    # tests for return mode == sklearn
    scaler_file = (glob(str(Path(detection_timestamp_dir / "*.joblib"))))[0]
    inference_loader = DataLoader()
    inference_loader.load_scaler(scaler_file)
    test_path = Path(Path(dummy_csv) / "test" / "*.csv")
    for file in glob(str(test_path)):
        results, original = inference_loader.load_inference_data(file, features)
        assert isinstance(results, pd.DataFrame)
        assert isinstance(original, pd.DataFrame)

    lstm_inference_loader = DataLoader(
        "pytorch",
        batch_size,
        lookback_period,
        lookahead_period,
        2,
        pin_memory,
        num_workers,
    )
    lstm_inference_loader.load_scaler(scaler_file)
    test_path = Path(Path(dummy_csv) / "test" / "*.csv")
    for file in glob(str(test_path)):

        results, original = lstm_inference_loader.load_inference_data(file, features)
        assert isinstance(results, torch.utils.data.DataLoader)
        assert isinstance(original, pd.DataFrame)
        for data in iter(results):
            assert isinstance(data, torch.Tensor)

        data = pd.read_csv(
            file, parse_dates=True, infer_datetime_format=True, index_col=0
        )
        assert original.equals(data)
