# pylint: skip-file

import csv
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.data_prep.split_data import DataSplitter


@pytest.fixture(scope="session")
def asset_dir(tmpdir_factory):
    assets_dir = tmpdir_factory.mktemp("assets")

    asset_dict = {
        "Blower": 200,
        "Agitator": 100,
        "Fan": 5,
        "Pump": 4,
        "Crusher": 1,
        "Pelletizer": 1,
    }
    for key, count in asset_dict.items():
        for i in range(count):
            filepath = Path(assets_dir / f"{key}_{i}.csv")

            with open(filepath, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["key, count"])
                writer.writerow([key, count])

    return assets_dir


@pytest.fixture(scope="session")
def asset_dir_timesplit(tmpdir_factory):
    assets_dir = tmpdir_factory.mktemp("assets_timesplit")
    asset_dict = {
        "Blower": 3,
        "Agitator": 2,
        "Fan": 2,
    }
    date_range = pd.date_range("2021-07-01", periods=864, freq="1H")
    features = [
        "Bearing Condition",
        "Motor Supply Frequency",
        "Nr. Of Starts Between Mea...nts",
        "Output Power",
        "Anomaly",
    ]

    np.random.seed(18)
    for key, count in asset_dict.items():
        for i in range(count):
            filepath = Path(assets_dir / f"{key}_{i}_20210701-20210722.csv")

            data = pd.DataFrame(
                np.random.randint(-100, 100, size=(864, 5)),
                columns=features,
            )

            data.set_index(date_range, inplace=True)
            data.index.name = "MEASUREMENT_TAKEN_ON(UTC)"

            data.loc[:, "Anomaly"] = np.random.randint(0, 1, size=(len(data), 1))
            data.to_csv(filepath)

    return assets_dir, sum(asset_dict.values())


def test_split_files(asset_dir):
    paths, apps = DataSplitter().split_files(asset_dir, test_size=0.2, random_state=42)

    # check that return type is correct
    assert isinstance(paths, dict)
    assert isinstance(apps, set)

    # check that train val test are the only keys in paths
    assert set(paths.keys()) == {"train", "val", "test"}

    # check that the list of filepaths correspond to the number of files in asset_dir
    assert sum((len(ls) for _, ls in paths.items())) == len(os.listdir(asset_dir))

    # check that the set of applications returned is as expected
    assert apps == {"Blower", "Agitator", "Fan", "Pump", "Crusher", "Pelletizer"}


def test_read_asset_dir(asset_dir):
    results = DataSplitter().read_asset_dir(asset_dir)

    # check that it is of correct return type
    assert isinstance(results, pd.DataFrame)

    # check that all files in asset_dir are represented in returned dataframe
    assert len(results) == len(os.listdir(asset_dir))

    # check that columns 'Filepath', 'Application' are in dataframe
    assert set(results.columns) == {"Filepath", "Application"}


@pytest.mark.parametrize("test_size", [0.2, 0.15])
def test_stratified_split_assets(asset_dir, test_size):
    ds = DataSplitter()
    asset_df = ds.read_asset_dir(asset_dir)
    train, test = ds._stratified_split_assets(
        asset_df, test_size=test_size, random_state=42
    )

    # check that train, val, test are of correct return types
    assert all((isinstance(df, pd.DataFrame) for df in (train, test)))

    # check that the rows in train, val, test sum up to number of rows in df
    assert (len(train) + len(test)) == len(asset_df)

    # check that test size is proportionate to test_size indicated
    assert np.isclose(len(test) / len(asset_df), test_size, atol=5e-03)

    # check that indices in train, test are distinct
    assert set(train.index).isdisjoint(set(test.index))

    # check that train val test are stratified against application
    app_prop = lambda df: df[df["Application"].isin(["Blower", "Agitator"])].groupby(
        ["Application"]
    ).count()["Filepath"] / len(df)
    assert np.allclose(app_prop(train), app_prop(test), atol=1e-02)

    # check that rare assets are in train set instead of test set
    assert len(train[train["Application"].isin(["Crusher", "Pelletizer"])]) > 0
    assert len(test[test["Application"].isin(["Crusher", "Pelletizer"])]) == 0


def test_split_by_time(asset_dir_timesplit):
    asset_dir, asset_count = asset_dir_timesplit
    time_col = "MEASUREMENT_TAKEN_ON(UTC)"
    paths, apps = DataSplitter().split_by_time(
        asset_dir, test_size=0.2, time_col=time_col
    )

    # check that return type is correct
    assert isinstance(paths, dict)
    assert isinstance(apps, set)

    # check that train val test are the only keys in paths
    assert set(paths.keys()) == {"train", "val", "test"}

    # check that no. of items in each partition correspond to the no. of files in asset_dir
    for _, ls in paths.items():
        assert len(ls) == asset_count

    # check that the set of applications returned is as expected
    assert apps == {"Blower", "Agitator", "Fan"}

    # check that files are chronologically partitioned as intended.
    for train_fp, val_fp, test_fp in zip(*paths.values()):
        train = pd.read_csv(
            train_fp,
            parse_dates=[time_col],
            infer_datetime_format=True,
            index_col=0,
        )
        val = pd.read_csv(
            val_fp,
            parse_dates=[time_col],
            infer_datetime_format=True,
            index_col=0,
        )
        test = pd.read_csv(
            test_fp,
            parse_dates=[time_col],
            infer_datetime_format=True,
            index_col=0,
        )
        for before, after in ((train, val), (val, test)):
            before_end = before.index.max().date()
            after_start = after.index.min().date()
            after_end = after.index.max().date()

            assert before_end <= after_start
            assert before_end < after_end
