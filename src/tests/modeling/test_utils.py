# pylint: skip-file
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.modeling.utils import (
    curve_shift,
    drop_final_lookahead,
    post_process,
    split_training_timeseries,
)

features = [
    "Bearing Condition",
    "Motor Supply Frequency",
]
time_col = "MEASUREMENT_TAKEN_ON(UTC)"
label_column = "Anomaly"

lookahead_period = 24


@pytest.fixture
def dummy_data():
    date_range = pd.date_range("2021-07-01", periods=10, freq="H")
    test_data = pd.DataFrame([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], columns=["Anomaly"])
    test_data.set_index(date_range, inplace=True)
    return test_data


@pytest.fixture
def dummy_data_2():
    date_range = pd.date_range("2021-07-01", periods=10, freq="H")
    test_data = pd.DataFrame([0, 0, 0, 0, 1, 0, 0, 0, 1, 0], columns=["Anomaly"])
    test_data.set_index(date_range, inplace=True)
    return test_data


@pytest.fixture
def dummy_data_3():
    np.random.seed(18)
    nrows = 200
    date_range = pd.date_range("2021-07-01", periods=nrows, freq="1H")
    cols = [
        "Acceleration RMS (Axial)",
        "Acceleration RMS (Radial)",
        "Acceleration RMS (Tangential)",
        "Anomaly",
    ]

    dummy_data = pd.DataFrame(
        np.random.random((nrows, len(cols))),
        columns=cols,
    )
    dummy_data.set_index(date_range, inplace=True)
    dummy_data.index.name = "MEASUREMENT_TAKEN_ON(UTC)"
    dummy_data.loc[:, "Anomaly"] = np.random.randint(0, 1, size=(len(dummy_data), 1))

    return dummy_data


@pytest.fixture
def dummy_data_4():
    date_range = pd.date_range("2021-07-01", periods=10, freq="H")
    test_data = pd.DataFrame([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], columns=["Anomaly"])
    test_data.set_index(date_range, inplace=True)
    return test_data


@pytest.fixture
def dummy_data_5():
    date_range = pd.date_range("2021-07-01", periods=10, freq="H")
    test_data = pd.DataFrame([0, 0, 0, 0, 1, 0, 1, 0, 1, 0], columns=["Anomaly"])
    test_data.set_index(date_range, inplace=True)
    return test_data


@pytest.fixture
def dummy_data_6():
    date_range = pd.date_range("2021-07-01", periods=10, freq="H")
    test_data = pd.DataFrame([0, 0, 0, 0, 1, 0, 0, 1, 1, 1], columns=["Anomaly"])
    test_data.set_index(date_range, inplace=True)
    return test_data


@pytest.fixture
def curve_shift_answer():
    dummy_answer = pd.DataFrame([0, 1, 1, 1, 0, 1, 1, 1], columns=["Anomaly"])
    return dummy_answer


@pytest.fixture
def curve_shift_answer_2():
    dummy_answer = pd.DataFrame([0, 1, 1, 1, 1, 1, 1], columns=["Anomaly"])
    return dummy_answer


@pytest.fixture
def drop_final_answer_1():
    dummy_answer = pd.DataFrame(
        [
            0,
            0,
            0,
            0,
            1,
            0,
        ],
        columns=["Anomaly"],
    )
    return dummy_answer


@pytest.fixture
def drop_final_answer_2():
    dummy_answer = pd.DataFrame([0, 0, 0, 0, 1, 0, 1, 0, 1], columns=["Anomaly"])
    return dummy_answer


@pytest.fixture
def drop_final_answer_3():
    dummy_answer = pd.DataFrame([0, 0, 0, 0, 1, 0, 0, 1, 1, 1], columns=["Anomaly"])
    return dummy_answer


@pytest.mark.parametrize(
    "data, answer",
    [("dummy_data", "curve_shift_answer"), ("dummy_data_2", "curve_shift_answer_2")],
)
def test_curve_shift(data, answer, request):
    data = request.getfixturevalue(data)
    answer = request.getfixturevalue(answer)
    test_df = curve_shift(data, "Anomaly", 3, drop_anomalous=True)
    assert isinstance(test_df, pd.DataFrame)

    test_df.reset_index(drop=True, inplace=True)
    assert test_df["Anomaly"].equals(answer["Anomaly"])


@pytest.mark.parametrize(
    "data, answer",
    [
        ("dummy_data_4", "drop_final_answer_1"),
        ("dummy_data_5", "drop_final_answer_2"),
        ("dummy_data_6", "drop_final_answer_3"),
    ],
)
def test_drop_final_lookahead(data, answer, request):
    data = request.getfixturevalue(data)
    answer = request.getfixturevalue(answer)
    test_df = drop_final_lookahead(data, 4, "Anomaly")
    assert isinstance(test_df, pd.DataFrame)

    test_df.reset_index(drop=True, inplace=True)
    assert test_df["Anomaly"].equals(answer["Anomaly"])


@pytest.mark.parametrize("shift_period", [5, 0])
def test_post_process(dummy_data_3, shift_period):
    X_inference = dummy_data_3.copy()
    X_inference.drop("Anomaly", axis=1, inplace=True)
    anomaly_score = dummy_data_3["Anomaly"].values.copy()
    y_predict = dummy_data_3["Anomaly"].values.copy()
    post_df, post_df_daily = post_process(
        X_inference, anomaly_score, y_predict, shift_period, lookback_period=3
    )
    for df in (post_df, post_df_daily):
        assert isinstance(df, pd.DataFrame)

    assert len(post_df) == len(dummy_data_3)
    ndays = len(dummy_data_3) // 24
    assert ndays <= len(post_df_daily) <= ndays + 1

    if shift_period > 0:
        assert len(post_df.columns) == (len(dummy_data_3.columns) - 1 + 3)
        assert len(post_df_daily.columns) == 3  # Measurement_taken_date is index
    elif shift_period == 0:
        assert len(post_df.columns) == (len(dummy_data_3.columns) - 1 + 2)
        assert len(post_df_daily.columns) == 2  # Measurement_taken_date is index


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

    date_range = pd.date_range("2021-07-01", periods=300, freq="H")
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
                    np.random.randint(100, size=(300, 2)),
                    columns=test_col[:2],
                )
            else:
                test_data = pd.DataFrame(
                    np.random.randint(100, size=(300, 3)),
                    columns=test_col,
                )

            test_data.set_index(date_range, inplace=True)
            test_data.index.name = time_col
            test_data["Asset_Operating"] = 1
            if partition != "inference":
                test_data[label_column] = 0
                test_data.iloc[130:140, -2] = 1
                test_data.iloc[180:184, -2] = 1
            test_data.to_csv(str(directory) + "/" + str(i) + ".csv")

    return str(timestamp_dir)


def test_split_training_timeseries(dummy_csv):
    for partition in ["train", "val", "inference"]:
        for filepath in glob(str(Path(Path(dummy_csv) / partition / "*.csv"))):
            if partition == "train":
                test_csv = pd.read_csv(
                    filepath, parse_dates=True, infer_datetime_format=True, index_col=0
                )
                test_csv = curve_shift(test_csv, label_column, lookahead_period, True)
                test_csv = test_csv.drop(test_csv[test_csv[label_column] == 1].index)
                test_csv = test_csv.drop(
                    test_csv[test_csv["Asset_Operating"] != 1].index
                )
                data_list = split_training_timeseries(test_csv)
                assert isinstance(data_list, list)

                print(data_list)
                assert len(data_list) > 1
                for data in data_list:
                    assert isinstance(data, pd.DataFrame)

            elif partition == "val":
                test_csv = pd.read_csv(
                    filepath, parse_dates=True, infer_datetime_format=True, index_col=0
                )
                test_csv = curve_shift(test_csv, label_column, lookahead_period, False)
                data_list = split_training_timeseries(test_csv)
                assert isinstance(data_list, list)
                assert len(data_list) == 1
                for data in data_list:
                    assert isinstance(data, pd.DataFrame)

            else:
                test_csv = pd.read_csv(
                    filepath, parse_dates=True, infer_datetime_format=True, index_col=0
                )
                data_list = split_training_timeseries(test_csv)
                assert isinstance(data_list, list)
                assert len(data_list) == 1
                for data in data_list:
                    assert isinstance(data, pd.DataFrame)
