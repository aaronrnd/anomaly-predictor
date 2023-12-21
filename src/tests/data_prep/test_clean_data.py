# pylint: skip-file

import copy

import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.data_prep.clean_data import DataCleaner

features_to_standardize = [
    "Output Power",
    "Overall Vibration",
    "Peak to Peak (X)",
    "Peak to Peak (Y)",
    "Peak to Peak (Z)",
    "Skin Temperature",
    "Speed",
    "Vibration (Axial)",
    "Vibration (Radial)",
    "Vibration (Tangential)",
    "Acceleration RMS (Axial)",
    "Acceleration RMS (Radial)",
    "Acceleration RMS (Tangential)",
    "Bearing Condition",
]

robust_args = {
    "with_centering": True,
    "with_scaling": True,
    "quantile_range": (25.0, 75.0),
    "copy": True,
    "unit_variance": False,
}


@pytest.fixture
def create_dummy_data():
    np.random.seed(18)
    date_range = pd.date_range("2021-07-01", periods=8640, freq="15T")
    test_col = [
        "Bearing Condition",
        "Motor Supply Frequency",
        "Nr. Of Starts Between Mea...nts",
        "Output Power",
        "Overall Vibration",
        "Peak to Peak (X)",
        "Peak to Peak (Y)",
        "Peak to Peak (Z)",
        "Skin Temperature",
        "Speed",
        "Total Number Of Starts",
        "Total Running Time",
        "Vibration (Axial)",
        "Vibration (Radial)",
        "Vibration (Tangential)",
        "Acceleration RMS (Axial)",
        "Acceleration RMS (Radial)",
        "Acceleration RMS (Tangential)",
        "Anomaly",
    ]

    test_data = pd.DataFrame(
        np.random.randint(-100, 100, size=(8640, 19)),
        columns=test_col,
    )

    test_data.set_index(date_range, inplace=True)
    test_data.index.name = "MEASUREMENT_TAKEN_ON(UTC)"
    test_data = test_data.drop(test_data.iloc[50:750].index)
    test_data = test_data.drop(test_data.iloc[2000:2100].index)
    test_data = test_data.drop(test_data.iloc[3200:3850].index)
    test_data = test_data.drop(test_data.iloc[5000:5800].index)
    test_data = test_data.drop(test_data.iloc[7000:7100].index)
    test_data = test_data.mask(np.random.random(test_data.shape) < 0.1)
    test_data.loc[6000:, "Speed"] = 0
    test_data.loc[:, "Anomaly"] = np.random.randint(0, 1, size=(len(test_data), 1))
    test_data.loc[:, "Bearing Condition"] = np.random.randint(
        0, 9, size=(len(test_data), 1)
    )
    test_shape = test_data.shape

    return test_data, test_shape


@pytest.fixture
def create_scale_dummy():
    test_data = pd.DataFrame(
        {
            "Speed": [50, 50, 50, 50, 50],
            "Peak to Peak (X)": [22, 23, 25, 26, 27],
            "Vibration (Axial)": [13, 15, 19, 20, 22],
            "Skin Temperature": [33, 34, 35, 36, 35],
            "Bearing Condition": [0, 0, 0, 0, 0],
        }
    )
    minmax_answer = pd.DataFrame(
        {
            "Speed": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Peak to Peak (X)": [0.0, 0.2, 0.6, 0.8, 1.0],
            "Vibration (Axial)": [0.0, 0.2, 0.7, 0.8, 1.0],
            "Skin Temperature": [0.0, 0.3, 0.7, 1.0, 0.7],
            "Bearing Condition": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    robust_answer = pd.DataFrame(
        {
            "Speed": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Peak to Peak (X)": [-1.0, -0.7, 0.0, 0.3, 0.7],
            "Vibration (Axial)": [-1.2, -0.8, 0.0, 0.2, 0.6],
            "Skin Temperature": [-2.0, -1.0, 0.0, 1.0, 0.0],
            "Bearing Condition": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    standard_answer = pd.DataFrame(
        {
            "Speed": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Peak to Peak (X)": [-1.4, -0.9, 0.2, 0.8, 1.3],
            "Vibration (Axial)": [-1.4, -0.8, 0.4, 0.7, 1.3],
            "Skin Temperature": [-1.6, -0.6, 0.4, 1.4, 0.4],
            "Bearing Condition": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return (test_data, minmax_answer, robust_answer, standard_answer)


@pytest.fixture
def create_dummy_date_dict():
    dummy_date_dict = {
        "start_date": [
            pd.Timestamp("2021-07-01 00:00:00", freq="H"),
            pd.Timestamp("2021-07-08 18:00:00", freq="H"),
            pd.Timestamp("2021-07-30 03:00:00", freq="H"),
            pd.Timestamp("2021-08-18 09:00:00", freq="H"),
            pd.Timestamp("2021-09-14 11:00:00", freq="H"),
        ],
        "end_date": [
            pd.Timestamp("2021-07-01 12:00:00", freq="H"),
            pd.Timestamp("2021-07-29 02:00:00", freq="H"),
            pd.Timestamp("2021-08-11 15:00:00", freq="H"),
            pd.Timestamp("2021-09-06 04:00:00", freq="H"),
            pd.Timestamp("2021-09-28 23:00:00", freq="H"),
        ],
    }
    dummy_clean_dict = {
        "start_date": [
            pd.Timestamp("2021-07-01 00:00:00", freq="H"),
            pd.Timestamp("2021-07-08 18:00:00", freq="H"),
            pd.Timestamp("2021-08-18 09:00:00", freq="H"),
            pd.Timestamp("2021-09-14 11:00:00", freq="H"),
        ],
        "end_date": [
            pd.Timestamp("2021-07-01 12:00:00", freq="H"),
            pd.Timestamp("2021-08-11 15:00:00", freq="H"),
            pd.Timestamp("2021-09-06 04:00:00", freq="H"),
            pd.Timestamp("2021-09-28 23:00:00", freq="H"),
        ],
    }
    return dummy_date_dict, dummy_clean_dict


@pytest.fixture
def dummy_data_list():
    np.random.seed(55)
    nrows_1 = 100
    nrows_2 = 50
    test_col = [
        "Speed",
        "Peak to Peak (X)",
        "Vibration (Axial)",
        "Skin Temperature",
        "Bearing Condition",
    ]
    df_1 = pd.DataFrame(
        np.random.rand(nrows_1, len(test_col)),
        columns=test_col,
    )
    df_2 = pd.DataFrame(
        np.random.rand(nrows_2, len(test_col)),
        columns=test_col,
    )

    for df in (df_1, df_2):
        for col in test_col:
            if col == "Speed":
                df["Speed"] = df["Speed"].mask(np.random.random(len(df)) < 0.3, other=0)
            elif col == "Bearing Condition":
                df.loc[df["Speed"] == 0, "Bearing Condition"] = 0
            else:
                df[col] = df[col].mask(np.random.random(len(df)) < 0.1)
    return [df_1, df_2]


@pytest.fixture
def zero_speed_list():
    np.random.seed(55)
    nrows = 50
    test_col = [
        "Speed",
        "Peak to Peak (X)",
        "Vibration (Axial)",
        "Skin Temperature",
        "Bearing Condition",
    ]
    df = pd.DataFrame(
        np.random.rand(nrows, len(test_col)),
        columns=test_col,
    )

    for col in test_col:
        if col == "Speed":
            df["Speed"] = np.zeros(nrows)
        elif col == "Bearing Condition":
            df.loc[df.index < nrows // 2, "Bearing Condition"] = 0
            df.loc[df.index >= nrows // 2, "Bearing Condition"] = np.nan
        else:
            df[col] = df[col].mask(np.random.random(len(df)) < 0.1)
    return [df]


@pytest.fixture
def small_data_list():
    df = pd.DataFrame(
        {
            "Speed": [50, 50, 50, 50, 0],
            "Peak to Peak (X)": [22, 23, np.nan, 26, 27],
            "Vibration (Axial)": [13, 15, 19, 20, np.nan],
            "Skin Temperature": [33, 34, 35, 36, 35],
            "Bearing Condition": [5, 6, 3, 1, np.nan],
        }
    )
    return [df]


@pytest.fixture
def list_of_df_float():
    df = pd.DataFrame(
        {
            "Speed": [50, 50, 50, 50, 0],
            "Peak to Peak (X)": [22, 23, 19, 26, 27],
            "Vibration (Axial)": [13, 15, 19, 20, 10],
            "Anomaly": [0.83, 1, 0, 0.2, 1.32],
            "Bearing Condition": [0.766, 1.20, 3, 8.011, 3],
        }
    )
    return [df]


def test_clean_data(create_dummy_data):
    test_df, test_shape = create_dummy_data
    test_list = DataCleaner().clean_data(
        test_df,
        features_to_standardize,
        standardization_method="RobustScaler",
        scaler_args=robust_args,
        outlier_threshold=0.5,
        outlier_window=24,
        nan_range_threshold=48,
        impute_nan_window=48,
        impute_nan_period=1,
        impute_nan_center=True,
        bearing_cond_fill=-1,
        convert_to_int=["Bearing Condition", "Anomaly"],
    )

    assert isinstance(test_list, list)

    for item in test_list:

        # Check that it's the correct number of columns
        assert item.shape[1] == len(features_to_standardize) + 1

        # Check that it's DataFrames within the list
        assert isinstance(item, pd.DataFrame)

        # check that features ["Anomaly","Bearing Condition"] are not floats
        assert (item["Anomaly"].apply(float.is_integer).all()) == True
        assert (item["Bearing Condition"].apply(float.is_integer).all()) == True


def test_keep_features(create_dummy_data):
    test_df, test_shape = create_dummy_data
    test_df = DataCleaner()._keep_features(test_df, features_to_standardize)
    assert test_df.shape[1] == len(features_to_standardize) + 1
    assert isinstance(test_df, pd.DataFrame)


def test_check_negatives(create_dummy_data):
    test_df, test_shape = create_dummy_data
    test_df = DataCleaner()._check_negatives(test_df)
    assert test_df.shape == test_shape
    assert ((test_df < 0).any().any()) == False
    assert isinstance(test_df, pd.DataFrame)


def test_check_standardization(create_scale_dummy):
    test_df, minmax_answer, robust_answer, standard_answer = create_scale_dummy

    minmax_df = DataCleaner()._standardization(test_df, test_df.columns, "MinMaxScaler")
    robust_df = DataCleaner()._standardization(
        test_df, test_df.columns, "RobustScaler", robust_args
    )
    standard_df = DataCleaner()._standardization(
        test_df, test_df.columns, "StandardScaler"
    )

    assert np.round(minmax_df, 1).astype(float).equals(minmax_answer)
    assert np.round(robust_df, 1).astype(float).equals(robust_answer.astype(float))
    assert np.round(standard_df, 1).astype(float).equals(standard_answer.astype(float))
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(minmax_answer, pd.DataFrame)


def test_check_outliers(create_dummy_data):
    test_df, test_shape = create_dummy_data
    test_standardized_df = DataCleaner()._standardization(
        test_df, features_to_standardize, "RobustScaler", robust_args
    )
    test_df = DataCleaner()._check_outliers(
        test_standardized_df, test_df, test_df.columns, 0.5, 5
    )
    assert test_df.shape == test_shape
    assert isinstance(test_df, pd.DataFrame)


@pytest.mark.parametrize("test_input", ["dummy_data_list", "zero_speed_list"])
def test_preprocess_bearing_cond(test_input, request):
    original_data_list = copy.deepcopy(request.getfixturevalue(test_input))
    dfs = DataCleaner()._preprocess_bearing_cond(request.getfixturevalue(test_input))

    # check that return type is correct
    assert isinstance(dfs, list)
    assert all((isinstance(df, pd.DataFrame) for df in dfs))

    # check that no rows & columns are added, removed or replaced
    for orig_df, processed_df in zip(original_data_list, dfs):
        assert orig_df.shape == processed_df.shape
        assert set(orig_df.columns) == set(processed_df.columns)

    # check that there exists no instances where speed is 0 and bearing cond is non null
    for df in original_data_list:
        assert len(df[(df["Bearing Condition"].notnull()) & (df["Speed"] == 0)]) > 0
    for df in dfs:
        assert len(df[(df["Bearing Condition"].notnull()) & (df["Speed"] == 0)]) == 0


@pytest.mark.parametrize(
    "test_input", ["dummy_data_list", "zero_speed_list", "small_data_list"]
)
def test_impute_nans(test_input, request):
    original_data_list = copy.deepcopy(request.getfixturevalue(test_input))
    cleaner = DataCleaner()
    dfs = cleaner._preprocess_bearing_cond(request.getfixturevalue(test_input))
    if test_input == "small_data_list":
        dfs = cleaner._impute_nans(dfs, lookback_period=3, min_periods=1, center=False)
    else:
        dfs = cleaner._impute_nans(dfs)

    # check that return type is correct
    assert isinstance(dfs, list)
    assert all((isinstance(df, pd.DataFrame) for df in dfs))

    # check that no rows & columns are added, removed or replaced
    for orig_df, processed_df in zip(original_data_list, dfs):
        assert orig_df.shape == processed_df.shape
        assert set(orig_df.columns) == set(processed_df.columns)

    # check that there remains no missing values for all columns except Bearing Condition
    for df in dfs:
        assert np.all(df.loc[:, df.columns != "Bearing Condition"].isnull().sum() == 0)

    # check that dataframe is correctly imputed with rolling mean and other values remains unchanged
    if test_input == "small_data_list":

        answer = pd.DataFrame(
            {
                "Speed": [50, 50, 50, 50, 0],
                "Peak to Peak (X)": [22, 23, 22.5, 26, 27],
                "Vibration (Axial)": [13, 15, 19, 20, 18],
                "Skin Temperature": [33, 34, 35, 36, 35],
                "Bearing Condition": [5, 6, 3, 1, 1],
            }
        )
        assert dfs[0].astype(float).equals(answer.astype(float))


@pytest.mark.parametrize("test_input", ["dummy_data_list", "zero_speed_list"])
def test_impute_bearing_cond(test_input, request):
    original_data_list = copy.deepcopy(request.getfixturevalue(test_input))
    fill_value = 999
    cleaner = DataCleaner()
    dfs = cleaner._preprocess_bearing_cond(request.getfixturevalue(test_input))
    dfs = cleaner._impute_nans(dfs)
    dfs = cleaner._impute_bearing_cond(dfs, fill_value=fill_value)

    # check that return type is correct
    assert isinstance(dfs, list)
    assert all((isinstance(df, pd.DataFrame) for df in dfs))

    # check that no rows & columns are added, removed or replaced
    for orig_df, processed_df in zip(original_data_list, dfs):
        assert orig_df.shape == processed_df.shape
        assert set(orig_df.columns) == set(processed_df.columns)

    # check that there remains no missing values for bearing condition
    for df in dfs:
        assert df["Bearing Condition"].isnull().sum() == 0

    # check that imputed value is as specified
    if test_input == "zero_speed_list":
        assert np.all(df["Bearing Condition"] == fill_value)


def test_float_to_int(list_of_df_float):
    float_df = DataCleaner()._float_to_int(
        list_of_df_float, ["Anomaly", "Bearing Condition"]
    )

    # check that return type is correct
    assert isinstance(float_df, list)
    assert all((isinstance(df, pd.DataFrame) for df in float_df))

    # check that features ["Anomaly","Bearing Condition"] are not floats
    assert (float_df[0]["Anomaly"].apply(float.is_integer).all()) == True
    assert (float_df[0]["Bearing Condition"].apply(float.is_integer).all()) == True
