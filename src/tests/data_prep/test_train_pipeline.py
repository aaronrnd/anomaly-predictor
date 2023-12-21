# pylint: skip-file

import datetime
import json
import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.data_prep.feature_engineering import FeatureEngineer
from src.anomaly_predictor.data_prep.train_pipeline import run_pipeline

CORRECT_ANOMALY_LABEL = 1
TIME_COL = "MEASUREMENT_TAKEN_ON(UTC)"
CUT_OFF_DATE = "01-07-21"


@pytest.fixture(scope="session")
def input_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("input"))


@pytest.fixture(scope="session")
def annotation_xlsx(tmpdir_factory, input_dir):
    annotation_dir = Path(os.sep.join([input_dir, "annotations"]))
    annotation_dir.mkdir()
    filepath = os.sep.join([str(annotation_dir), "annotation.xlsx"])
    column_names = [
        "Rename",
        "Start Date",
        "Start Time",
        "End Date",
        "End Time",
        "Description",
    ]
    anomaly = [
        "Blower_1",
        pd.to_datetime("1-Jul-21", dayfirst=True),
        0,
        pd.to_datetime("2-Jul-21", dayfirst=True),
        0,
        "ANOMALY",
    ]
    values = [[], column_names, anomaly]
    df = pd.DataFrame(values)
    df.to_excel(filepath, index=False, header=False)
    return os.sep.join(["annotations", "annotation.xlsx"])


@pytest.fixture(scope="session")
def assets_dir(tmpdir_factory, input_dir):
    np.random.seed(8642)
    asset_dir = Path(os.sep.join([input_dir, "assets"]))
    asset_dir.mkdir()
    asset_dict = {
        "Blower": 20,
        "Agitator": 10,
        "Fan": 5,
        "Pump": 4,
        "Crusher": 1,
    }

    cols = [
        "Motor Supply Frequency",
        "Nr. Of Starts Between Mea...nts",
        "Output Power",
        "Speed",
        "Peak to Peak (X)",
        "Vibration (Axial)",
        "Skin Temperature",
        "Bearing Condition",
    ]
    for key, count in asset_dict.items():
        for i in range(count):
            asset_name = f"{key}_{i}"
            asset_subdir = Path(asset_dir / asset_name)
            asset_subdir.mkdir()
            filepath = Path(asset_subdir / f"{asset_name}_20210701-20210829.xlsx")
            time_range = pd.date_range("2021-07-01", periods=216, freq="H")
            df_list = []
            speed_copy = None
            for col in cols:
                df = pd.DataFrame([time_range], index=[TIME_COL]).transpose()
                df["data"] = np.random.randint(-10, 1000, size=216)
                if col == "Speed":
                    df["data"] = df["data"].mask(
                        np.random.random(len(df)) < 0.3, other=0
                    )
                    speed_copy = df["data"].copy()
                elif col == "Bearing Condition":
                    df.loc[speed_copy == 0, "data"] = 0
                else:
                    df["data"] = df["data"].mask(np.random.random(len(df)) < 0.1)
                df_list.append(df)
            writer = pd.ExcelWriter(filepath, engine="openpyxl")
            for col, df in zip(cols, df_list):
                df.to_excel(writer, sheet_name=col, index=False)
            writer.save()

    return str(asset_dir)


@pytest.fixture(scope="session")
def interim_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("interim"))


@pytest.fixture(scope="session")
def processed_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("processed"))


@pytest.fixture(scope="session")
def encoder_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("encoder"))


@pytest.fixture(scope="session")
def config_args(
    input_dir, annotation_xlsx, assets_dir, interim_dir, processed_dir, encoder_dir
):
    with hydra.initialize("../../../conf"):
        std_features = [
            "Output Power",
            "Speed",
            "Peak to Peak (X)",
            "Vibration (Axial)",
            "Skin Temperature",
            "Motor Supply Frequency",
            "Bearing Condition",
        ]
        split_json = None
        overrides = [
            f"data_prep.pipeline.input_dir={input_dir}",
            f"data_prep.pipeline.interim_dir={interim_dir}",
            f"data_prep.pipeline.processed_dir={processed_dir}",
            f"data_prep.ingest_data.assets_dir={assets_dir}",
            f"data_prep.ingest_data.annotation_list=[{annotation_xlsx}]",
            f"data_prep.split_data.follow_split=",
            f"data_prep.clean_data.features_to_standardize={std_features}",
        ]
        return hydra.compose(config_name="train_pipeline.yml", overrides=overrides)


@pytest.fixture(scope="session")
def follow_split_config_args(
    input_dir, annotation_xlsx, assets_dir, interim_dir, processed_dir, encoder_dir
):
    with hydra.initialize("../../../conf"):
        std_features = [
            "Output Power",
            "Speed",
            "Peak to Peak (X)",
            "Vibration (Axial)",
            "Skin Temperature",
            "Motor Supply Frequency",
            "Bearing Condition",
        ]
        json_file_path = Path(
            processed_dir / (os.listdir(f"{processed_dir}")[0]) / "split_data.json"
        )

        overrides = [
            f"data_prep.pipeline.input_dir={input_dir}",
            f"data_prep.pipeline.interim_dir={interim_dir}",
            f"data_prep.pipeline.processed_dir={processed_dir}",
            f"data_prep.ingest_data.assets_dir={assets_dir}",
            f"data_prep.ingest_data.annotation_list=[{annotation_xlsx}]",
            f"data_prep.split_data.follow_split={json_file_path}",
            f"data_prep.clean_data.features_to_standardize={std_features}",
        ]

        return hydra.compose(config_name="train_pipeline.yml", overrides=overrides)


def test_run_pipeline(
    config_args, processed_dir, input_dir, annotation_xlsx, interim_dir
):
    args = config_args
    assert args["data_prep"]["pipeline"]["input_dir"] == input_dir
    paths, processed_dir_path = run_pipeline(args)
    # Check that the output is as intended (dict with values being list)
    assert isinstance(paths, dict)
    assert set(paths.keys()) == {"train", "val", "test"}
    assert isinstance(processed_dir_path, Path)

    # Check that the unique number of assets in each train, val, test adds up to the number of assets in input folder
    if args["data_prep"]["split_data"]["by"] == "asset":
        assert sum((len(ls) for _, ls in paths.items())) == len(
            glob(str(Path(interim_dir / "*" / "*")))
        )

    # check that the filepaths in returned dict are indeed found in output_dir
    assert set(glob(str(Path(processed_dir / "*" / "train" / "*")))) == set(
        paths["train"]
    )
    assert set(glob(str(Path(processed_dir / "*" / "val" / "*")))) == set(paths["val"])
    assert set(glob(str(Path(processed_dir / "*" / "test" / "*")))) == set(
        paths["test"]
    )

    # check that json file was generated as split_data.json and stored in processed_dir
    timestamp = (Path(paths["train"][0]).parent).parent.stem
    assert os.path.isfile(Path(processed_dir / timestamp / "split_data.json"))

    for _, ls in paths.items():

        filepath = ls[0]
        df = pd.read_csv(filepath)

        # Check that newly added columns from feature engineering module exists on
        assert all(
            col in df.columns
            for col in [
                "Variable_speed",
                "Application_Blower",
                "Application_Fan",
                "Application_Crusher",
                "Application_Agitator",
                "Application_Pump",
            ]
        )

        # Test that no missing value exist
        # assert np.all(df.isnull().sum() == 0)


def test_follow_split_pipeline(
    follow_split_config_args, processed_dir, input_dir, assets_dir
):
    args = follow_split_config_args
    json_file_path = Path(
        processed_dir / (os.listdir(f"{processed_dir}")[0]) / "split_data.json"
    )

    # Check that the arg from hydra is expected filepath
    assert (args["data_prep"]["split_data"]["follow_split"]) == str(json_file_path)
    paths, processed_dir_path = run_pipeline(args)

    # Check that the output is as intended (dict with values being list)
    assert isinstance(paths, dict)
    assert set(paths.keys()) == {"train", "val", "test"}
    assert isinstance(processed_dir_path, Path)

    # Check that the unique number of assets in each train, val, test adds up to the number of assets in input folder
    if args["data_prep"]["split_data"]["by"] == "asset":
        assert (
            sum((len(ls) for _, ls in paths.items()))
            == len(list(Path(assets_dir).glob("**/"))) - 1
        )

    # Since there's multiple process timestamps, to pick the latest created folder
    latest_dir = Path(max(glob(str(Path(processed_dir / "*"))), key=os.path.getmtime))

    # check that the filepaths in returned dict are indeed found in output_dir
    assert set(glob(str(Path(latest_dir / "train" / "*")))) == set(paths["train"])
    assert set(glob(str(Path(latest_dir / "val" / "*")))) == set(paths["val"])
    assert set(glob(str(Path(latest_dir / "test" / "*")))) == set(paths["test"])

    # Open specified follow_split.json file
    with open(Path(json_file_path), "r") as fp:
        loaded_json_file = json.load(fp)

    # Open saved .json file in current processed timestamp folder
    with open(Path(latest_dir / "split_data.json"), "r") as fp:
        saved_json_file = json.load(fp)

    # check that the .json in current process timestamp folder is same as the one we referenced in split_data.follow_split
    assert isinstance(loaded_json_file, dict)
    assert isinstance(saved_json_file, dict)
    assert loaded_json_file == saved_json_file

    for _, ls in paths.items():

        filepath = ls[0]
        df = pd.read_csv(filepath)

        # Check that newly added columns from feature engineering module exists on
        assert all(
            col in df.columns
            for col in [
                "Variable_speed",
                "Application_Blower",
                "Application_Fan",
                "Application_Crusher",
                "Application_Agitator",
                "Application_Pump",
            ]
        )

        # Test that no missing value exist
        # assert np.all(df.isnull().sum() == 0)
