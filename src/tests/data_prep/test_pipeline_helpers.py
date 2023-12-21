# pylint: skip-file

import os
import shutil
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.data_prep.clean_data import DataCleaner
from src.anomaly_predictor.data_prep.feature_engineering import FeatureEngineer
from src.anomaly_predictor.data_prep.pipeline_helpers import (
    feature_engineer_df_list,
    get_clean_data_config,
    get_split_dict_from_json,
    run_annotation_ingestion,
    run_data_splitter,
    setup_logging_and_dir,
    update_split_dict,
)

CORRECT_ANOMALY_LABEL = 1
TIME_COL = "MEASUREMENT_TAKEN_ON(UTC)"
CUT_OFF_DATE = "01-07-21"
tr_timestamp = "20220303"
tr_timesplit_timestamp = "20220308"
inf_timestamp = "20220404"


@pytest.fixture(scope="session")
def input_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("input"))


@pytest.fixture(scope="session")
def annotation_xlsx(tmpdir_factory, input_dir):
    annotation_dir = Path(input_dir / "annotations")
    annotation_dir.mkdir()
    filepath = Path(annotation_dir / "annotation.xlsx")
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
    asset_dir = Path(input_dir / "assets")
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
                    df["data"] = df["data"].mask(np.random.random(len(df)) < 0.02)
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
def config_args(input_dir, annotation_xlsx, assets_dir, interim_dir, processed_dir):
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
        convert_to_int = ["Bearing Condition"]
        overrides = [
            f"data_prep.pipeline.input_dir={input_dir}",
            f"data_prep.pipeline.interim_dir={interim_dir}",
            f"data_prep.pipeline.processed_dir={processed_dir}",
            f"data_prep.ingest_data.assets_dir={assets_dir}",
            f"data_prep.ingest_data.annotation_list=[{annotation_xlsx}]",
            f"data_prep.split_data.follow_split=",
            f"data_prep.clean_data.features_to_standardize={std_features}",
            f"data_prep.clean_data.convert_to_int={convert_to_int}",
        ]
        return hydra.compose(config_name="train_pipeline.yml", overrides=overrides)


def test_setup_logging_and_dir(config_args):
    args = config_args["data_prep"]

    input_tr, interim_tr, processed_tr = setup_logging_and_dir(
        args, tr_timestamp, mode="training"
    )

    input_tr_ts, interim_tr_ts, processed_tr_ts = setup_logging_and_dir(
        args, tr_timesplit_timestamp, mode="training"
    )

    input_inf, interim_inf, processed_inf = setup_logging_and_dir(
        args, inf_timestamp, mode="inference"
    )

    # Check that return type is as expected
    for dir in (input_tr, interim_tr, processed_tr):
        assert isinstance(dir, Path)

    for dir in (input_inf, interim_inf, processed_inf):
        assert isinstance(dir, Path)

    # Check that returned processed directories exist:
    for dir in (processed_tr, processed_inf):
        assert dir.is_dir()

    # Check that train, val, test subfolders exist in training mode but not in inference mode
    for key in ("train", "val", "test"):
        assert Path(processed_tr / key).is_dir()
        assert not Path(processed_inf / key).is_dir()

    # Check that the processed folders and inference mode's interim folder is timestamp:
    assert interim_tr.stem != tr_timestamp
    assert processed_tr.stem == tr_timestamp

    for dir in (interim_inf, processed_inf):
        assert dir.stem == inf_timestamp


def test_run_annotation_ingestion(config_args, input_dir, interim_dir, assets_dir):
    args = config_args["data_prep"]
    # tests for training mode
    run_annotation_ingestion(
        args,
        Path(input_dir),
        Path(interim_dir / tr_timestamp),
        mode="training",
    )

    # Check that num of files in interim directory is same as in asset directory
    assert len(set(os.listdir(Path(interim_dir / tr_timestamp)))) == len(
        set(os.listdir(assets_dir))
    )

    # Check that "Anomaly" column is present in interim files during training mode
    df_fp = list(glob(str(Path(interim_dir / tr_timestamp / "*.csv"))))[0]
    df = pd.read_csv(
        df_fp, parse_dates=[TIME_COL], infer_datetime_format=True, index_col=TIME_COL
    )
    assert "Anomaly" in df.columns

    # tests for inference mode
    run_annotation_ingestion(
        args,
        Path(input_dir),
        Path(interim_dir / inf_timestamp),
        mode="inference",
    )

    # Check that num of files in interim directory is same as in asset directory
    assert len(set(os.listdir(Path(interim_dir / inf_timestamp)))) == len(
        set(os.listdir(assets_dir))
    )

    # Check that "Anomaly" column is not present in interim files during inference mode
    df_fp = list(glob(str(Path(interim_dir / inf_timestamp / "*.csv"))))[0]
    df = pd.read_csv(
        df_fp, parse_dates=[TIME_COL], infer_datetime_format=True, index_col=TIME_COL
    )
    assert "Anomaly" not in df.columns


@pytest.mark.parametrize(
    "split_by,timestamp", [("asset", tr_timestamp), ("time", tr_timesplit_timestamp)]
)
def test_run_data_splitter(
    config_args, interim_dir, processed_dir, split_by, timestamp
):
    # tests for this function are made under assumption it's training pipeline
    # since function is not used during inference mode

    if timestamp == tr_timesplit_timestamp:
        # copy the interim files from tr_timestamp since it was not previously ingested.
        shutil.copytree(
            Path(interim_dir / tr_timestamp), Path(interim_dir / tr_timesplit_timestamp)
        )
    args = config_args["data_prep"]
    args["split_data"]["by"] = split_by
    split_dict, apps = run_data_splitter(
        args,
        Path(interim_dir / timestamp),
        Path(processed_dir / timestamp),
    )
    assert config_args["data_prep"]["split_data"]["follow_split"] == ""

    assert isinstance(split_dict, dict)
    assert set(split_dict.keys()) == {"train", "val", "test"}
    assert isinstance(apps, (set, list))
    assert set(apps) == set(("Blower", "Agitator", "Fan", "Pump", "Crusher"))


@pytest.mark.parametrize("timestamp", [(tr_timestamp), (tr_timesplit_timestamp)])
def test_get_split_dict_from_json(processed_dir, timestamp):
    # tests for this function are made under assumption it's training pipeline
    # since function is not used during inference mode
    tr_timestamp2 = "20220505"
    for key in ("train", "val", "test"):
        Path(processed_dir / tr_timestamp2 / key).mkdir(parents=True, exist_ok=True)

    follow_split_json = Path(processed_dir / timestamp / "split_data.json")
    split_dict, apps = get_split_dict_from_json(
        Path(follow_split_json),
        Path(processed_dir / tr_timestamp2),
    )

    # Check that split_data.json is copied into new processed_dir
    assert Path(processed_dir / tr_timestamp2 / "split_data.json").is_file()

    # Check that the files in second processed directory is same as first
    # it should be the same since all other params for data preprocessing are the same.
    for key in ("train", "val", "test"):
        assert set(os.listdir(Path(processed_dir / tr_timestamp2 / key))) == set(
            os.listdir(Path(processed_dir / timestamp / key))
        )


def test_update_split_dict():
    interim_split_dict = {
        "train": ["/data/interim/old_timestamp/Pump8_20210701-20211231.csv"],
        "val": ["/data/interim/old_timestamp/Pump25_20210701-20211231.csv"],
        "test": ["/data/interim/old_timestamp/Pump33_20210701-20211231.csv"],
    }
    interim_dir = Path("/data/interim/new_timestamp")
    updated_split_dict = update_split_dict(interim_split_dict, interim_dir)

    # Check that func returns dict of lists
    assert isinstance(updated_split_dict, dict)
    assert isinstance(updated_split_dict["train"], list)

    # Check that timestamp has been updated
    assert Path(updated_split_dict["test"][0]).parent.stem == interim_dir.stem


def test_feature_engineer_df_list(config_args, interim_dir, processed_dir):
    # will choose a certain csv file in processed dir from prev tests to test this
    filepath = list(glob(str(Path(interim_dir / inf_timestamp / "*.csv"))))[-5]
    data = pd.read_csv(
        filepath, parse_dates=[TIME_COL], infer_datetime_format=True, index_col=TIME_COL
    )
    fe = FeatureEngineer()
    fe.fit_encoder(set(("Blower", "Agitator", "Fan", "Pump", "Crusher")))
    cd_config = get_clean_data_config(config_args["data_prep"])
    cleaned_df_list = DataCleaner().clean_data(data, **cd_config)
    output_path = feature_engineer_df_list(
        cleaned_df_list,
        config_args["data_prep"],
        Path(filepath),
        fe,
        Path(processed_dir / inf_timestamp),
    )

    # Check that there is indeed a file saved in processed_dir
    assert len(glob(str(Path(processed_dir / inf_timestamp / "*.csv")))) == 1


def test_get_clean_data_config(config_args):
    cd_config = get_clean_data_config(config_args["data_prep"])

    # assert return type is as expected
    assert isinstance(cd_config, dict)
    assert isinstance(cd_config["scaler_args"]["quantile_range"], tuple)
