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
from joblib import dump
from sklearn.preprocessing import OneHotEncoder

from src.anomaly_predictor.data_prep.inference_pipeline import run_pipeline

CORRECT_ANOMALY_LABEL = 1
TIME_COL = "MEASUREMENT_TAKEN_ON(UTC)"
CUT_OFF_DATE = "01-09-21"


@pytest.fixture(scope="session")
def input_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("input"))


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
        "new_app": 10,
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
            filepath = Path(asset_subdir / f"{asset_name}_20210901-202101029.xlsx")
            time_range = pd.date_range("2021-09-01", periods=216, freq="H")
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
    enc_dir = Path(tmpdir_factory.mktemp("encoder"))
    encoder = OneHotEncoder(handle_unknown="ignore")
    applications = [
        "Agitator",
        "Blower",
        "Crusher",
        "Fan",
        "Pelletizer",
        "Pump",
    ]
    encoder.fit(pd.DataFrame(applications, columns=["Application"]))
    dump(encoder, Path(enc_dir / "enc.joblib"))
    return enc_dir


@pytest.fixture(scope="session")
def config_args(input_dir, assets_dir, interim_dir, processed_dir, encoder_dir):
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
        convert_to_int = ["Bearing Condition"]
        overrides = [
            f"data_prep.pipeline.input_dir={input_dir}",
            f"data_prep.pipeline.interim_dir={interim_dir}",
            f"data_prep.pipeline.processed_dir={processed_dir}",
            f"data_prep.ingest_data.assets_dir={assets_dir}",
            f"data_prep.ingest_data.annotation_list=[]",
            f"data_prep.clean_data.features_to_standardize={std_features}",
            f"data_prep.clean_data.convert_to_int={convert_to_int}",
            f"artifacts.model_dir={encoder_dir}",
            f"artifacts.encoder_name=enc",
        ]
        return hydra.compose(config_name="inference_pipeline.yml", overrides=overrides)


def test_run_pipeline(config_args, processed_dir, input_dir, interim_dir, encoder_dir):
    assert config_args["data_prep"]["pipeline"]["input_dir"] == input_dir
    
    paths, _ = run_pipeline(config_args)

    # Check that the output is as intended
    assert isinstance(paths, list)

    # Check that the unique number of assets in processed tallies with number of assets in input folder
    assert len(glob(str(Path(processed_dir / "*" / "*")))) == len(
        glob(str(Path(interim_dir / "*" / "*")))
    )

    # check that the filepaths returned are indeed found in output_dir
    assert set(glob(str(Path(processed_dir / "*" / "*")))) == set(paths)

    filepath = paths[0]
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
            "Asset_Operating",
        ]
    )

    # Test that no missing value exist
    assert np.all(df.isnull().sum() == 0)
