# pylint: skip-file
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from src.anomaly_predictor.data_prep.feature_engineering import FeatureEngineer

application_file_list = [
    (Path("../data/interim/test/Agitator1001_20210701-20211116"), "Agitator"),
    (Path("../data/interim/test/Blower_9_20210701-20211122"), "Blower"),
    (Path("../data/interim/test/Fan - Cooling Tower Fan_3_20210701-20211122"), "Fan"),
    (Path("../data/interim/test/Fan-CoolingTowerFan_3_20210701-202111ß22"), "Fan"),
]

variable_speed_data = pd.DataFrame(
    np.round(np.random.randint(1000, size=1000), -2), columns=["Motor Supply Frequency"]
)
fixed_speed_data = pd.DataFrame(
    np.random.choice([0, 900], 1000), columns=["Motor Supply Frequency"]
)
zero_speed_data = pd.DataFrame(
    np.zeros(1000).astype(int), columns=["Motor Supply Frequency"]
)
speed_data_list = [
    (variable_speed_data["Motor Supply Frequency"], 1),
    (fixed_speed_data["Motor Supply Frequency"], 0),
    (zero_speed_data["Motor Supply Frequency"], 0),
]
application_set = {"Pump", "Fan", "Blower", "Crusher", "Agitator", "Pelletizer"}
test_path = Path("../data/interim/test/Agitator1001_20210701-20211116")


@pytest.fixture(scope="session")
def encoder_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("encoder"))

sample_statistical_data = pd.DataFrame([1,2,3,4,5,6,7,8,9],columns=['test'])
sample_abs_median_data = pd.DataFrame([[1,np.NaN],[2,np.NaN],[3,3.0],[4,2.0],[5,1.0],[6,0.0],[7,1.0],[8,2.0],[9,3.0]],columns=['test',"test_rolling_median_diff"])
sample_abs_iqr_data = pd.DataFrame([[1,np.NaN],[2,np.NaN],[3,3.0],[4,3.0],[5,3.0],[6,3.0],[7,3.0],[8,3.0],[9,3.0]],columns=['test',"test_rolling_iqr_diff"])


def test_fit_encoder():
    feature_engineer = FeatureEngineer()
    feature_engineer.fit_encoder(application_set)
    assert len(feature_engineer.encoder.get_feature_names_out()) == len(application_set)


def test_save_encoder(encoder_dir):
    feature_engineer = FeatureEngineer()
    encoder = feature_engineer.fit_encoder(application_set)
    feature_engineer.save_encoder(encoder_dir, "enc")
    assert len(glob(str(Path(encoder_dir / "*.joblib")))) == 1


def test_load_encoder(encoder_dir):
    feature_engineer = FeatureEngineer()
    path = Path(encoder_dir / "enc.joblib")
    feature_engineer.load_encoder(path)
    assert len(feature_engineer.encoder.get_feature_names_out()) == len(application_set)


def test_engineer_features():
    feature_engineer = FeatureEngineer()
    feature_engineer.fit_encoder(application_set)
    data = feature_engineer.engineer_features(variable_speed_data, test_path,"Motor Supply Frequency")
    assert data["Application_Agitator"].unique() == 1
    assert data["Application_Pump"].unique() == 0
    assert data["Application_Blower"].unique() == 0
    assert data["Application_Crusher"].unique() == 0
    assert data["Application_Fan"].unique() == 0
    assert data["Application_Pelletizer"].unique() == 0
    assert data["Variable_speed"].unique() == 1


def test_create_one_hot_application():
    feature_engineer = FeatureEngineer()
    feature_engineer.fit_encoder(application_set)
    data = feature_engineer.create_one_hot_encode_application(
        variable_speed_data, test_path
    )
    assert data.shape[1] == variable_speed_data.shape[1] + 6
    assert data["Application_Agitator"].unique() == 1
    for app in application_set:
        assert "Application_" + app in data.columns


@pytest.mark.parametrize("test_application,expected_application", application_file_list)
def test_extract_application(test_application, expected_application):
    assert (
        FeatureEngineer().extract_application(test_application) == expected_application
    )


@pytest.mark.parametrize("test_speed,expected_bool", speed_data_list)
def test_determine_variable_speed_bool(test_speed, expected_bool):
    assert FeatureEngineer()._determine_variable_speed_bool(test_speed) == expected_bool

def test_extract_abs_median_diff():
    assert (
        FeatureEngineer().extract_abs_median_diff(sample_statistical_data,3,['test']).round(2).equals(sample_abs_median_data)
    )

def test_extract_abs_iqr_diff():
    assert (
        FeatureEngineer().extract_abs_iqr_diff(sample_statistical_data,3,['test']).round(2).equals(sample_abs_iqr_data)
    )
