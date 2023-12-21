# pylint: skip-file

import json
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from joblib import dump
from omegaconf import OmegaConf
from sklearn.preprocessing import RobustScaler

from src.anomaly_predictor.modeling.data_loaders.data_loader import DataLoader
from src.anomaly_predictor.modeling.inference_pipeline import (
    predict_and_postprocess,
    run_inference,
)
from src.anomaly_predictor.modeling.models.isolation_forest import ABBIsolationForest
from src.anomaly_predictor.modeling.models.lstm_ae import LSTMAutoEncoder
from src.anomaly_predictor.utils import format_omegaconf


@pytest.fixture(scope="session")
def process_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("process"))


@pytest.fixture(scope="session")
def evaluation_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("evaluation"))


@pytest.fixture(scope="session")
def prediction_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("prediction"))


@pytest.fixture(scope="session")
def timestamp_dir(process_dir):
    timestamp_dir = Path(os.sep.join([process_dir, "20220222_000000"]))
    timestamp_dir.mkdir()
    return str(timestamp_dir)


@pytest.fixture(scope="session")
def model_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("model"))


@pytest.fixture(scope="session")
def training_csv():
    col = [
        "Peak to Peak (X)",
        "Motor Supply Frequency",
    ]
    train_data = pd.DataFrame(
        np.random.randint(100, size=(1000, 2)),
        columns=col,
    )

    date_range = pd.date_range("2021-07-01", periods=1000, freq="H")
    train_data.set_index(date_range, inplace=True)
    train_data.index.name = "MEASUREMENT_TAKEN_ON(UTC)"
    return train_data


@pytest.fixture(scope="session")
def testing_csv(timestamp_dir):
    np.random.seed(8642)
    testing_dir = Path(os.sep.join([timestamp_dir, "testing"]))
    testing_dir.mkdir()
    date_range = pd.date_range("2021-07-01", periods=100, freq="H")
    test_col = [
        "Peak to Peak (X)",
        "Motor Supply Frequency",
    ]

    for i in range(3):
        test_data = pd.DataFrame(
            np.random.randint(100, size=(100, 2)),
            columns=test_col,
        )
        test_data.set_index(date_range, inplace=True)
        test_data.index.name = "MEASUREMENT_TAKEN_ON(UTC)"
        test_data["Asset_Operating"] = 1
        test_data.to_csv(str(testing_dir) + "/" + str(i) + ".csv")
    return str(testing_dir)


@pytest.fixture(scope="session")
def create_model(training_csv, model_dir):
    model = ABBIsolationForest()
    model.fit(training_csv)
    model_name = "IsolationForest"
    model_path = Path(Path(model_dir) / f"{model_name}.joblib")
    dump(model, model_path)
    return model_name


@pytest.fixture(scope="session")
def create_scaler(training_csv, model_dir):
    scaler = RobustScaler()
    scaler.fit(training_csv)
    scaler_name = "RobustScaler"
    scaler_path = Path(Path(model_dir) / f"{scaler_name}.joblib")
    dump(scaler, scaler_path)
    return scaler_name

@pytest.fixture(scope="session")
def correct_time():
    return pd.Timestamp("2021-07-10T00")

@pytest.fixture(scope="session")
def process_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("process"))


@pytest.fixture(scope="session")
def timestamp_dir(process_dir):
    timestamp_dir = Path(os.sep.join([process_dir, "20220222_000000"]))
    timestamp_dir.mkdir()
    return str(timestamp_dir)

@pytest.fixture(scope="session")
def dummy_csv(timestamp_dir):
    np.random.seed(8642)

    date_range = pd.date_range("2021-07-01", periods=100, freq="H")
    test_col = [
        "Peak to Peak (X)",
        "Motor Supply Frequency",
        "Anomaly",
    ]

    for partition in ["train", "val", "test"]:
        directory = Path(os.sep.join([timestamp_dir, partition]))
        directory.mkdir()
        for i in range(3):
            test_data = pd.DataFrame(
                np.random.randint(100, size=(100, 3)),
                columns=test_col,
            )
            test_data["Asset_Operating"] = 1
            test_data.set_index(date_range, inplace=True)
            test_data.index.name = "MEASUREMENT_TAKEN_ON(UTC)"
            test_data["Anomaly"] = 0
            test_data.loc[80:84, "Anomaly"] = 1
            test_data.to_csv(str(directory) + "/" + str(i) + ".csv")

    dummy_split_dict = {
        "train": ["1", "2", "3"],
        "test": ["1", "2", "3"],
        "val": ["1", "2", "3"],
    }
    with open(Path(Path(timestamp_dir) / "split_data.json"), "w") as filepath:
        json.dump(dummy_split_dict, filepath)

    return str(timestamp_dir)


@pytest.fixture(scope="session")
def lstm_args(dummy_csv,model_dir):
    try:
        hydra.initialize("../../../conf")
    except:
        pass

    std_features = [
        "Peak to Peak (X)",
        "Motor Supply Frequency",
    ]
    plot_features = ["Peak to Peak (X)"]
    overrides = [
        f"modeling.train_pipeline.model_name=LSTMAE",
        f"modeling.train_pipeline.processed_dir={dummy_csv}",
        f"modeling.train_pipeline.model_dir={model_dir}",
        f"modeling.data_loader.feature_to_standardize={std_features}",
        f"modeling.data_loader.init.lookback_period={5}",
        f"modeling.evaluation.plotting_features={plot_features}",
        f"modeling.data_loader.init.batch_size={10}",
        f"modeling.lstmae_training.n_epochs={2}",
        f"modeling.data_loader.init.pin_memory={False}",
        f"modeling.data_loader.init.num_workers={0}",
        f"modeling.data_loader.init.statistical_window={2}",
        f"modeling.data_loader.init.lookahead_period={5}",
        f"modeling.train_pipeline.setup_mlflow={False}",
        f"modeling.train_pipeline.mlflow_autolog={False}",  
    ]
    return hydra.compose(config_name="train_pipeline.yml", overrides=overrides)


@pytest.fixture(scope="session")
def fit_lstm_and_scaler(lstm_args,model_dir):
    args = lstm_args["modeling"]
    dataloader = DataLoader(
        return_mode="pytorch", batch_size=168, lookback_period=5, lookahead_period=0,statistical_window=0,pin_memory=False,num_workers=0,
    )
    data_loader_config = args["data_loader"]
    data_loader_config = OmegaConf.to_container(data_loader_config, resolve=False)
    feature_data, _ = dataloader.load_train_data(
        Path(args["train_pipeline"]["processed_dir"]) / "train",
        data_loader_config["feature_to_standardize"],
    )
    val_data, _ = dataloader.load_train_data(
        Path(args["train_pipeline"]["processed_dir"]) / "val",
        data_loader_config["feature_to_standardize"],
        fit_scaler=False,
    )
    n_features = 3
    lstmae_config = format_omegaconf(args["lstmae"])
    lstmae_config["model_params"]["n_features"] = n_features
    
    model = LSTMAutoEncoder(**lstmae_config)
    training_features = feature_data
    model.fit(
        training_features,
        val_data,
    )
    model.save_model(model_dir, "LSTMAE")
    with open(Path(Path(model_dir) / "model_params.json"), "w") as filepath:
            json.dump(model.get_params(), filepath)
    #print (Path(model_dir).glob())
    return model,model_dir

@pytest.fixture(scope="session")
def inference_args(testing_csv,prediction_dir,create_scaler,fit_lstm_and_scaler):
    try:
        hydra.initialize("../../../conf")
    except:
        pass
    std_features = [
        "Peak to Peak (X)",
        "Motor Supply Frequency",
    ]
    plot_features = ["Peak to Peak (X)"]
    lookahead_period = 10
    overrides = [
        f"modeling.inference_pipeline.inference_dir={testing_csv}",
        f"modeling.inference_pipeline.prediction_dir={prediction_dir}",
        f"modeling.data_loader.feature_to_standardize={std_features}",
        f"modeling.data_loader.init.statistical_window={0}",
        f"modeling.data_loader.init.lookback_period={5}",
        f"modeling.data_loader.init.pin_memory={False}",
        f"modeling.data_loader.init.num_workers={0}",
        f"modeling.data_loader.init.lookahead_period={lookahead_period}",
        f"modeling.visualizations.plotting_features={plot_features}",
        f"artifacts.model_dir={fit_lstm_and_scaler[1]}",
        f"artifacts.model_name=LSTMAE",
        f"artifacts.scaler_name={create_scaler}",
    ]
    return hydra.compose(config_name="inference_pipeline.yml", overrides=overrides)

@pytest.fixture(scope="session")
def load_lstm(inference_args):
    artifact_args = inference_args["artifacts"]
    model_dir = Path(artifact_args["model_dir"])
    model_params = json.load(open(Path(model_dir / "model_params.json")))
    model_architecture = model_params["model_architecture"]
    del model_params["model_architecture"]
    model = LSTMAutoEncoder(model_params=model_architecture, **model_params,)
    model.load_model(Path(model_dir / f"LSTMAE.pt"))
    return model

def test_run_inference(inference_args, prediction_dir):
    run_inference(inference_args)
    folder = prediction_dir + "/" + os.listdir(prediction_dir)[0]
    for folder_name in ["0", "1", "2"]:
        assert folder_name in os.listdir(folder+"/")

def test_predict_and_postprocess(inference_args,fit_lstm_and_scaler,prediction_dir):
    model = fit_lstm_and_scaler[0]
    artifact_args = inference_args["artifacts"]
    args = inference_args["modeling"]
    model_dir = Path(artifact_args["model_dir"])
    binarizing_threshold = 0.5
    data_loader_config = format_omegaconf(args["data_loader"])
    data_loader = DataLoader(return_mode="pytorch", **(args["data_loader"]["init"]))
    data_loader.load_scaler(Path(model_dir / f'{artifact_args["scaler_name"]}.joblib'))
    expected_fp = Path(args["inference_pipeline"]["inference_dir"])
    files = expected_fp.glob("**/*.csv")
    for file in files:
        inference_data, original_data = data_loader.load_inference_data(
            file, data_loader_config["feature_to_standardize"], drop_last=False
        )
        postprocessed_data,postprocessed_daily = predict_and_postprocess(
            args,
            model,
            inference_data,
            original_data,
            file,
            Path(prediction_dir),
            binarizing_threshold
        )
        assert isinstance(postprocessed_data,pd.DataFrame)
        assert isinstance(postprocessed_daily,pd.DataFrame)
