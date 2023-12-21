# pylint: skip-file
import json
import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.anomaly_predictor.modeling.data_loaders.data_loader import DataLoader
from src.anomaly_predictor.modeling.models.isolation_forest import ABBIsolationForest
from src.anomaly_predictor.modeling.models.lstm_ae import LSTMAutoEncoder
from src.anomaly_predictor.modeling.train_pipeline import (
    run_training,
    setup_logging_and_dir,
    train_model,
)
from src.anomaly_predictor.utils import format_omegaconf

## Need create overall dir then add the other created folders under


@pytest.fixture(scope="session")
def process_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("process"))


@pytest.fixture(scope="session")
def timestamp_dir(process_dir):
    timestamp_dir = Path(os.sep.join([process_dir, "20220222_000000"]))
    timestamp_dir.mkdir()
    return str(timestamp_dir)


@pytest.fixture(scope="session")
def model_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("models"))


@pytest.fixture(scope="session")
def forecast_dir(model_dir):
    forecast_dir = Path(os.sep.join([model_dir, "forecast"]))
    forecast_dir.mkdir(exist_ok=True)
    return str(forecast_dir)


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
def config_args(dummy_csv, model_dir):
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
        f"modeling.train_pipeline.model_name=IsolationForest",
        f"modeling.train_pipeline.processed_dir={dummy_csv}",
        f"modeling.train_pipeline.model_dir={model_dir}",
        f"modeling.data_loader.feature_to_standardize={std_features}",
        f"modeling.evaluation.plotting_features={plot_features}",
        f"modeling.data_loader.init.lookahead_period={5}",
        f"modeling.train_pipeline.setup_mlflow={False}",
        f"modeling.train_pipeline.mlflow_autolog={False}",
    ]
    return hydra.compose(config_name="train_pipeline.yml", overrides=overrides)


@pytest.fixture(scope="session")
def dummy_training_data(config_args):
    args = config_args["modeling"]
    dataloader = DataLoader(batch_size=10, lookback_period=5, lookahead_period=0)
    data_loader_config = args["data_loader"]
    data_loader_config = OmegaConf.to_container(data_loader_config, resolve=False)
    feature_data, label_data = dataloader.load_train_data(
        Path(args["train_pipeline"]["processed_dir"]) / "train",
        data_loader_config["feature_to_standardize"],
    )
    return feature_data, label_data


@pytest.fixture(scope="session")
def fit_model_and_scaler(config_args):
    args = config_args["modeling"]
    dataloader = DataLoader(batch_size=10, lookback_period=5, lookahead_period=0)
    data_loader_config = args["data_loader"]
    data_loader_config = OmegaConf.to_container(data_loader_config, resolve=False)
    feature_data, label_data = dataloader.load_train_data(
        Path(args["train_pipeline"]["processed_dir"]) / "train",
        data_loader_config["feature_to_standardize"],
    )
    model = ABBIsolationForest(args["isolation_forest"])
    training_features = feature_data
    model.fit(training_features)
    return model, dataloader


@pytest.fixture(scope="session")
def lstm_args(dummy_csv, model_dir):
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
def lstm_training_data(lstm_args):
    args = lstm_args["modeling"]
    dataloader = DataLoader(
        return_mode="pytorch",
        batch_size=168,
        lookback_period=5,
        lookahead_period=0,
        statistical_window=2,
        pin_memory=False,
        num_workers=0,
    )
    data_loader_config = args["data_loader"]
    data_loader_config = OmegaConf.to_container(data_loader_config, resolve=False)
    feature_data, label_data = dataloader.load_train_data(
        Path(args["train_pipeline"]["processed_dir"]) / "train",
        data_loader_config["feature_to_standardize"],
    )
    return feature_data, label_data


@pytest.fixture(scope="session")
def fit_lstm_and_scaler(lstm_args):
    args = lstm_args["modeling"]
    dataloader = DataLoader(
        return_mode="pytorch", batch_size=168, lookback_period=5, lookahead_period=0
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
    n_features = 2
    lstmae_config = format_omegaconf(args["lstmae"])
    lstmae_config["model_params"]["n_features"] = n_features
    model = LSTMAutoEncoder(**lstmae_config)
    training_features = feature_data
    model.fit(
        training_features,
        val_data,
    )
    return model, dataloader


@pytest.fixture(scope="session")
def evaluation_dictionary():
    evaluation_dictionary = {}
    for partition in ["train", "val", "test"]:
        colnames = [
            "Asset_Name",
            "Pointwise_Recall",
            "Pointwise_F1_Score",
            "Pointwise_FPR",
            "Overlap_Recall",
            "Overlap_F1_Score",
            "Overlap_FPR",
        ]
        data = pd.DataFrame(np.random.randint(100, size=(100, 4)), columns=colnames)
        evaluation_dictionary[partition] = data
    return evaluation_dictionary


@pytest.mark.parametrize("arg_file", ["config_args", "lstm_args"])
def test_run_training(arg_file, dummy_csv, request):
    args = request.getfixturevalue(arg_file)
    assert args["modeling"]["train_pipeline"]["processed_dir"] == dummy_csv
    test_metrics = run_training(args)
    assert isinstance(test_metrics, tuple)
    assert len(test_metrics) == 4
    for metric in test_metrics[:3]:
        assert isinstance(metric, float)


@pytest.mark.parametrize(
    "arg_file,training_data,modeltype",
    [
        ("config_args", "dummy_training_data", ABBIsolationForest),
        ("lstm_args", "lstm_training_data", LSTMAutoEncoder),
    ],
)
def test_train_model(arg_file, training_data, modeltype, forecast_dir, request):
    model_dir = glob(str(Path(Path(forecast_dir) / "*")))[0]
    model, model_path = train_model(
        request.getfixturevalue(arg_file)["modeling"],
        request.getfixturevalue(training_data)[0],
        request.getfixturevalue(training_data)[0],
        model_dir=Path(model_dir),
    )
    assert isinstance(model, modeltype)
    assert isinstance(model_path, Path)


def test_setup_logging_and_dir(config_args):
    modeling_conf = config_args["modeling"]
    conf_train = Path(Path(modeling_conf["train_pipeline"]["processed_dir"]) / "train")
    conf_model = modeling_conf["train_pipeline"]["model_dir"]
    conf_model_forecast = Path(Path(conf_model) / "forecast")
    test_timestamp_forecast = Path(Path(conf_model) / "forecast" / "123")
    test_timestamp_eval = Path(test_timestamp_forecast / "evaluation")
    train_dir, model_dir, eval_dir = setup_logging_and_dir(modeling_conf, "123")
    assert conf_train == train_dir
    assert test_timestamp_forecast == model_dir
    assert test_timestamp_eval == eval_dir
