import json
import logging
import os
import warnings
from pathlib import Path
from typing import Tuple, Union

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.preprocessing import RobustScaler

from src.anomaly_predictor.modeling.data_loaders.data_loader import DataLoader
from src.anomaly_predictor.modeling.evaluation.eval_wrapper import EvaluationWrapper
from src.anomaly_predictor.modeling.models.isolation_forest import ABBIsolationForest
from src.anomaly_predictor.modeling.models.lstm_ae import LSTMAutoEncoder
from src.anomaly_predictor.modeling.utils import curve_shift
from src.anomaly_predictor.modeling.visualization import Visualizer
from src.anomaly_predictor.utils import (
    copy_logs,
    create_timestamp,
    export_artifact,
    format_omegaconf,
    get_asset_name,
    get_conf_dir,
    mlflow_log_artifacts,
    mlflow_setup,
    save_out_config,
    save_out_copy,
    timer,
)

# suppress warnings for now to monitor info logs.
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../conf", config_name="train_pipeline.yml")
def run_training(args: DictConfig) -> Tuple[float, float, float]:
    """Runs the training pipeline. Performs the following steps:
    1. Fit Model on training data
    2. Loop through training, validation and test partitions for prediction and evaluation.
    3. Save evaluation.csv
    3. Extract out top N and bottom N assets from evaluation dataframe for visualization
    4. Save visualized plots

    Args:
        args (DictConfig): Hydra config file

    Returns:
        Tuple[float, float, float, Path]: Tuple of model metrics FPR, Recall, F1_score and model directory
    """
    modeling_args = args["modeling"]
    try:
        logger.info("Starting Training Pipeline")
        # Create paths & timestamping folders
        timestamp = create_timestamp()
        mlflow_init_status, mlflow_run = mlflow_setup(timestamp, args)
        train_dir, model_dir, eval_dir = setup_logging_and_dir(modeling_args, timestamp)
        copy_logs(model_dir)

        # Create copies of data_prep and modeling conf files for reproducible workflow
        generate_local_copies(
            modeling_args["train_pipeline"]["conf_dir"],
            modeling_args["train_pipeline"]["processed_dir"],
            model_dir,
        )

        # get robust scalers args and use individual scaler if so specified else use global_scaler
        robust_scaler_args = format_omegaconf(
            modeling_args["data_loader"]["robust_scaler"]
        )
        robust_scaler_args["quantile_range"] = tuple(
            robust_scaler_args["quantile_range"]
        )

        if modeling_args["use_individual_scaler"]:
            scaler_dir = fit_individual_scalers(
                args["data_prep"]["split_data"]["by"],
                modeling_args,
                model_dir,
                robust_scaler_args,
                **modeling_args["col_names"],
            )
        else:
            scaler_dir = None

        # Init dataloader and load datasets
        data_loader_config = format_omegaconf(modeling_args["data_loader"])
        logger.info("Initialized Dataloader")
        model = modeling_args["train_pipeline"]["model_name"]
        return_mode = "pytorch" if model == "LSTMAE" else "sklearn"
        dataloader = DataLoader(
            return_mode=return_mode,
            **(modeling_args["data_loader"]["init"]),
            args=robust_scaler_args,
            scaler_dir=scaler_dir,
        )
        training_features, _ = dataloader.load_train_data(
            Path(train_dir),
            data_loader_config["feature_to_standardize"],
            shuffle=data_loader_config["shuffle"],
        )

        val_features = None
        if model == "LSTMAE":
            val_features, _ = dataloader.load_train_data(
                Path(Path(modeling_args["train_pipeline"]["processed_dir"]) / "val"),
                data_loader_config["feature_to_standardize"],
                fit_scaler=False,
            )

        logger.info(
            "%s/train is loaded.", modeling_args["train_pipeline"]["processed_dir"]
        )
        copy_logs(model_dir)

        # Exporting scaler from dataloader
        export_artifact(dataloader.scaler, model_dir, "RobustScaler")

        # Fit model and export model
        logger.info("Fitting %s", modeling_args["train_pipeline"]["model_name"])
        model, model_dir = train_model(
            modeling_args, training_features, val_features, model_dir
        )

        metrics = None
        if modeling_args["evaluation"]["to_run"]:
            feature_list = data_loader_config["feature_to_standardize"]
            evalwrapper = EvaluationWrapper(
                modeling_args, model, eval_dir, dataloader, feature_list, model_dir
            )
            # Create evaluation predictions for visualization & save them
            metrics = evalwrapper.predict_all_assets()
            copy_logs(model_dir)

            with open(Path(model_dir / "metrics.json"), "w") as filepath:
                json.dump(metrics, filepath)

        logger.info("Saving the model...")
        model.save_model(model_dir, modeling_args["train_pipeline"]["model_name"])
        copy_logs(model_dir)

        if mlflow_init_status:
            mlflow_log_artifacts(modeling_args, metrics, model_dir, mlflow_run)

        if modeling_args["evaluation"]["to_run"]:
            return (
                metrics["test_mean_Pointwise_FPR"],
                metrics["test_mean_Pointwise_Recall"],
                metrics["test_mean_Pointwise_F1_Score"],
                model_dir,
            )
        else:
            return 0, 0, 0, model_dir
    except Exception as e:
        logger.error(e, exc_info=True)
        copy_logs(model_dir)
        raise


def train_model(
    args: DictConfig,
    training_data: Union[pd.DataFrame, torch.utils.data.DataLoader],
    val_data: Union[pd.DataFrame, torch.utils.data.DataLoader],
    model_dir: Path,
) -> Tuple[Union[ABBIsolationForest, LSTMAutoEncoder], Path]:
    """Fit Model with training data and parameters from Config File.
    Saves Model and its parameters as well.

    Args:
        args (DictConfig): Config file args (Hydra)
        training_data (Union[pd.DataFrame, torch.utils.data.DataLoader]):
            Training Data DataFrame or pytorch dataloader
        val_data (Union[pd.DataFrame, torch.utils.data.DataLoader]):
            Validation Data DataFrame or pytorch dataloader
        model_dir (Path): Timestamped model directory

    Returns:
        Tuple[Union[ABBIsolationForest, LSTMAutoEncoder], str]: Returns Fitted
        Model and model_directory
    """
    if args["train_pipeline"]["model_name"] == "IsolationForest":
        model = ABBIsolationForest(args["isolation_forest"])
        model.fit(training_data)

    elif args["train_pipeline"]["model_name"] == "LSTMAE":
        n_features = next(iter(training_data)).size()[-1]
        lstmae_config = format_omegaconf(args["lstmae"])
        lstmae_config["model_params"]["n_features"] = n_features
        if "early_stopping_params" in lstmae_config.keys():
            path_name = lstmae_config["early_stopping_params"]["path"]
            lstmae_config["early_stopping_params"]["path"] = Path(model_dir / path_name)
        model = LSTMAutoEncoder(**lstmae_config)
        n_epochs = args["lstmae_training"]["n_epochs"]
        history = model.fit(
            training_data,
            val_data,
            loss_reduction="mean",
            n_epochs=n_epochs,
            model_dir=model_dir,
        )
        loss_visualizer = Visualizer()
        loss_visualizer.set_save_dir(model_dir)
        loss_visualizer.plot_epoch_losses(history)

    logger.info("%s is initialized and fitted.", args["train_pipeline"]["model_name"])
    return model, Path(model_dir)


def setup_logging_and_dir(args: DictConfig, timestamp: str) -> Tuple[Path, Path, Path]:
    """Given args from hydra config file, sets up timestamped folder in
    processed directory, along with train, val, test subfolders.

    Args:
        args (DictConfig): Hydra config as specified in yaml file
        timestamp (str): shared timestamp to reference current job

    Returns:
        Tuple[Path, Path, Path]: Tuple containing train directory, model directory
        and evaluation directory with timestamp where relevant.
    """
    # Create paths from imports
    processed_dir = Path(args["train_pipeline"]["processed_dir"])
    train_dir = Path(processed_dir / "train")
    model_dir = Path(args["train_pipeline"]["model_dir"])
    lookahead_period = args["data_loader"]["init"]["lookahead_period"]
    task = "detection" if lookahead_period == 0 else "forecast"
    model_dir = Path(model_dir / task)

    # Create timestamped folders
    model_dir = Path(model_dir / timestamp)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(model_dir / "evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    for partition in ["train", "test", "val"]:
        partition_dir = Path(eval_dir / partition)
        partition_dir.mkdir(parents=True, exist_ok=True)
        if args["evaluation"]["create_encoder_output_tsne"]:
            tsne_dir = Path(partition_dir / "tsne")
            tsne_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"The current working directory is {os.getcwd()}")
    logger.info("Defining processed dir, train dir, model dir, evaluation dir")
    logger.info(f"Processed dir is {processed_dir.resolve()}")
    logger.info(f"Training dir is {train_dir.resolve()}")
    logger.info(f"Model dir is {model_dir.resolve()}")
    logger.info(f"Evaluation dir is {eval_dir.resolve()}")
    logger.info(f"Refer to timestamped folder: {timestamp}")

    return train_dir, model_dir, eval_dir


def generate_local_copies(conf_dir: Path, processed_dir: Path, model_dir: Path):
    """Generates local copies of files used in data_prep pipeline and modeling pipeline

    Args:
        conf_dir (Path): Path to conf directory
        processed_dir (Path): Path to processed folder
        model_dir (Path): Path to current model timestamp folder
    """
    # Save out a copy of split_data.json, data_prep.yml, encoder.joblib to processed_dir
    for filename in ("split_data.json", "encoder.joblib"):
        filepath = Path(Path(processed_dir) / filename)
        save_out_copy(filepath, Path(model_dir / filename))

    saved_config = save_out_config(model_dir)
    if not saved_config:
        # Save out a copy of train_pipeline.yml to current timestamped folder
        conf_path = get_conf_dir() or Path(conf_dir)
        modeling_yml_path = Path(conf_path / "train_pipeline.yml")
        save_out_copy(modeling_yml_path, Path(model_dir / "train_pipeline.yml"))


def fit_individual_scalers(
    by: str,
    args: HydraConfig,
    model_dir: Path,
    scaler_args: dict,
    time_col: str = "MEASUREMENT_TAKEN_ON(UTC)",
    label_column: str = "Anomaly",
) -> Path:
    """Creates scaler folder in model_dir for indidual scalers

    Args:
        by (str): Expects values "asset" or "time".
        args (HydraConfig): Config file args (Hydra)
        model_dir (Path): Path to current model timestamp folder
        scaler_args(dict): arguments for initializing RobustScaler
        time_col (str, optional): Column containing date time index. Defaults to
            "MEASUREMENT_TAKEN_ON(UTC)".
        label_column (str, optional): Column containing Anomaly labels. Defaults to "Anomaly".

    Returns:
        Path: directory where individual scalers are stored.
    """
    scaler_dir = Path(model_dir / "scalers")
    scaler_dir.mkdir(parents=True, exist_ok=True)

    # get dictionary of various processed files
    if by == "asset":
        processed_dir = Path(args["train_pipeline"]["processed_dir"])
        asset_filepaths = processed_dir.glob("**/*.csv")
    elif by == "time":
        train_dir = Path(Path(args["train_pipeline"]["processed_dir"]) / "train")
        asset_filepaths = train_dir.glob("*.csv")

    assets = {}
    for asset_filepath in asset_filepaths:
        asset_name = get_asset_name(asset_filepath)

        if asset_name in assets.keys():
            assets[asset_name].append(asset_filepath)
        else:
            assets[asset_name] = [asset_filepath]

    # fit individual scalers
    lookahead_period = args["data_loader"]["init"]["lookahead_period"]
    features = format_omegaconf(args["data_loader"])["feature_to_standardize"]
    for asset_name, files in assets.items():
        scaler = RobustScaler(**scaler_args)
        main_data = None
        for file in files:
            data = pd.read_csv(
                file,
                parse_dates=[time_col],
                infer_datetime_format=True,
                index_col=0,
            )
            if lookahead_period > 0:
                data = curve_shift(data, label_column, lookahead_period)
            if main_data is not None:
                main_data = pd.concat([main_data, data], axis=0)
            else:
                main_data = data

        normal_data = main_data[main_data[label_column] == 0][features]
        scaler.fit(normal_data)
        export_artifact(scaler, scaler_dir, asset_name)
    logger.info(f"{len(assets)} scalers have been fitted and saved to {scaler_dir}.")
    return scaler_dir


if __name__ == "__main__":
    with timer("Running Training Pipeline"):
        run_training()
