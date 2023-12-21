import logging
import logging.config
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from joblib import dump
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@contextmanager
def timer(task: str = "Task"):
    """Logs how much time a code block takes

    Args:
        task (str, optional): Name of task, for logging purposes. Defaults to "Task".
    """
    start_time = time.time()
    yield
    logger.info(f"{task} completed in {time.time() - start_time:.5} seconds ---")


def create_timestamp() -> str:
    """Creates a string of current timestamp.

    Returns:
        str: current timestamp.
    """
    return str(datetime.today().strftime("%Y%m%d_%H%M%S"))


def format_omegaconf(args: DictConfig) -> dict:
    """Converts omegaConf DictConfig format to Dictionary.

    Args:
        args (DictConfig):  DictConfig config yml.

    Returns:
        dict: Dictionary format of args
    """
    return OmegaConf.to_container(args, resolve=False)


def export_artifact(
    artifact: Any,
    artifact_dir: Path,
    artifact_name: str,
):
    """Export model and saved them according to artifact_dir

    Args:
        artifact (Any): any python object to be exported as artifact
        artifact_dir (Path): Directory to place artifact details
        artifact_name (str): Name of artifact
    """
    dump(artifact, Path(Path(artifact_dir) / f"{artifact_name}.joblib"))
    logger.info("%s has been exported to %s", artifact_name, artifact_dir)


def setup_logging(
    logging_config_path="./conf/base/logging.yml", default_level=logging.INFO
):
    """Set up configuration for logging utilities.

    Args:
        logging_config_path (str, optional): Path to YAML file containing configuration for Python logger. Defaults to "./conf/base/logging.yml".
        default_level (_type_, optional): logging object. Defaults to logging.INFO.
    """
    try:
        with open(logging_config_path, "rt") as file:
            log_config = yaml.safe_load(file.read())
        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")


def mlflow_init(
    args: DictConfig, run_name: str, setup_mlflow: bool = False, autolog: bool = False
) -> Union[bool, Union[None, mlflow.entities.Run]]:
    """Initialise MLflow connection.

    Args:
        args (DictConfig): Dictionary containing the pipeline's configuration passed from Hydra.
        run_name (str): Run Name to be logged in MLflow
        setup_mlflow (bool, optional): Choice to set up MLflow connection. Defaults to False.
        autolog (bool, optional): Choice to set up MLflow's autolog. Defaults to False.

    Returns:
        Union[bool, Union[None, mlflow.entities.Run]]: Returns two items. First a boolean value indicative of success
        of intialising connection with MLflow server. Secondly a tuple containing an object containing the data and properties for the MLflow run.
        On failure, the function returns a null value.
    """
    init_success = False
    mlflow_run = None
    if setup_mlflow:

        # Create MLflow experiment
        mlflow.set_tracking_uri(args["train_pipeline"]["mlflow_tracking_uri"])

        try:
            mlflow.create_experiment(
                name=args["train_pipeline"]["mlflow_exp_name"],
                artifact_location=args["train_pipeline"]["mlflow_artifact_location"],
            )
        except:
            logger.warning(
                "Experiment %s has already been created",
                args["train_pipeline"]["mlflow_exp_name"],
            )

        try:
            mlflow.set_experiment(args["train_pipeline"]["mlflow_exp_name"])

            if autolog:
                mlflow.autolog()

            mlflow.start_run(run_name=run_name)

            if "MLFLOW_HPTUNING_TAG" in os.environ:
                mlflow.set_tag("hptuning_tag", os.environ.get("MLFLOW_HPTUNING_TAG"))

            mlflow_run = mlflow.active_run()
            init_success = True
            logger.info("MLflow initialisation has succeeded.")
            logger.info("UUID for MLflow run: {}".format(mlflow_run.info.run_id))
        except:
            logger.error("MLflow initialisation has failed.")

    return init_success, mlflow_run


def mlflow_log(mlflow_init_status: bool, log_function: str, **kwargs):
    """Custom function for utilising MLflow's logging functions.

    This function is only relevant when the function `mlflow_init`
    returns a "True" value, translating to a successful initialisation
    of a connection with an MLflow server.

    Args:
        mlflow_init_status (bool): Boolean value indicative of success of intialising connectionwith MLflow server.
        log_function (str): Name of MLflow logging function to be used. See https://www.mlflow.org/docs/latest/python_api/mlflow.html
    """
    if mlflow_init_status:
        try:
            method = getattr(mlflow, log_function)
            method(
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key in method.__code__.co_varnames
                }
            )
        except Exception as error:
            logger.error(error)


def mlflow_setup(timestamp: str, args: DictConfig):
    """Init MLflow and log configs to MLflow & GCSpip

    Args:
        timestamp (str): shared timestamp to reference current job
        args (DictConfig): Config file args (Hydra)
    """

    data_prep_cfg = args["data_prep"]
    modeling_cfg = args["modeling"]

    # Initialise MLflow
    mlflow_init_status, mlflow_run = mlflow_init(
        modeling_cfg,
        run_name=timestamp,
        setup_mlflow=modeling_cfg["train_pipeline"]["setup_mlflow"],
        autolog=modeling_cfg["train_pipeline"]["mlflow_autolog"],
    )

    mlflow_log(mlflow_init_status, "log_params", params=modeling_cfg["train_pipeline"])

    # Log saved model, metrics, configs and visuals to MLflow and GCSpip
    if mlflow_init_status:

        lstmae_params = format_omegaconf(modeling_cfg["lstmae"])
        model_params = lstmae_params.pop("model_params")
        mlflow.log_param("N_epochs", modeling_cfg["lstmae_training"]["n_epochs"])
        mlflow.log_params(model_params)
        mlflow.log_params(lstmae_params)
        mlflow.log_param(
            "outlier_threshold", data_prep_cfg["clean_data"]["outlier_threshold"]
        )
        mlflow.log_params(modeling_cfg["data_loader"]["init"])
        model_tags = format_omegaconf(modeling_cfg["mlflow_tags"])
        mlflow.set_tags(model_tags)

    return mlflow_init_status, mlflow_run


def mlflow_log_artifacts(
    args: DictConfig,
    metrics: dict,
    model_dir: Path,
    mlflow_run: mlflow.entities.Run,
):
    """Logs artifacts found in model_dir and logs metrics if applicable. Ends mlflow run.

    Args:
        args (DictConfig): Modeling Config file args (Hydra)
        metrics (dict): mean and standard deviation metrics of train, val, test directory
        model_dir (Path): model directory where model and other artifacts
            like scalers and encoders are stored.
        mlflow_run (mlflow.entities.Run): object containing the data and
            properties for the MLflow run.
    """
    if args["evaluation"]["to_run"]:
        mlflow.log_metrics(metrics)
    logger.info("Exporting the model...")
    artifact_uri = mlflow.get_artifact_uri()
    mlflow.log_param("artifact_uri", artifact_uri)
    logger.info("Artifact URI: %s", str(artifact_uri))
    logger.info(
        "Model training with MLflow run ID %s has completed.",
        str(mlflow_run.info.run_id),
    )
    mlflow.log_artifact(model_dir)
    mlflow.end_run()


def save_out_copy(copy_filepath: Path, save_filepath: Path):
    """Saves out a copy of a file from a given filepath into a specified save filepath.

    Args:
        copy_filepath (Path): Filepath of file to be copied
        save_filepath (Path): Filepath where copy is to be saved to
    """
    try:
        shutil.copyfile(copy_filepath, save_filepath)
        logger.info(
            "Copy of %s has been saved into %s.",
            copy_filepath,
            save_filepath,
        )
    except:
        logger.warning(
            "Unable to copy %s to %s.",
            copy_filepath,
            save_filepath,
        )


def resample_hourly(data: pd.DataFrame) -> pd.DataFrame:
    """Resamples the granularity to hourly for a given DataFrame, the DataFrame index must be in datetime format

    Args:
        data (pd.DataFrame): DataFrame to be resampled

    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    data = data.resample("H").mean()
    data.sort_index(inplace=True)
    return data


def split_long_nans(data: pd.DataFrame, nan_range_threshold: int = 120) -> list:
    """Checks a DataFrame for consecutive long ranges of NaNs and splits the DataFrame on them

    Args:
        data (pd.DataFrame): DataFrame to be split by long ranges of NaNs
        nan_range_threshold (int, optional): Threshold window for range of consecutive NaNs. Defaults to 120.

    Returns:
        list: List containing the split DataFrame
    """
    timestamp_df = data.copy()
    timestamp_df["group_no"] = timestamp_df.isna().all(axis=1).cumsum()

    split_condition = timestamp_df["group_no"].value_counts() > 1

    list_group_no = (
        timestamp_df["group_no"].value_counts()[split_condition].index.sort_values()
    )

    date_range_dict = _generate_date_range(timestamp_df, "group_no", list_group_no)

    timestamp_df.drop(columns="group_no", inplace=True)

    clean_dict = _check_nan_range(timestamp_df, date_range_dict, nan_range_threshold)

    list_dataframes = []

    for i in np.arange(len(clean_dict["start_date"])):
        if i == 0:
            list_dataframes.append(
                timestamp_df.loc[
                    clean_dict["start_date"][i] : clean_dict["end_date"][i]
                ]
            )

        else:
            list_dataframes.append(
                timestamp_df.loc[
                    clean_dict["start_date"][i]
                    + pd.to_timedelta(1, unit="h") : clean_dict["end_date"][i]
                ]
            )

    return list_dataframes


def _generate_date_range(data: pd.DataFrame, group_col: str, group_list: list) -> dict:
    """Creates dictionary containing date ranges for desired ranges of data

    Args:
        data (pd.DataFrame): DataFrame with ranges of data
        group_col (str): Column containing the labeled group numbers for each range of data
        group_list (list): List containg the group numbers for the desired ranges of data

    Returns:
        dict: Dictionary of date ranges for desired ranges of data
    """
    reference_date_dict = {"start_date": [], "end_date": []}

    for group_no in group_list:
        start_date = data[data[group_col] == group_no].index.min()
        end_date = data[data[group_col] == group_no].index.max()
        reference_date_dict["start_date"].append(start_date)
        reference_date_dict["end_date"].append(end_date)

    return reference_date_dict


def _check_nan_range(
    data: pd.DataFrame, date_range_dict: dict, nan_range_threshold: int = 120
) -> dict:
    """Calculates the length of NaN ranges and removes ranges which are below the specified threshold

    Args:
        data (pd.DataFrame): DataFrame with ranges of NaNs
        date_range_dict (dict): Dictionary of date ranges to be checked
        nan_range_threshold (int, optional): Threshold window for range of consecutive NaNs. Defaults to 120.

    Returns:
        dict: Dictionary of date ranges to be checked
    """
    # Calculate the length of the range of NaNs between the dates
    for i in np.arange(len(date_range_dict["start_date"]) - 1):
        length_of_nans = len(
            data.loc[
                date_range_dict["end_date"][i]
                + pd.to_timedelta(1, unit="h") : date_range_dict["start_date"][i + 1]
            ]
        )
        # From the given nan_range_threshold revise the dictionary of split dates
        # If it's lower than the threhold the NaNs are to be kept and imputed later on
        if length_of_nans < nan_range_threshold:
            date_range_dict["start_date"][i + 1] = None
            date_range_dict["end_date"][i] = None

    # Clean None values from revised_date_dict
    clean_dict = {"start_date": [], "end_date": []}

    for key in date_range_dict:
        for date in date_range_dict[key]:
            if date is not None:
                clean_dict[key].append(date)
    return clean_dict


def copy_logs(save_dir: Path, save_name: str = None):
    """Copies hydra logs from default outputs folder to specified folder and
    overwrites name if specified.

    Args:
        save_dir (Path): Directory in which to save the copied logs.
        save_name (str, optional): Overwrites file name if specified. Defaults to None.
    """
    try:
        log_filepath = Path(Path(os.getcwd()) / f"{HydraConfig.get().job.name}.log")
        save_name = save_name or HydraConfig.get().job.name
        save_filepath = Path(Path(save_dir) / f"{save_name}.log")
        save_out_copy(log_filepath, save_filepath)
    except:
        pass


def get_conf_dir() -> Path:
    """Retrieves config directory within which hydra config files are stored.

    Returns:
        Path: Config directory of config files
    """
    try:
        config_sources = HydraConfig.get().runtime.config_sources
    except:
        config_sources = []

    for source in config_sources:
        if source["provider"] == "main":
            return Path(source["path"])
    logger.info("No config directory found, using specified config dir.")
    return None


def save_out_config(
    processed_dir: Path, config_filename: str = "train_pipeline.yml"
) -> bool:
    """Saves out the config values used in Hydra run to the specified directory.

    Args:
        processed_dir (Path): Directory in which config file is intended to be saved.
        config_filename (str, optional): Name for config file to be saved under.
            Defaults to "train_pipeline.yml".

    Returns:
        bool: Whether Hydra's config.yml is successfully saved
    """
    config_saved_filepath = Path(Path(processed_dir) / config_filename)
    try:
        subdir = HydraConfig.get().output_subdir
        config_path = Path(Path(os.getcwd()) / subdir / "config.yaml")
        save_out_copy(config_path, config_saved_filepath)
    except:
        pass

    return os.path.exists(config_saved_filepath)


def get_asset_name(filepath: Path) -> str:
    """Retrieves asset name from filepath. Assumes file naming follows convention of
    '<asset_name>_YYYYMMDD-YYYYMMDD'

    Example:
    - Given "**/**/pump_001-YYYYMMDD-YYYYMMDD.xlsx", returns pump_001
    - Given "**/**/pump001-YYYYMMDD-YYYYMMDD.xlsx", returns pump001

    Args:
        filepath (Path): filepath of asset data
    Returns:
        str: asset_name from '<asset_name>_YYYYMMDD-YYYYMMDD'
    """
    return "_".join(Path(filepath).stem.split("_")[:-1])
