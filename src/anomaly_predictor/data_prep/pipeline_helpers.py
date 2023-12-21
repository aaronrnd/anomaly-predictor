import json
import logging
import os
import re
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import DictConfig

from src.anomaly_predictor.data_prep.feature_engineering import FeatureEngineer
from src.anomaly_predictor.data_prep.ingest_data import AnnotationIngestor
from src.anomaly_predictor.data_prep.split_data import DataSplitter
from src.anomaly_predictor.utils import format_omegaconf, get_asset_name

logger = logging.getLogger(__name__)


def setup_logging_and_dir(args: DictConfig, timestamp: str, mode: str) -> tuple:
    """Given args from hydra config file, sets up timestamped folder in
    processed directory. If mode is training, train, val, test subfolders too.

    Args:
        args (DictConfig): Hydra config as specified in yaml file
        timestamp (str): shared timestamp to reference current job.
        mode (str): Accepts values "training" or "inference".

    Returns:
        Tuple: A tuple (Path, Path, Path) representing input directory, interim directory
        and processed directory with timestamp. Note that if mode is training,
        interim directory is not pointing to a timestamped folder because such a
        folder would not be generated if path to split_data.json was
        provided. If mode is inference, interim directory is with timestamp.
    """
    input_dir = Path(args["pipeline"]["input_dir"])
    interim_dir = Path(args["pipeline"]["interim_dir"])
    processed_dir = Path(args["pipeline"]["processed_dir"])
    Path(processed_dir / timestamp).mkdir(parents=True, exist_ok=True)

    logger.info(f"The current working directory is {os.getcwd()}")
    logger.info("Defining input, interim, processed dir")
    logger.info(f"Input dir is {input_dir.resolve()}")
    logger.info(f"Interim dir is {interim_dir.resolve()}")
    logger.info(f"Processed dir is {processed_dir.resolve()}")

    processed_dir = Path(processed_dir / timestamp)
    logger.info(f"Refer to timestamped folder: {timestamp}")

    if mode == "training":
        for key in ("train", "val", "test"):
            Path(processed_dir / key).mkdir(parents=True, exist_ok=True)
    elif mode == "inference":
        interim_dir = Path(interim_dir / timestamp)

    return input_dir, interim_dir, processed_dir


def run_annotation_ingestion(
    args: DictConfig, input_dir: Path, interim_dir: Path, mode: str
) -> None:
    """Does necessary treatment of 'ingest_data' values in hydra config. Ingests
    asset and annotation xlsx files from input_dir and saves csv files to interim_dir.

    Args:
        args (DictConfig): Hydra config as specified in yaml file.
        input_dir (Path): Input directory with annotations and assets xlsx files.
        interim_dir (Path): Interim directory to save the asset csv files.
        mode (str):  Accepts values "training" or "inference" to be passed to AnnotationIngestor.
    """
    logger.info("Prefixing input_dir path for assets_dir and annotation_list")
    assets_dir = Path(input_dir / args["ingest_data"]["assets_dir"])
    annotation_list = [
        Path(input_dir / file) for file in args["ingest_data"]["annotation_list"]
    ]
    offset_hours = (
        args["ingest_data"]["asset_timezone"]
        - args["ingest_data"]["annotation_timezone"]
    )
    paths = AnnotationIngestor(mode=mode).ingest_data_annotations(
        annotation_list=annotation_list,
        assets_dir=assets_dir,
        time_col=args["ingest_data"]["time_col"],
        cut_off_date=args["ingest_data"]["cut_off_date"],
        resulting_dir=interim_dir,
        offset_hours=offset_hours,
    )
    logger.info(f"Assets ingested: {len(paths)}")


def run_data_splitter(
    args: DictConfig, interim_dir: Path, processed_dir: Path
) -> tuple:
    """Given a directory of asset csv files, splits assets into train, val,
    test sets and generates a dictionary with values being a list of assets'
    respective filepaths. Saves the dictionary as json file to processed
    directory of current timestamped job.

    Args:
        args (DictConfig): Hydra config as specified in yaml file.
        interim_dir (Path): Directory where asset raw values are stored as csv files.
        processed_dir (Path): The timestamped processed directory to save split_data.json.

    Returns:
        Tuple: A tuple (dict, Set[str]) where dict has keys 'train', 'val', 'test' and
        value being list of filepaths and set contains the unique values of
        Applications seen when splitting files.
    """

    if args["split_data"]["by"] == "asset":
        interim_split_dict, applications = DataSplitter().split_files(
            interim_dir,
            args["split_data"]["test_size"],
            args["split_data"]["random_state"],
        )
    elif args["split_data"]["by"] == "time":
        interim_split_dict, applications = DataSplitter().split_by_time(
            interim_dir,
            args["split_data"]["test_size"],
            args["ingest_data"]["time_col"],
        )
    logger.info("Data splitting completed")

    logger.info(
        f"Saving split dictionary to: {Path(processed_dir / 'split_data.json')}"
    )
    with open(Path(processed_dir / "split_data.json"), "w") as filepath:
        json.dump(interim_split_dict, filepath)

    return interim_split_dict, applications


def get_split_dict_from_json(follow_split_json: Path, processed_dir: Path) -> tuple:
    """Loads JSON file from follow_split as Python dictionary to be
    returned. Also returns set of applications extracted from said dictionary.
    Saves follow_split JSON to processed directory of current timestamped job.

    Args:
        follow_split_json (Path): path pointing to json file that describes a
            train val test split.
        processed_dir (Path): The timestamped processed directory to save split_data.json.

    Returns:
        Tuple: A tuple (dict, Set[str]) where dict has keys 'train', 'val', 'test' and
        value being list of filepaths and set contains the unique values of
        Applications from split files.
    """
    with open(follow_split_json, "r") as filepath:
        interim_split_dict = json.load(filepath)

    logger.info(
        f"Saving split dictionary to: {Path(processed_dir / 'split_data.json')}"
    )
    with open(Path(processed_dir / "split_data.json"), "w") as filepath:
        json.dump(interim_split_dict, filepath)

    all_csv_paths = []
    for key in interim_split_dict:
        all_csv_paths.extend(interim_split_dict[key])

    all_applications = []
    for path in all_csv_paths:
        interim_filename = Path(path).stem
        application_name = "".join(re.sub(r"[^a-zA-Z]", "", interim_filename))
        all_applications.append(application_name)

    return interim_split_dict, set(all_applications)


def update_split_dict(interim_split_dict: dict, interim_dir: Path) -> dict:
    """
    Takes in a data split dictionary from a previous training run, and updates its
    timestamped paths with timestamps from the current interim directory.

    Args:
        interim_split_dict (dict): Data split dictionary with 'train', 'val', 'test' and
            value being list of timestamped interim filepaths to be updated.
        interim_dir (Path): New timestamped interim directory.

    Returns:
        dict: A Python dictionary with keys 'train', 'val', 'test' and value being
        list of updated interim filepaths.
    """
    new_timestamp = interim_dir.stem

    for partition, path_list in interim_split_dict.items():
        updated_path_list = []
        for path in path_list:
            old_timestamp = Path(path).parent.stem
            updated_path_list.append(path.replace(old_timestamp, new_timestamp))
        interim_split_dict[partition] = updated_path_list

    return interim_split_dict


def feature_engineer_df_list(
    cleaned_df_list: List[pd.DataFrame],
    args: DictConfig,
    path: Path,
    feature_engineer: FeatureEngineer,
    output_dir: Path,
) -> Path:
    """Loops through list of cleaned dataframes and writes engineered features.

    Args:
        cleaned_df_list (List[pd.DataFrame]): list of cleaned dataframes after
            splitting long consecutive missing timeframes.
        args (DictConfig): Hydra config as specified in yaml file.
        path (Path): the original path.
        feature_engineer (FeatureEngineer): custom FeatureEngineer object.
        output_dir (Path): the directory path to save the final dataframe.

    Returns:
        Path: the path in which the dataframe is saved.
    """
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    has_nonempty_dataframe = False
    for cleaned_df in cleaned_df_list:
        if cleaned_df.shape[0] > 0:
            has_nonempty_dataframe = True
            feat_engr_df = feature_engineer.engineer_features(
                data=cleaned_df,
                file_name=path,
                **args["feature_engineering"],
            )
            start = str(feat_engr_df.index.min().date()).replace("-", "")
            end = str(feat_engr_df.index.max().date()).replace("-", "")
            asset_name = get_asset_name(path)

            output_path = Path(output_dir / f"{asset_name}_{start}-{end}.csv")
            feat_engr_df.to_csv(output_path)
            missing_vals = feat_engr_df.isnull().sum().sum()
            if missing_vals > 0:
                logger.warning(f"{output_path}: {missing_vals} missing values remain.")
            logger.debug(f"Saved file at: {output_path}")
        else:
            output_path = None

    if not has_nonempty_dataframe:
        logger.warning("%s produced a cleaned dataframe with length 0.", str(path))

    return output_path


def get_clean_data_config(args: DictConfig) -> dict:
    """Changing args["clean_data"] and its values to primitive types. Also
    converts args["clean_data"]["scaler_args"]["quantile_range"] to tuple.

    Args:
        args (DictConfig): Hydra config as specified in yaml file.

    Returns:
        dict: args["clean_data"] in primitive type.
    """
    clean_data_config = format_omegaconf(args["clean_data"])
    clean_data_config["scaler_args"]["quantile_range"] = tuple(
        clean_data_config["scaler_args"]["quantile_range"]
    )
    return clean_data_config
