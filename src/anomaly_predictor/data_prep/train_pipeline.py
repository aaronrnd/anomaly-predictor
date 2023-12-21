import logging
from glob import glob
from pathlib import Path
from typing import Tuple

import dask
import hydra
import pandas as pd
from omegaconf import DictConfig

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
from src.anomaly_predictor.utils import (
    copy_logs,
    create_timestamp,
    get_conf_dir,
    save_out_config,
    save_out_copy,
    timer,
)

logger = logging.getLogger(__name__)
dask.config.set(scheduler="multiprocessing")


@hydra.main(config_path="../../../conf", config_name="train_pipeline.yml")
def run_pipeline(args: DictConfig) -> Tuple[dict, Path]:
    """Main function to ingest and clean data. This function does the following:
    1. ingest asset xlsx and annotations into csv files in interim folder.
    2. Splits assets into train, val and test sets.
    3. Removes negatives, imputes missing data and outliers.
    4. Include engineered features.

    Args:
        args (DictConfig): Hydra config as specified in yaml file.
    Returns:
        Tuple(dict,Path): A tuple containing 1. a dict which has keys 'train', 'val',
            'test' and value being list of filepaths pointing to where processed data
            is stored. 2. The processed folder directory.
    """
    data_prep_args = args["data_prep"]
    try:
        timestamp = create_timestamp()
        input_dir, interim_dir, processed_dir = setup_logging_and_dir(
            data_prep_args, timestamp, mode="training"
        )
        copy_logs(processed_dir)

        saved_config = save_out_config(processed_dir)
        if not saved_config:
            # Save out a copy of train_pipeline.yml to current timestamped folder
            conf_path = get_conf_dir() or Path(data_prep_args["pipeline"]["conf_dir"])
            config_yml_path = Path(conf_path / "train_pipeline.yml")
            save_out_copy(config_yml_path, Path(processed_dir / "train_pipeline.yml"))

        # Use predefined split or run annotation ingestion if predefined split is empty / invalid
        follow_split_json = data_prep_args["split_data"]["follow_split"]
        if follow_split_json:
            logging.info(f"Following previous data split {follow_split_json}")
            interim_split_dict, applications = get_split_dict_from_json(
                Path(follow_split_json), processed_dir
            )
            prev_interim_exists = Path(interim_split_dict["train"][0]).parent.exists()
            if (
                not prev_interim_exists
                and data_prep_args["split_data"]["by"] == "asset"
            ):
                interim_dir = Path(interim_dir / timestamp)
                with timer("Annotation ingestion"):
                    run_annotation_ingestion(
                        data_prep_args, input_dir, interim_dir, mode="training"
                    )
                interim_split_dict = update_split_dict(interim_split_dict, interim_dir)
            copy_logs(processed_dir)
        if not follow_split_json or (
            data_prep_args["split_data"]["by"] == "time" and not prev_interim_exists
        ):
            interim_dir = Path(interim_dir / timestamp)
            with timer("Annotation ingestion"):
                run_annotation_ingestion(
                    data_prep_args, input_dir, interim_dir, mode="training"
                )
            interim_split_dict, applications = run_data_splitter(
                data_prep_args, interim_dir, processed_dir
            )
            copy_logs(processed_dir)
        for key, fp_list in interim_split_dict.items():
            logger.info(f"{key}: {len(fp_list)} files")

        time_col = data_prep_args["ingest_data"]["time_col"]
        clean_data_config = get_clean_data_config(data_prep_args)
        data_cleaner = DataCleaner()
        feature_engineer = FeatureEngineer()
        feature_engineer.fit_encoder(applications)
        feature_engineer.save_encoder(
            Path(processed_dir), data_prep_args["inference_encoder"]["enc_name"]
        )

        delayed_results = []
        for key, csv_paths in interim_split_dict.items():
            for path in csv_paths:
                data = dask.delayed(pd.read_csv)(
                    path,
                    parse_dates=[time_col],
                    infer_datetime_format=True,
                    index_col=time_col,
                )
                cleaned_df_list = dask.delayed(data_cleaner.clean_data)(
                    data, **clean_data_config
                )
                output_path = dask.delayed(feature_engineer_df_list)(
                    cleaned_df_list,
                    data_prep_args,
                    Path(path),
                    feature_engineer,
                    Path(processed_dir / key),
                )
                delayed_results.append(output_path)

        with timer("Data Cleaning and Feature Engineering"):
            delayed_results = dask.compute(*delayed_results)

        copy_logs(processed_dir)

        return {
            key: list(glob(str(Path(processed_dir / key / "*.csv"))))
            for key in ("train", "val", "test")
        }, processed_dir

    except Exception as e:
        logger.error(e, exc_info=True)
        copy_logs(processed_dir)
        raise


if __name__ == "__main__":
    with timer("Data preparation"):
        run_pipeline()
