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
    run_annotation_ingestion,
    setup_logging_and_dir,
)
from src.anomaly_predictor.utils import copy_logs, create_timestamp, timer

logger = logging.getLogger(__name__)
dask.config.set(scheduler="multiprocessing")

@hydra.main(config_path="../../../conf", config_name="inference_pipeline.yml")
def run_pipeline(args: DictConfig) -> Tuple[list,Path]:
    """Main function to ingest and clean data. This function does the following:
    1. ingest asset xlsx and annotations into csv files in interim folder.
    2. Removes negatives, imputes missing data and outliers.
    3. Include engineered features (If missing OHE put it as 0).

    Args:
        args (DictConfig): Hydra config as specified in yaml file.
    Returns:
        Tuple(list,path): A tuple containing the list of filepaths pointing to where
        processed data is stored and the processed folder directory.
    """
    data_prep_args = args["data_prep"]
    try:
        timestamp = create_timestamp()
        input_dir, interim_dir, processed_dir = setup_logging_and_dir(
            data_prep_args, timestamp, mode="inference"
        )

        with timer("Annotation ingestion"):
            run_annotation_ingestion(
                data_prep_args, input_dir, interim_dir, mode="inference"
            )

        time_col = data_prep_args["ingest_data"]["time_col"]
        clean_data_config = get_clean_data_config(data_prep_args)

        data_cleaner = DataCleaner()
        feature_engineer = FeatureEngineer()
        enc_path = Path(args["artifacts"]["model_dir"]) / (
            args["artifacts"]["encoder_name"] + ".joblib"
        )
        feature_engineer.load_encoder(enc_path)

        csv_paths = glob(str(Path(interim_dir / "*.csv")))
        delayed_results = []
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
                processed_dir,
            )
            delayed_results.append(output_path)

        with timer("Data Cleaning and Feature Engineering"):
            delayed_results = dask.compute(*delayed_results)

        copy_logs(processed_dir)

        return glob(str(Path(processed_dir / "*.csv"))), processed_dir

    except Exception as e:
        logger.warning("Error in data_prep", e)

if __name__ == "__main__":
    with timer("Data preparation for inference"):
        run_pipeline()
