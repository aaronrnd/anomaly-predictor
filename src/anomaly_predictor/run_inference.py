import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.anomaly_predictor.data_prep.inference_pipeline import run_pipeline
from src.anomaly_predictor.modeling.inference_pipeline import run_inference
from src.anomaly_predictor.utils import copy_logs, timer

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="inference_pipeline.yml")
def run_endtoend_inference(main_cfg: DictConfig):
    """Wrapper function to run Data Prep pipeline and inference pipeline.
    Will include argparse for edits to override default hydra config parameters

    Args:
        main_cfg (DictConfig): Hydra config file
    """
    model_dir = Path(main_cfg["artifacts"]["model_dir"])

    trained_args = OmegaConf.load(Path(model_dir / "train_pipeline.yml"))

    # read in model and encoder name from train_pipeline.yml
    main_cfg["artifacts"]["encoder_name"] = trained_args["data_prep"][
        "inference_encoder"
    ]["enc_name"]
    main_cfg["artifacts"]["model_name"] = trained_args["modeling"]["train_pipeline"][
        "model_name"
    ]

    # Reading configs for data_prep and data_loader from train_pipeline.yml
    data_prep_cfg = trained_args["data_prep"]
    data_loader_cfg = trained_args["modeling"]["data_loader"]

    # Override data_prep configs with train_pipeline.yml
    for key in ("clean_data", "feature_engineering"):
        main_cfg["data_prep"][key] = data_prep_cfg[key]
    logger.info("Running Data Prep Pipeline...")
    _, processed_dir = run_pipeline(main_cfg)
    main_cfg["modeling"]["inference_pipeline"]["inference_dir"] = str(processed_dir)

    # Override in data loader configs with train_pipeline.yml
    for key in ("batch_size", "pin_memory", "num_workers"):
        data_loader_cfg["init"][key] = main_cfg["modeling"]["data_loader"]["init"][key]
    if trained_args["modeling"]["use_individual_scaler"]:
        data_loader_cfg["init"]["scaler_dir"] = str(Path(model_dir / "scalers"))
    main_cfg["modeling"]["data_loader"] = data_loader_cfg

    logger.info("Running Inference...")
    prediction_dir = run_inference(main_cfg)

    logger.info("End to End Pipeline Completed.")
    copy_logs(prediction_dir)


if __name__ == "__main__":
    with timer("End to End Inference"):
        run_endtoend_inference()
