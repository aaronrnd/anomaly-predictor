import logging
from typing import Union

import hydra
from omegaconf import DictConfig

from src.anomaly_predictor.data_prep.train_pipeline import run_pipeline
from src.anomaly_predictor.modeling.train_pipeline import run_training
from src.anomaly_predictor.utils import copy_logs, timer

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="train_pipeline.yml")
def run_endtoend_training(main_cfg: DictConfig) -> Union[float, float, float]:
    """Wrapper function to run Data Prep pipeline and training pipeline.
    Will include argparse for edits to override default hydra config parameters.

    Args:
        main_cfg (DictConfig): Hydra config file

    Returns:
        Union[float, float, float]: Returns model metrics FPR, Recall and F1_score
    """
    logger.info("Running Data Prep Pipeline...")
    _, processed_dir = run_pipeline(main_cfg)
    main_cfg["modeling"]["train_pipeline"]["processed_dir"] = str(processed_dir)

    logger.info("Running Training...")
    test_FPR, test_F1, test_Recall, model_dir = run_training(main_cfg)

    logger.info("End to End Pipeline Completed.")
    copy_logs(model_dir)

    return test_FPR, test_F1, test_Recall


if __name__ == "__main__":
    with timer("End to End Training"):
        run_endtoend_training()
