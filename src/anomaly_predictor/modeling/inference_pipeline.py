import logging
import warnings
from json import load
from pathlib import Path
from typing import Tuple, Union

import hydra
import pandas as pd
from omegaconf import DictConfig, omegaconf
from sklearn.ensemble import IsolationForest

from src.anomaly_predictor.modeling.data_loaders.data_loader import DataLoader
from src.anomaly_predictor.modeling.models.isolation_forest import ABBIsolationForest
from src.anomaly_predictor.modeling.models.lstm_ae import LSTMAutoEncoder
from src.anomaly_predictor.modeling.utils import (
    floor_predictions_to_zero_for_nonoperating,
    post_process,
)
from src.anomaly_predictor.modeling.visualization import Visualizer
from src.anomaly_predictor.utils import (
    create_timestamp,
    format_omegaconf,
    get_asset_name,
    timer,
)

# suppress warnings for now to monitor info logs.
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../conf", config_name="inference_pipeline.yml")
def run_inference(args: DictConfig) -> Path:
    """Performs inference using trained model and training parameters from hydra
    config. Inference data to be taken from processed_data directory.
    Postprocessed csv(s) and visualisation(s) of prediction csv(s) will
    be saved to prediction directory.

    Args:
        args (DictConfig): Arguments from inference_pipeline.yml

    Returns:
        Path: Directory where predictions are to be saved in
    """
    artifact_args = args["artifacts"]
    model_dir = Path(artifact_args["model_dir"])
    args = args["modeling"]

    logger.info("Starting Inferencing")
    error_list = []
    model_name = artifact_args["model_name"]
    return_mode = "pytorch" if model_name == "LSTMAE" else "sklearn"
    if model_name == "IsolationForest":
        model = ABBIsolationForest()
        model.load_model(Path(model_dir / f"{model_name}.joblib"))
        binarizing_threshold = 0.5

    elif model_name == "LSTMAE":
        model_params = load(open(Path(model_dir / "model_params.json")))
        binarizing_threshold = model_params["pred_threshold"]
        model_architecture = model_params["model_architecture"]
        del model_params["model_architecture"]
        model = LSTMAutoEncoder(model_params=model_architecture, **model_params)
        model.load_model(Path(model_dir / f"{model_name}.pt"))

    logger.info("Model Loaded.")

    # visualizer = Visualizer()
    # visualization_config = format_omegaconf(args["visualizations"])

    data_loader_config = format_omegaconf(args["data_loader"])
    data_loader = DataLoader(return_mode=return_mode, **(args["data_loader"]["init"]))
    data_loader.load_scaler(Path(model_dir / f'{artifact_args["scaler_name"]}.joblib'))

    timestamp = create_timestamp()
    prediction_dir = Path(args["inference_pipeline"]["prediction_dir"])
    prediction_dir = Path(prediction_dir / timestamp)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    expected_fp = Path(args["inference_pipeline"]["inference_dir"])
    files = expected_fp.glob("**/*.csv")
    for file in files:
        asset_path = Path(prediction_dir / Path(file).stem)
        asset_path.mkdir(parents=True, exist_ok=True)
        try:
            inference_data, original_data = data_loader.load_inference_data(
                file, data_loader_config["feature_to_standardize"], drop_last=False
            )
            postprocessed_data,postprocessed_daily = predict_and_postprocess(
                args,
                model,
                inference_data,
                original_data,
                file,
                asset_path,
                binarizing_threshold
            )
            colname = [
                col
                for col in postprocessed_daily.columns
                if "DAILY_AGGREGATED_PREDICTION" in col
            ]
            daily_agg_predictions = postprocessed_daily[colname]  # Fix this line
            daily_agg_predictions = daily_agg_predictions.dropna()

            if args["inference_pipeline"]["create_visualizations"]:
                visualizer.set_save_dir(asset_path)
                visualizer.plot_anomaly(
                    postprocessed_data,
                    ground_truth=None,
                    predictions=daily_agg_predictions,
                    asset_name=get_asset_name(str(Path(file).stem)),
                    features=visualization_config["plotting_features"],
                    use_data_datetimeindex=False,
                )
                logger.info("Visualizations for %s is completed.", str(file))
        except Exception as e:
            error_list.append((file, e))

    if len(error_list) > 0:
        logger.info("\nERRORS Occured!")
    for item, errormessage in error_list:
        asset_name = Path(item).stem
        logger.warning(
            "Inference for %s is not completed. Error Message : %s",
            asset_name,
            errormessage
        )

    logger.info("Inferencing has completed.")
    return prediction_dir

def predict_and_postprocess(
    args:omegaconf,
    model: Union[LSTMAutoEncoder,IsolationForest],
    inference_data: DataLoader,
    original_data: DataLoader,
    file: Path,
    asset_path: Path,
    binarizing_threshold:float)->Tuple[pd.DataFrame,pd.DataFrame]:
    """Uses trained model to perform inference on inference data
    and returns postprocessed predictions. Resultant prediction csv(s) will be saved to
    "asset_path/file.csv". Function will return postprocessed_data and postprocessed_daily
    dataframes for plotting of visualizations too.

    Args:
        args (omegaconf): hydra configs.
        model (Union[LSTMAutoEncoder,IsolationForest]): Trained Model.
        inference_data (DataLoader): Inference data tensors loaded by data_loader.
        original_data (DataLoader): Original data tensors loaded by data_loader.
        file (Path): Original file name.
        asset_path (Path): Parent directory of eventual folder.
        binarizing_threshold (float): Binarizing threshold for binarizing of Reconstruction Errors.

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame]: Tuple of postprocessed_data and postprocessed_daily df.
    """
    predictions = model.predict(inference_data)
    prediction_len = len(predictions)
    predictions = pd.Series(predictions, index=original_data[-prediction_len:].index)
    prediction_scores = model.predict_anomaly_score(inference_data)
    prediction_scores = pd.Series(prediction_scores, index=original_data[-prediction_len:].index)
    predictions = floor_predictions_to_zero_for_nonoperating(original_data,predictions)
    prediction_scores = floor_predictions_to_zero_for_nonoperating(original_data,prediction_scores)
    logger.info("prediction for %s is completed.", str(file))
    postprocessed_data, postprocessed_daily = post_process(
        original_data,
        prediction_scores,
        predictions,
        args["data_loader"]["init"]["lookahead_period"],
        binarizing_threshold=binarizing_threshold
    )

    logger.info("post processing for %s is completed. ", str(file))
    postprocessed_data.to_csv(str(asset_path / Path(file).stem) + ".csv")
    postprocessed_daily.to_csv(str(asset_path / Path(file).stem) + "_daily.csv")
    return postprocessed_data,postprocessed_daily

if __name__ == "__main__":
    with timer("Running Inference Pipeline"):
        run_inference()
