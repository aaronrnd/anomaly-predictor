import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.anomaly_predictor.modeling.data_loaders.data_loader import DataLoader
from src.anomaly_predictor.modeling.evaluation.evaluation_metrics import ModelEvaluator
from src.anomaly_predictor.modeling.models.isolation_forest import ABBIsolationForest
from src.anomaly_predictor.modeling.models.lstm_ae import LSTMAutoEncoder
from src.anomaly_predictor.modeling.utils import (
    floor_predictions_to_zero_for_nonoperating,
    infer_datetimeindex,
)
from src.anomaly_predictor.modeling.visualization import Visualizer
from src.anomaly_predictor.utils import copy_logs, format_omegaconf, get_asset_name

logger = logging.getLogger(__name__)


class EvaluationWrapper:
    """Wrapper to facilitate evaluation of models in pipeline."""

    def __init__(
        self,
        args: DictConfig,
        model: Union[ABBIsolationForest, LSTMAutoEncoder],
        eval_dir: Path,
        dataloader: DataLoader,
        feature_list: list,
        model_dir: Path,
    ):
        """Initializes parameters for EvaluationWrapper.

        Args:
            args (DictConfig): Hydra config as specified in yaml file.
            model (Union[ABBIsolationForest, LSTMAutoEncoder]): Model being evaluated
                against.
            eval_dir (Path): Directory intended to contain evaluation metrics for train,
                val, test partition, along with visualizations for top and botton n
                assets if specified. Also will contain precision recall curve and
                reconstruction error for each evaluated timepoint of data.
            dataloader (DataLoader): DataLoader for loading data for train, evaluation
                and inference purposes.
            feature_list (list): List of features to be scaled.
            model_dir (Path): Directory intended to contain model, other artifacts
                such as scalers and evaluation metrics, and other byproducts such
                as copy of config file, json file of model parameters.
        """
        self.args = args
        self.model = model
        self.eval_dir = eval_dir
        self.dataloader = dataloader
        self.feature_list = feature_list
        self.visualizer = Visualizer()
        self.pointwise_evaluator = ModelEvaluator()
        self.overlap_evaluator = ModelEvaluator("overlap")
        self.evaluation_dict = None
        self.processed_dir = Path(args["train_pipeline"]["processed_dir"])
        self.model_name = args["train_pipeline"]["model_name"]
        self.model_dir = model_dir

    def predict_all_assets(self) -> Dict[str, float]:
        """Runs through a loop that predict and evaluates all assets in processed_dir.

        Returns:
            Dict[str, float]: mean and standard deviation metrics across train, val,
                test partitions.
        """
        if self.model_name == "LSTMAE" and self.model.pred_threshold is None:
            logger.info(
                "No prediction threshold initialized. Finding best threshold..."
            )
            self.init_best_threshold(**self.args["pred_threshold_search"])

        metrics, self.evaluation_dict = {}, {}
        for partition in ["train", "val", "test"]:  # Opportunity to use dask
            evaluation_df, anomalous_list = self._predict_and_evaluate(partition)
            self.evaluation_dict[partition] = evaluation_df
            for metric in ("Pointwise_FPR", "Overlap_FPR"):
                metrics[f"{partition}_mean_{metric}"] = evaluation_df[metric].mean()
                metrics[f"{partition}_STD_{metric}"] = evaluation_df[metric].std()

            anomalous_asset_df = evaluation_df[
                evaluation_df["Asset_Name"].isin(anomalous_list)
            ]
            for metric in (
                "Pointwise_Recall",
                "Pointwise_F1_Score",
                "Overlap_Recall",
                "Overlap_F1_Score",
            ):
                metrics[f"{partition}_mean_{metric}"] = anomalous_asset_df[
                    metric
                ].mean()
                metrics[f"{partition}_STD_{metric}"] = anomalous_asset_df[metric].std()

            output_path = Path(self.eval_dir / partition / "evaluation.csv")
            evaluation_df.to_csv(str(output_path), index=False)
            logger.info(
                "All %s assets have been evaluated and saved in %s.",
                partition,
                self.eval_dir,
            )
            logger.info(
                "There are %s anomalous assets in %s partition.",
                len(anomalous_list),
                partition,
            )
            copy_logs(self.model_dir)
            if (
                self.args["evaluation"]["create_visualization"]
                and self.model_name != "IsolationForest"
            ):
                for criterion in self.args["visualization_args"]:
                    self._create_visualizations(
                        partition=partition,
                        criterion=criterion,
                        anomalous_list=anomalous_list,
                    )

        # change nan values to 0 in case there are no anomalous assets
        metrics = {
            metric: (0 if pd.isnull(val) else val) for metric, val in metrics.items()
        }

        logger.info("Evaluation is completed.")
        copy_logs(self.model_dir)
        return metrics

    def _predict_and_evaluate(self, partition: str) -> Tuple[pd.DataFrame, list]:
        """Loops through all assets in given partition to predict and evaluate them.
        Evaluation will be based on ground_truth and predicted values.

        Args:
            partition (str): Expects values in ("train", "val", "test") to know
                which partition subdirectory to refer to.

        Returns:
            Tuple[pd.DataFrame, list) where DataFrame is Evaluation dataframe
                of pointwise and overlapping metrics in the given partition, list
                contains filepaths of assets with at least one anomalous instance.
        """

        partition_dir = Path(self.processed_dir / partition)
        # create df for evaluation metrics
        evaluation_df = pd.DataFrame(columns=self.args["evaluation"]["columns"])
        anomalous_list, truths, concat_recon_errors = [], [], []
        for file in partition_dir.glob("**/*.csv"):
            try:
                feature_data, label_data, data = self.dataloader.load_eval_data(
                    file, self.feature_list, drop_last=False
                )
                if feature_data is None:  # feature data < lookback_period
                    continue
                if self.model_name == "IsolationForest":
                    preds = self.model.predict(feature_data)
                elif self.model_name == "LSTMAE":
                    recon_errors = self.model.get_reconstruction_error(feature_data)
                    preds = self.model.predict(feature_data, recon_errors=recon_errors)
                    label_data = label_data[-len(preds) :]
                    truths.extend(label_data)
                    concat_recon_errors.extend(recon_errors)
                    # save out recon errors:
                    reconstructed_csv_path = Path(
                        self.eval_dir / partition / f"{file.stem}.csv"
                    )
                    data["Reconstruction_Error"] = pd.Series(
                        recon_errors, index=data.index[-len(recon_errors) :]
                    )
                    data.to_csv(str(Path(reconstructed_csv_path)))
                    preds = floor_predictions_to_zero_for_nonoperating(data, preds)
                pointwise_metrics = self.pointwise_evaluator.evaluate(label_data, preds)
                # Overlap requires pd.Series with datetime index & name = 'Anomaly'
                if type(self.model) == LSTMAutoEncoder:
                    date_time_index = data.iloc[: (len(label_data))].index

                elif type(self.model) == ABBIsolationForest:
                    date_time_index = feature_data.index

                label_data = pd.Series(
                    label_data, index=date_time_index, name="Anomaly"
                )
                preds = pd.Series(preds, index=date_time_index, name="Anomaly")
                overlap_metrics = self.overlap_evaluator.evaluate(label_data, preds)
                evaluation_row = pd.DataFrame(
                    [[str(file), *pointwise_metrics, *overlap_metrics]],
                    columns=self.args["evaluation"]["columns"],
                )
                evaluation_df = evaluation_df.append(evaluation_row)
                if label_data.sum() > 0:
                    anomalous_list.append(str(file))

                # create tsne visualizations
                if self.args["evaluation"]["create_encoder_output_tsne"]:
                    tsne_dir = Path(self.eval_dir / partition / "tsne")
                    self.visualizer.set_save_dir(tsne_dir)
                    self.encode_and_save_tsne(feature_data, label_data, file)

            except Exception as e:
                logger.info("%s is problematic.", str(file))
                logger.warning(e)

        if self.model_name == "LSTMAE":
            self.visualizer.set_save_dir(self.eval_dir / partition)
            cur_threshold = self.model.pred_threshold
            self.visualizer.plot_precision_recall_curve(
                truths, concat_recon_errors, cur_threshold
            )
            max_error = max(concat_recon_errors)
            logger.info(f"Max {partition} reconstruction error: {max_error}")
            if partition == "train":
                self.model.max_reconstruction_error = max_error
            for quantile in (0.95, 0.75, 0.5):
                quan_val = np.quantile((concat_recon_errors), quantile)
                logger.info(f"{quantile:.0%} reconstruction error: {quan_val}")
            copy_logs(self.model_dir)
        return evaluation_df, anomalous_list

    def _create_visualizations(
        self, partition: str, criterion: str, anomalous_list: list
    ) -> None:
        """Creates Visualizations based on criterion specified in config file.
        Will extract top N and bottom N assets based on config file as well.

        Args:
            partition (str): Expects values in ("train", "val", "test") to know
                which partition subdirectory to refer to.
            criterion (str): Expects values ("Recall", "F1_Score", "FPR") to know
                which metric it is using to retrieve top / bottom assets.
            anomalous_list (list): list of assets which are anomalous.
        """
        plotting_dict = self._extract_assets_to_plot(
            **self.args["visualization_args"][criterion],
            partition=partition,
            criterion=criterion,
            anomalous_list=anomalous_list,
        )
        logger.info(
            "Extracted top %s assets and bottom %s assets based on %s",
            self.args["visualization_args"][criterion]["top_n_assets"],
            self.args["visualization_args"][criterion]["bottom_n_assets"],
            criterion,
        )
        copy_logs(self.model_dir)
        evaluation_config = format_omegaconf(self.args["evaluation"])
        visualization_dir = Path(self.eval_dir / partition)
        Path(visualization_dir / "Top_Assets" / criterion).mkdir(
            parents=True, exist_ok=True
        )
        Path(visualization_dir / "Bottom_Assets" / criterion).mkdir(
            parents=True, exist_ok=True
        )
        for asset in plotting_dict[partition]:

            # Sort assets in visualization directory
            self._sort_top_bottom_asset(
                plotting_dict,
                partition,
                asset,
                self.args["visualization_args"][criterion]["top_n_assets"],
                visualization_dir,
                criterion,
            )

            # Create predictions to be visualized
            outputs = self._visualization_predict(asset)
            original_data, asset_name, ground_truth, anom_scores = outputs
            anom_scores.index = infer_datetimeindex(original_data, anom_scores)
            ground_truth.index = infer_datetimeindex(original_data, ground_truth)
            agg_predictions = anom_scores.groupby(anom_scores.index.date).mean()
            agg_predictions = pd.Series(
                self.model.predict(None, recon_errors=agg_predictions),
                index=agg_predictions.index,
            )
            # Plot predictions
            self.visualizer.plot_anomaly(
                original_data,
                ground_truth,
                agg_predictions,
                asset_name=asset_name,
                features=evaluation_config["plotting_features"],
                use_data_datetimeindex=False,
            )
        logger.info("Visualizations are created successfully in %s", self.eval_dir)

    def _extract_assets_to_plot(
        self,
        criterion: str,
        top_n_assets: int,
        bottom_n_assets: int,
        partition: str,
        anomalous_list: list,
    ) -> Dict[str, list]:
        """Extract top N and bottom N assets by sorted values according to criterion.
        Returns a dictionary of {"train":List[str], "val":List[str], "test":List[str]}.

        Args:
            criterion (str): Criterion to sort df on.
            top_n_assets (int): Number of top assets to extract out.
            bottom_n_assets (int): Number of bottom assets to extract out.
            partition (str): Expects values in ("train", "val", "test").
            anomalous_list (list): list of assets which are anomalous.

        Returns:
            dict: Dictionary of {partition:list of assets (top+bottom)}.
        """
        plotting_dict = {}
        evaluating_dataframe = self.evaluation_dict[partition]

        if criterion in (
            "Pointwise_F1_Score",
            "Pointwise_Recall",
            "Overlap_F1_Score",
            "Overlap_Recall",
        ):
            evaluating_dataframe = evaluating_dataframe[
                evaluating_dataframe["Asset_Name"].isin(anomalous_list)
            ]
        ascending = "fpr" in criterion.lower()
        evaluating_dataframe.sort_values(
            by=criterion, inplace=True, ascending=ascending
        )
        top_n_assets_list = (
            evaluating_dataframe["Asset_Name"].head(top_n_assets).tolist()
        )
        bottom_n_assets_list = (
            evaluating_dataframe["Asset_Name"].tail(bottom_n_assets).tolist()
        )
        plotting_dict[partition] = top_n_assets_list + bottom_n_assets_list

        return plotting_dict

    def _visualization_predict(
        self, asset_filepath: Path
    ) -> Tuple[pd.DataFrame, str, pd.DataFrame, pd.Series]:
        """Generates information needed for plotting model predictions on asset.
        Anomaly scores is returned instead of actual binarized prediction because
        the scores will be daily aggregated first then binarized during visualization.

        Args:
            asset_filepath (Path): Path of asset being evaluated & plotted.

        Returns:
            Tuple[pd.DataFrame, str, pd.DataFrame, pd.Series]: a tuple representing
                the loaded dataset, asset name of the loaded dataset, ground truth
                of the asset and the predicted labels of the asset.
        """
        feature_data, ground_truth, original_data = self.dataloader.load_eval_data(
            Path(asset_filepath), self.feature_list, drop_last=False
        )
        ground_truth = pd.DataFrame(ground_truth, columns=["Anomaly"])
        anomaly_scores = pd.Series(self.model.predict_anomaly_score(feature_data))
        asset_name = get_asset_name(asset_filepath)

        return original_data, asset_name, ground_truth, anomaly_scores

    def _sort_top_bottom_asset(
        self,
        plotting_dict: dict,
        partition: str,
        asset_index: int,
        top_n_assets: int,
        viz_dir: Path,
        criterion: str,
    ) -> None:
        """Sorts assets in plotting_dict and sets their save directory.

        Args:
            plotting_dict (dict): Dictionary of {partition:list of assets (top+bottom)}.
            partition (str): Partition key in plotting_dict.
            asset_index (int): Index of asset in list in plotting_dict.
            top_n_assets (int): Number of top assets to extract out.
            viz_dir (Path): Path to visualization directory.
            criterion (str): Expects values ("Recall", "F1_Score", "FPR").
        """
        if plotting_dict[partition].index(asset_index) <= top_n_assets - 1:
            self.visualizer.set_save_dir(Path(viz_dir / "Top_Assets" / criterion))
        else:
            self.visualizer.set_save_dir(Path(viz_dir / "Bottom_Assets" / criterion))

    def encode_and_save_tsne(
        self, feature_data: torch.Tensor, label_data: pd.DataFrame, file: Path
    ) -> None:
        """Run windowed feature data into trained encoded portion of LSTM AE, and
        run TSNE on encoded vectors. Saves TSNE Visualizations out to TSNE directory.

        Args:
            feature_data (torch.Tensor): Windowed Data.
            label_data (pd.DataFrame): Labels to indicate if data is anomalous.
            file (Path): File name.
        """
        encoded_data = pd.DataFrame()
        for _, x in enumerate(feature_data):
            x = x.to(self.model.device)
            output = self.model.model.encode(x)
            encoded_vector = pd.DataFrame(output).astype("float")
            encoded_data = encoded_data.append(encoded_vector)
        self.visualizer.plot_tsne(encoded_data, label_data, file)

    def init_best_threshold(
        self,
        max_fpr: float = 0.2,
        terminating_recall: float = 0.3,
        default_pred_threshold: float = 0.5,
    ) -> None:
        """Tries to find threshold such that training set has max pointwise recall while
         still keeping fpr under max_fpr.

        Args:
            max_fpr (float, optional): Max tolerable false positive rate. Defaults to 0.2.
            terminating_recall (float, optional): Terminates search if recall falls
                below this value. Defaults to 0.3.
            default_pred_threshold (float, optional): Default pred threshold if model is
                unable to meet max_fpr and terminating_recall. Defaults to 0.5.
        """
        truths, concat_recon_errors = [], []
        train_dir = Path(self.processed_dir / "train")
        for file in train_dir.glob("*.csv"):
            try:
                feature_data, label_data, _ = self.dataloader.load_eval_data(
                    file, self.feature_list, drop_last=False
                )
                if feature_data is None:  # feature data < lookback_period
                    continue
                recon_errors = self.model.get_reconstruction_error(feature_data)
                label_data = label_data[-len(recon_errors) :]
                truths.extend(label_data)
                concat_recon_errors.extend(recon_errors)
            except Exception as e:
                logger.info("%s is problematic.", str(file))
                logger.warning(e)
        rounded_errors = np.round(concat_recon_errors, 1)
        thresholds = np.unique(rounded_errors).astype("float64")
        metric_cols = ["threshold", "recall", "f1", "fpr"]
        threshold_metrics = pd.DataFrame(columns=metric_cols)
        logger.info(f"Evaluating up to {len(thresholds)} thresholds...")
        for threshold in thresholds[1:-1]:
            preds = np.where(concat_recon_errors < threshold, 0, 1)
            metrics = self.pointwise_evaluator.evaluate(truths, preds)
            recall, _, _ = metrics
            if recall < terminating_recall:
                break
            threshold_row = pd.DataFrame([[threshold, *metrics]], columns=metric_cols)
            threshold_metrics = pd.concat(
                [threshold_metrics, threshold_row], ignore_index=True
            )
        output_path = Path(self.model_dir / "threshold_metrics.csv")
        if len(threshold_metrics) > 0:
            threshold_metrics.round(3).to_csv(str(output_path), index=False)
        else:
            logger.info(
                f"There are either no anomalous assets in train partition or none of the thresholds can satisfy {terminating_recall}."
            )

        valid_models = threshold_metrics[threshold_metrics["fpr"] <= max_fpr]
        if len(valid_models) == 0:
            logger.info(f"There are no models that has max_fpr lower than {max_fpr}.")
            logger.info(f"Pred threshold initialized as {default_pred_threshold}.")
            self.model.pred_threshold = default_pred_threshold
        else:
            best_threshold = threshold_metrics.iloc[valid_models["recall"].idxmax, 0]
            logger.info(
                f"Pred threshold initialized as {best_threshold}, given max fpr of {max_fpr}"
            )
            self.model.pred_threshold = best_threshold
