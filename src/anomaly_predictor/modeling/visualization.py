from math import ceil
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve

from src.anomaly_predictor.modeling.utils import infer_datetimeindex


class Visualizer:
    """Class to facilitate plotting of anomaly predictions"""

    def __init__(self, save_dir=None):
        self.truth_color = "green"
        self.pred_color = "#e07070"
        self.plot_face_color = (0.9, 0.9, 0.9)
        self.save_dir = save_dir

    def plot_anomaly(
        self,
        data: pd.DataFrame,
        ground_truth: Union[None, pd.Series] = None,
        predictions: Union[None, pd.Series] = None,
        anomaly_scores: Union[None, pd.Series] = None,
        asset_name: str = "Asset",
        features: Union[None, list] = None,
        n_cols: int = 1,
        use_data_datetimeindex: bool = True,
    ):
        """Plots data in a univariate fashion while highlighting known and
        predicted anomalous periods

        Args:
            data (pd.DataFrame): feature values
            ground_truth (Union[None, pd.Series]): pandas Series of 0 and 1
                where 1 indicates anomaly or an imminent anomaly. Defaults to None.
            predictions (Union[None, pd.Series]): pandas Series of 0 and 1
                where 1 indicates predicted anomaly or a forecasted anomaly. Defaults to None.
            anomaly_scores (Union[None, pd.Series]): pandas Series of floats between
                0 and 1 where the higher the value, the higher the probability of
                anomaly. Defaults to None.
            asset_name (str, optional): Name of asset being plotted. Defaults to "Asset".
            features (Union[None, list], optional): list of features in data to
                be plotted. If none is specified, will plot all features.
                Defaults to None.
            n_cols (int, optional): Number of columns in subplot. Defaults to 1.
            use_data_datetimeindex (bool, optional): whether to use data's
                DatetimeIndex when plotting predictions, ground_truth and
                anomaly_scores. Defaults to True.
        """

        features = features or data.columns
        n_rows = ceil((len(features)) / n_cols)
        plt.figure(figsize=(18, 6 * n_rows))

        for i, feature in enumerate(features, start=1):
            lines = []
            ax = plt.subplot(n_rows, n_cols, i)
            ln = ax.plot(data.index, data[feature], label="Value")
            ax.margins(x=0)
            ax.set_title(f"{asset_name}: {feature}")
            ax.set_facecolor(self.plot_face_color)
            ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
            lines.extend(ln)

            if ground_truth is not None:
                if use_data_datetimeindex:
                    ground_truth.index = infer_datetimeindex(data, ground_truth)
                self._plot_anoms(ax, ground_truth, self.truth_color)

            if predictions is not None:
                if use_data_datetimeindex:
                    predictions.index = infer_datetimeindex(data, predictions)
                self._plot_anoms(ax, predictions, self.pred_color)

            if anomaly_scores is not None:
                if use_data_datetimeindex:
                    idx = infer_datetimeindex(data, anomaly_scores)
                else:
                    idx = anomaly_scores.index
                lines = self._plot_anomaly_scores(ax, anomaly_scores, idx, lines)

            if len(lines) > 1:
                ax.legend(lines, [l.get_label() for l in lines], loc="upper right")

        plt.tight_layout()

        start = str(data.index.min().date()).replace("-", "")
        end = str(data.index.max().date()).replace("-", "")
        plt.savefig(Path(self.save_dir / f"{asset_name}_{start}-{end}.png"))
        plt.close()

    @staticmethod
    def _plot_anoms(ax: plt.axes, labels: pd.Series, color: str) -> None:
        """Plots vertical span on ax to indicate anomalous period

        Args:
            ax (plt.axes): axes on which the span is plotted
            labels (pd.Series): pandas Series of 0 and 1 where 1 indicates anomaly
            color (str): color of plotted span
        """
        time, val = labels.index, labels.values
        splits = np.where(val[1:] != val[:-1])[0] + 1
        splits = np.concatenate(([0], splits, [len(val) - 1]))
        for i in range(len(splits) - 1):
            if val[splits[i]]:
                ax.axvspan(time[splits[i]], time[splits[i + 1]], color=color, alpha=0.5)

    def _plot_anomaly_scores(
        self, ax: plt.axes, anomaly_scores: pd.Series, index: pd.Series, lines: list
    ) -> list:
        """Plots the anomaly scores on the same AxesSubPlot but on a separate y-scale

        Args:
            ax (plt.axes): original axes on which features are plotted
            anomaly_scores (pd.Series): Anomaly scores
            index (pd.Series): datetime index of the given data
            lines (list): a list of mpl.Line2D objects for label retrieval when
                showing legend of plot

        Returns:
            list: a list of mpl.Line2D objects for label retrieval when
                showing legend of plot
        """
        ax2 = ax.twinx()
        ln = ax2.plot(index, anomaly_scores, color="red", label="Anomaly Score")
        ax2.set_ylabel("Anomaly Score")
        ax2.set_ylim(0, 3)
        ax2.margins(x=0)
        lines.extend(ln)
        return lines

    def set_truth_span_color(self, color: str) -> None:
        """Changing span color for plotting anomalous periods in ground truth

        Args:
            color (str): color of plotted span
        """
        self.truth_color = color

    def set_pred_span_color(self, color: str) -> None:
        """Changing span color for plotting anomalous periods in predictions

        Args:
            color (str): color of plotted span
        """
        self.pred_color = color

    def set_save_dir(self, save_dir: Path) -> None:
        """Specify directory in which plots from Visualizer are saved

        Args:
            save_dir (Path): directory for saving plots
        """
        self.save_dir = save_dir

    def plot_epoch_losses(self, losses: dict, filename: str = "model_loss"):
        """Plots epoch loss encountered during neural network training

        Args:
            losses (dict): A dictionary with values being list of losses across epochs.
            filename (str, optional): Filename for saved plot. Defaults to "model_loss".
        """
        plt.figure(figsize=(18, 6))
        index = list(range(1, len(losses["train"]) + 1))
        for key in losses:
            plt.plot(index, losses[key], label=key)
        plt.legend(loc="upper right")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(Path(self.save_dir / f"{filename}.png"))
        plt.close()

    def plot_precision_recall_curve(
        self,
        ground_truth: np.ndarray,
        reconstruction_errors: np.ndarray,
        current_threshold: float,
        filename: str = "precision_recall_curve",
    ):
        """Given ground truth labels and reconstruction errors, plots precision
        recall curve to aid in finding the best threshold for predictions

        Args:
            ground_truth (np.ndarray): array of values {-1, 1} or {0, 1} where
                1 indicates positive.
            reconstruction_errors (np.ndarray): array of floats which can either
                be probability estimates or non-thresholded values.
            current_threshold (float): The current binarizing threshold specified in config
            filename (str, optional): Filename for saved plot. Defaults to
                "precision_recall_curve".
        """
        plt.figure(figsize=(18, 6))
        precision, recall, threshold = precision_recall_curve(
            ground_truth, reconstruction_errors, pos_label=1
        )
        plt.plot(threshold, precision[1:], label="Precision")
        plt.plot(threshold, recall[1:], label="Recall")
        plt.axvline(
            x=current_threshold, color=self.pred_color, linestyle="--", linewidth=1
        )
        plt.legend(loc="upper center")
        plt.title("Precision Recall Curve")
        plt.xlabel("Threshold")
        plt.ylabel("Precision/Recall")
        plt.savefig(Path(self.save_dir / f"{filename}.png"))
        plt.close()

    def plot_tsne(
        self,
        encoded_data: pd.DataFrame,
        label_data: pd.DataFrame = None,
        file: Path = "TSNE",
    ):
        """Given encoded data and labels, plots TSNE's visualization to
        identify any potential clusters

        Args:
            encoded_data (pd.DataFrame): Single vector encoded data.
            label_data (pd.DataFrame, optional): Anomaly label for encoded data. Defaults to None.
            file (Path, optional): Filename for saved plot. Defaults to "TSNE".
        """
        tsne = TSNE(
            n_components=2, verbose=1, perplexity=15, n_iter=1000, random_state=42
        )
        tsne_results = tsne.fit_transform(encoded_data)
        print("t-SNE completed!")
        encoded_data["tsne-2d-one"] = tsne_results[:, 0]
        encoded_data["tsne-2d-two"] = tsne_results[:, 1]
        encoded_data["label"] = label_data
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="label",
            data=encoded_data,
            legend="full",
            alpha=0.4,
        )
        plt.title(f"TSNE visualization for {file.stem}")
        plt.savefig(Path(self.save_dir / f"{file.stem}.png"))
        plt.close()
