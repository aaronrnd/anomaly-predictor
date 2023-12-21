# pylint: skip-file
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.modeling.visualization import Visualizer


@pytest.fixture(scope="session")
def save_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("plots")


@pytest.fixture
def date_range():
    return pd.date_range("2021-07-01", periods=20, freq="H")


@pytest.fixture
def ground_truth(date_range):
    return pd.DataFrame(
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        index=date_range,
        columns=["Label"],
    )


@pytest.fixture
def predictions(date_range):
    return pd.DataFrame(
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        index=date_range,
        columns=["Label"],
    )


@pytest.fixture
def smaller_predictions(date_range):

    return pd.DataFrame(
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        index=date_range[-15:],
        columns=["Label"],
    )


@pytest.fixture
def anomaly_scores(date_range):
    return pd.DataFrame(
        [
            0,
            0,
            0,
            0.6,
            0.7,
            0.65,
            0.22,
            0.03,
            0.01,
            0.12,
            0.12,
            0.13,
            0,
            0.78,
            0.86,
            0.86,
            0.65,
            0.6,
            0.3,
            0.2,
        ],
        index=date_range,
        columns=["Label"],
    )


@pytest.fixture
def smaller_anomaly_scores():
    return np.array(
        [
            0.65,
            0.22,
            0.03,
            0.01,
            0.12,
            0.12,
            0.13,
            0,
            0.78,
            0.86,
            0.86,
            0.65,
            0.6,
            0.3,
            0.2,
        ]
    )


@pytest.fixture
def dummy_data(date_range):
    return pd.DataFrame(
        np.random.random(size=(20, 2)),
        columns=["Vibration", "Acceleration"],
        index=date_range,
    )


@pytest.fixture
def losses():
    np.random.seed(18)
    return {"train": np.random.random((100,)), "val": np.random.random((100,))}


@pytest.fixture
def recon_errors():
    np.random.seed(18)
    return np.random.random((20,))


def test_plot_anomaly(
    dummy_data,
    ground_truth,
    predictions,
    anomaly_scores,
    save_dir,
    smaller_anomaly_scores,
    smaller_predictions,
):
    viz = Visualizer()
    viz.set_save_dir(save_dir)
    viz.plot_anomaly(dummy_data, ground_truth, predictions)
    viz.plot_anomaly(dummy_data, ground_truth, predictions, anomaly_scores)
    viz.plot_anomaly(
        dummy_data, ground_truth, smaller_predictions, smaller_anomaly_scores
    )
    
    # check that a figure gets saved in save_dir
    assert len(os.listdir(save_dir)) > 0

    # check that there are no open figure windows
    assert len(plt.get_fignums()) == 0


def test_plot_epoch_losses(losses, save_dir):
    viz = Visualizer()
    viz.set_save_dir(save_dir)

    n_files_initial = len(os.listdir(save_dir))
    viz.plot_epoch_losses(losses, filename="loss")
    assert len(os.listdir(save_dir)) - n_files_initial == 1
    assert "loss.png" in os.listdir(save_dir)

    # check that there are no open figure windows
    assert len(plt.get_fignums()) == 0


def test_plot_precision_recall_curve(ground_truth, recon_errors, save_dir):
    viz = Visualizer()
    viz.set_save_dir(save_dir)

    n_files_initial = len(os.listdir(save_dir))
    viz.plot_precision_recall_curve(ground_truth, recon_errors, 0.5, filename="rce")
    assert len(os.listdir(save_dir)) - n_files_initial == 1
    assert "rce.png" in os.listdir(save_dir)

    # check that there are no open figure windows
    assert len(plt.get_fignums()) == 0
