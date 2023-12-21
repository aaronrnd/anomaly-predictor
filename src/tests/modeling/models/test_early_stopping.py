import pytest
import torch

from src.anomaly_predictor.modeling.models.early_stopping import EarlyStopping
from src.anomaly_predictor.modeling.models.lstm_ae import LSTMAutoEncoder


@pytest.fixture
def model():
    model_params = {
        "n_layers": [2, 2],
        "hidden_size": [[32, 16], [16, 32]],
        "dropout": 0,
        "n_features": 20,
    }
    return LSTMAutoEncoder(model_params=model_params)


@pytest.fixture
def val_losses_fixed():
    return [
        0.6811,
        0.6941,
        0.6532,
        0.6546,
        0.6534,
        0.6489,
        0.6240,
        0.5685,
        0.5544,
        0.5921,
        0.5831,
        0.5756,
        0.5753,
        0.5677,
        0.6666,
        0.5544,
        0.5921,
        0.5831,
        0.5756,
        0.5753,
    ]


@pytest.mark.parametrize("patience", [5, 10])
def test_early_stopping(val_losses_fixed, patience, model):
    early_stopping = EarlyStopping(patience=patience, min_delta=0)

    stopped_epoch = None
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        val_loss = val_losses_fixed[epoch - 1]
        early_stopping(val_loss, model.model, epoch)

        if early_stopping.early_stop:
            stopped_epoch = epoch
            break

    if patience == 5:
        assert stopped_epoch == 14

    elif patience == 10:
        assert stopped_epoch is None
