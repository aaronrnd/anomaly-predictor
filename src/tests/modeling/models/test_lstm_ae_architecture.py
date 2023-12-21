# pylint: skip-file

import pytest
import torch

from src.anomaly_predictor.modeling.models.lstm_ae_architecture import (
    LSTMAEArchitecture,
)


@pytest.fixture
def data():
    # shape is samples * lookback * n_features
    return torch.randn((8, 5, 20))


@pytest.fixture
def data_2():
    # shape is samples * lookback * n_features
    return torch.randn((1, 5, 20))


@pytest.fixture
def params(data):
    return {
        "n_layers": [2, 4],
        "hidden_size": [[64, 32], [8, 16, 32, 64]],
        "dropout": 0,
        "n_features": data.shape[2],
    }


@pytest.fixture
def params_with_dropout(data):
    return {
        "n_layers": [2, 4],
        "hidden_size": [[64, 32], [8, 16, 32, 64]],
        "dropout": 0.3,
        "n_features": data.shape[2],
    }


@pytest.fixture
def model(params):
    return LSTMAEArchitecture(**params, device="cpu")


@pytest.fixture
def model_with_dropout(params_with_dropout):
    return LSTMAEArchitecture(**params_with_dropout, device="cpu")


def test_get_params(model, params):
    assert model.get_params() == params


@pytest.mark.parametrize(
    "lstmae,df",
    [
        ("model", "data"),
        ("model", "data_2"),
        ("model_with_dropout", "data"),
        ("model_with_dropout", "data_2"),
    ],
)
def test_forward(df, lstmae, request):
    test_data = request.getfixturevalue(df)
    output = (request.getfixturevalue(lstmae)).forward(test_data)
    assert isinstance(output, torch.Tensor)
    assert output.shape == test_data.shape
