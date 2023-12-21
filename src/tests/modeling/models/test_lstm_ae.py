# pylint: skip-file
import copy

import numpy as np
import pytest
import torch

from src.anomaly_predictor.modeling.models.lstm_ae import LSTMAutoEncoder

train_samples = 1000
val_samples = 200
lookback = 5
n_features = 20


@pytest.fixture
def train_data():
    # shape is samples * lookback * n_features
    torch.manual_seed(0)
    data = torch.randn((train_samples, lookback, n_features))
    return torch.utils.data.DataLoader(data, batch_size=32)


@pytest.fixture
def val_data():
    # shape is samples * lookback * n_features
    torch.manual_seed(20)
    data = torch.randn((val_samples, lookback, n_features))
    return torch.utils.data.DataLoader(data, batch_size=16)


@pytest.fixture
def model_params():
    return {
        "n_layers": [2, 2],
        "hidden_size": [[32, 16], [16, 32]],
        "dropout": 0,
        "n_features": n_features,
    }


def weights_and_biases(lstmae):
    return [
        lstmae.model.encoder[0].weight_hh_l0.detach().clone(),
        lstmae.model.encoder[0].weight_ih_l0.detach().clone(),
        lstmae.model.encoder[0].bias_ih_l0.detach().clone(),
        lstmae.model.encoder[0].bias_hh_l0.detach().clone(),
        lstmae.model.encoder[1].weight_hh_l0.detach().clone(),
        lstmae.model.encoder[1].weight_ih_l0.detach().clone(),
        lstmae.model.encoder[1].bias_ih_l0.detach().clone(),
        lstmae.model.encoder[1].bias_hh_l0.detach().clone(),
        lstmae.model.decoder[0].weight_hh_l0.detach().clone(),
        lstmae.model.decoder[0].weight_ih_l0.detach().clone(),
        lstmae.model.decoder[0].bias_ih_l0.detach().clone(),
        lstmae.model.decoder[0].bias_hh_l0.detach().clone(),
        lstmae.model.decoder[1].weight_hh_l0.detach().clone(),
        lstmae.model.decoder[1].weight_ih_l0.detach().clone(),
        lstmae.model.decoder[1].bias_ih_l0.detach().clone(),
        lstmae.model.decoder[1].bias_hh_l0.detach().clone(),
    ]


@pytest.mark.parametrize("n_epochs", [1, 3])
def test_fit(model_params, train_data, val_data, n_epochs):
    # initialize the model
    lstmae = LSTMAutoEncoder(model_params=model_params)
    initial_weights = weights_and_biases(lstmae)

    losses = lstmae.fit(train_data, val_data, n_epochs=n_epochs)
    post_training_weights = weights_and_biases(lstmae)

    # Check that model weights and biases have changed
    for initial, post in zip(initial_weights, post_training_weights):
        assert not torch.equal(initial, post)

    # Check that return type is correct
    assert isinstance(losses, dict)

    # Check that there are 2 keys for train and val
    assert len(losses.keys()) == 2

    # Check that length of losses is of correct type and equal to number of epochs
    for key in losses:
        assert len(losses[key]) == n_epochs
        assert isinstance(losses[key], list)

    # check that method runs even when only train_data is provided
    model = LSTMAutoEncoder(model_params=model_params)
    losses = model.fit(train_data, n_epochs=n_epochs)
    assert len(losses.keys()) == 1


@pytest.mark.parametrize("mode", ["all", "last_timepoint"])
def test_get_reconstruction_error(model_params, train_data, val_data, mode):
    # initialize and fit the model
    lstmae = LSTMAutoEncoder(model_params=model_params, reconstruction_error_mode=mode)
    lstmae.fit(train_data)

    post_training_weights = weights_and_biases(lstmae)
    errors = lstmae.get_reconstruction_error(val_data)

    # Check return type and length is as expected
    assert isinstance(errors, np.ndarray)
    assert len(errors) == val_samples

    # Check that weights remain the same after reconstruction
    post_pred_weights = weights_and_biases(lstmae)
    for post_train, post_pred in zip(post_training_weights, post_pred_weights):
        assert torch.equal(post_train, post_pred)


@pytest.mark.parametrize("threshold", [0.3, 0.7, 1.5])
def test_predict(model_params, train_data, val_data, threshold):
    # initialize and fit the model
    lstmae = LSTMAutoEncoder(model_params=model_params, pred_threshold=threshold)
    lstmae.fit(train_data)

    pred = lstmae.predict(val_data)

    # Check return type, length and shape is as expected
    assert isinstance(pred, np.ndarray)
    assert len(pred) == val_samples
    assert pred.shape == (val_samples,)

    # Check returned values is 0 or 1
    assert (np.isin(pred, [0, 1])).all()


@pytest.mark.parametrize("max_error", [0.5, 0.7, 3])
def test_predict_anomaly_score(model_params, train_data, val_data, max_error):
    # initialize and fit the model
    lstmae = LSTMAutoEncoder(
        model_params=model_params, max_reconstruction_error=max_error
    )
    lstmae.fit(train_data)
    scores = lstmae.predict_anomaly_score(val_data)

    # Check return type, length and shape is as expected
    assert isinstance(scores, np.ndarray)
    assert len(scores) == val_samples
    assert scores.shape == (val_samples,)
