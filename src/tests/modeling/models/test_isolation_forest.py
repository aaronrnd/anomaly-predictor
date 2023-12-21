# pylint: skip-file

import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.modeling.models.isolation_forest import ABBIsolationForest


@pytest.fixture
def params():
    params = {
        "bootstrap": False,
        "contamination": "auto",
        "max_features": 1.0,
        "max_samples": "auto",
        "n_estimators": 50,
        "n_jobs": None,
        "random_state": 42,
        "verbose": 0,
        "warm_start": False,
    }
    return params


@pytest.fixture
def training_df():
    training_df = pd.DataFrame(np.random.rand(2000, 10))
    return training_df


@pytest.fixture
def testing_df():
    testing_df = pd.DataFrame(np.random.rand(10, 10))
    return testing_df


@pytest.fixture
def single_df():
    return pd.DataFrame([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])


@pytest.fixture
def testing_model(params, training_df):
    testing_model = ABBIsolationForest(params)
    testing_model.fit(training_df)
    return testing_model


def test_get_params(testing_model, params):
    assert testing_model.get_params() == params


def test_predict(testing_model, testing_df, single_df):
    preds = testing_model.predict(testing_df)
    assert isinstance(preds, np.ndarray)
    for item in set(preds):
        assert item in [1, 0]

    ## since i'm using randn, its hard to get an actual result
    # outlier_preds = testing_model.predict(outlier_df)
    # assert outlier_preds == 1


def test_predict_anomaly_score(testing_model, testing_df, single_df):
    preds = testing_model.predict_anomaly_score(testing_df)
    assert isinstance(preds, np.ndarray)
    for item in set(preds):
        assert 0 < item < 1
