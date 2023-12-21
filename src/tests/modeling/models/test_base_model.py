# pylint: skip-file
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.anomaly_predictor.modeling.models.base_model import BaseModel


@pytest.fixture
def dt_params():
    return {"random_state": 8642, "max_depth": 3}


@pytest.fixture
def dt_model(dt_params):
    model = DecisionTreeClassifier(**dt_params)
    np.random.seed(18)
    training_data = pd.DataFrame(np.random.rand(200, 10))
    training_labels = pd.DataFrame(np.random.randint(2, size=200))
    model.fit(training_data, training_labels)
    return model


@pytest.fixture(scope="session")
def model_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("models"))


def test_save_model(dt_model, dt_params, model_dir):
    abb_model = BaseModel()
    abb_model.model = dt_model
    abb_model.params = dt_params

    # Check no other joblib files before save_model
    assert len(glob(str(Path(model_dir / "*.joblib")))) == 0
    abb_model.save_model(model_dir, "decisiontree")

    # Check 1 joblib file after save_model
    assert len(glob(str(Path(model_dir / "*.joblib")))) == 1


def test_load_model(model_dir):
    filepath = Path(model_dir / "decisiontree.joblib")
    abb_model2 = BaseModel()
    abb_model2.load_model(filepath)

    # Check that the model is as we had previously initialized
    dt_params = abb_model2.model.get_params()
    assert dt_params["random_state"] == 8642
    assert dt_params["max_depth"] == 3
