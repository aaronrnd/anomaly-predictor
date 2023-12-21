# pylint: skip-file

import copy

import numpy as np
import pandas as pd
import pytest

from src.anomaly_predictor.modeling.evaluation.evaluation_metrics import ModelEvaluator


@pytest.fixture
def ground_truth():
    date_range = pd.date_range("2021-07-01", periods=10, freq="H")
    y_true = pd.Series([0, 1, 0, 0, 1, 0, 0, 0, 1, 1], index=date_range, name="Anomaly")
    return y_true


@pytest.fixture
def predictions():
    date_range = pd.date_range("2021-07-01", periods=10, freq="H")
    y_pred = pd.Series([0, 0, 0, 1, 1, 0, 1, 0, 0, 1], index=date_range, name="Anomaly")
    return y_pred


def test_modelEvaluator(ground_truth, predictions):
    pointwise = ModelEvaluator()
    recall, f1, fpr = pointwise.evaluate(ground_truth, predictions)
    # check that return type is correct and of correct range
    for score in (recall, f1, fpr):
        assert isinstance(score, float)
        assert 0 <= score <= 1

    # check that returned value is correct
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)
    assert fpr == pytest.approx(0.333, abs=1e-3)

    overlap = ModelEvaluator("overlap")
    recall, f1, fpr = overlap.evaluate(ground_truth, predictions)
    # check that return type is correct and of correct range
    for score in (recall, f1, fpr):
        assert isinstance(score, float)
        assert 0 <= score <= 1

    # check that returned value is correct
    assert recall == pytest.approx(0.666, abs=1e-3)
    assert f1 == pytest.approx(0.666, abs=1e-3)
    assert fpr == pytest.approx(0.2, abs=1e-3)


def test_generate_time_segments(predictions):
    pred_segments = ModelEvaluator._generate_time_segments(predictions.to_frame())
    assert isinstance(pred_segments, list)
    assert len(pred_segments) == 3
    anomaly_segments = pred_segments
    for range in anomaly_segments:
        assert isinstance(range, tuple)
        assert len(range) == 2
        for item in range:
            assert isinstance(item, np.int64)


def test_get_true_negatives():

    tn = ModelEvaluator._get_true_negatives(
        [[(0, 2)], [(2, 4)], [(4, 6)]], last_index=6
    )
    assert tn == 0

    # Test with leaving 1 empty range
    test_ranges = [
        [[], [(2, 4)], [(4, 6)]],
        [[(0, 2)], [], [(4, 6)]],
        [[(0, 2)], [(2, 4)], []],
    ]
    for test_range in test_ranges:
        tn = ModelEvaluator._get_true_negatives(test_range, last_index=6)
        assert tn == 1

    tn = ModelEvaluator._get_true_negatives([[], [], []], last_index=5)
    assert tn == 1

    with pytest.raises(Exception) as e_info:
        ModelEvaluator._get_true_negatives([[(0, 2)], [(2, 4)], [(4, 6)]], last_index=5)
