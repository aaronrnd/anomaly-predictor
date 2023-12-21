"""Miscellaneous utilities/functions to assist the model training workflow, and
are specific to ABB's problem statement are to be contained here."""
from typing import Tuple, Union

import numpy as np
import pandas as pd

from src.anomaly_predictor.modeling.evaluation.evaluation_metrics import ModelEvaluator
from src.anomaly_predictor.utils import resample_hourly


def curve_shift(
    data: pd.DataFrame, column: str, lookahead_period: int, drop_anomalous: bool = False
) -> pd.DataFrame:
    """Prepares data for forecasting models. Drops ambiguous points in final lookahead window. Shifts the labels in a
    dataframe column by specified number of periods and backfills the labels. If required drops rows of anomalous data.

    Args:
        data (pd.DataFrame): DataFrame containing the column to be shifted
        column (str): Column to be shifted
        lookahead_period (int): Number of periods for data to be shifted by
        drop_anomalous (bool, optional): Drops rows with anomalous data. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with shifted data and no ambiguity in ground truth
    """
    shift_data = data.copy()
    if lookahead_period > 0:
        shift_data = drop_final_lookahead(shift_data, lookahead_period, column)
    condition = shift_data[column] == 1
    for index in shift_data[condition].index:
        shift_data.loc[
            (index - pd.to_timedelta(abs(lookahead_period), unit="h")) : index, column
        ] = 1
        if drop_anomalous:
            shift_data.drop(index=index, inplace=True)
        shift_data.fillna(0, inplace=True)

    return shift_data


def drop_final_lookahead(
    data: pd.DataFrame, lookahead_lookback_period: int, label_col: str = "Anomaly"
) -> pd.DataFrame:
    """When lookahead window is specified for forecast problems, and if final
    lookahead window points are all normal, it is unknown if the next point is
    normal or anomalous and hence such ambiguous points should be dropped.

    Suppose we have n points and lookahead window is 3, if points n-2, n-1, n
    are all normal, these 3 points should all be excluded since we do not know
    if point n+1 is anomalous or not. However, if all 3 points are anomalous,
    they should be retained. If point n-1 is anomalous, then only the last point
    should be excluded.

    Args:
        data (pd.DataFrame): DataFrame containing an anomaly column indicated by label_col.
        lookahead_lookback_period (int): Number of periods in lookahead window.
        label_col (str, optional): Name of anomaly column. Defaults to "Anomaly".

    Returns:
        pd.DataFrame: Dataframe with no ambiguous points for forecast problems.
    """
    last_lookahead_idx = data[-lookahead_lookback_period:].index
    labels = data.loc[last_lookahead_idx, label_col].values
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    splits = np.concatenate(([0], splits))
    if labels[splits[-1]] == 0:
        data = data.drop(labels=last_lookahead_idx[splits[-1] :])
    return data


def post_process(
    data: pd.DataFrame,
    anomaly_score: np.ndarray,
    y_predict: np.ndarray,
    lookahead_period: int,
    lookback_period: int = 0,
    binarizing_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Appends predictions, prediction scores to original dataframe. If
    lookahead_period is positive, indicative of a forecast problem, then an
    additional datetime column is added to represent the end of the forecast window.
    The hourly predictions are then pooled to make up the daily prediction dataframe.

    Args:
        data (pd.DataFrame): Original data with hourly granularity
        anomaly_score (np.ndarray): Predicted anomaly scores
        y_predict (np.ndarray): Predicted anomaly
        lookahead_period (int): Shifted period used in curve_shift feature
        lookback_period (int, optional): Window size over which aggregation / exponential
            smoothing is done. This is only relevant when lookahead_period > 0. Defaults to 0.
        binarizing_threshold (float, optional): threshold for binarizing pooled / smoothed score

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]. The first contains predictions at hourly level.
        The second represents the pooling of hourly predictions as a daily score / prediction.
    """
    assert lookahead_period >= 0, "lookahead_period cannot be negative."

    daily_data = pd.DataFrame(index=np.unique(data.index.date))
    daily_data.index.name = "MEASUREMENT_TAKEN_DATE"

    if lookahead_period > 0:
        for df in (data, daily_data):
            df["FORECAST_WINDOW_END"] = df.index + pd.Timedelta(
                lookahead_period, unit="H"
            )

    data["PREDICTED_ANOMALY_SCORE"] = anomaly_score
    data["PREDICTED_ANOMALY"] = y_predict

    daily_data["DAILY_AGGREGATED_SCORE"] = (
        data["PREDICTED_ANOMALY_SCORE"].groupby(data.index.date).mean()
    )
    daily_data["DAILY_AGGREGATED_PREDICTION"] = np.where(
        np.isnan(daily_data["DAILY_AGGREGATED_SCORE"]),
        np.nan,
        np.where(daily_data["DAILY_AGGREGATED_SCORE"] < binarizing_threshold, 0, 1),
    )

    if lookahead_period > 0:
        daily_data = daily_data.rename(
            columns={
                "DAILY_AGGREGATED_SCORE": "DAILY_AGGREGATED_SCORE_WINDOW",
                "DAILY_AGGREGATED_PREDICTION": "DAILY_AGGREGATED_PREDICTION_WINDOW",
            }
        )

        data = data.rename(
            columns={
                "PREDICTED_ANOMALY_SCORE": "PREDICTED_ANOMALY_SCORE_WINDOW",
                "PREDICTED_ANOMALY": "PREDICTED_ANOMALY_WINDOW",
            }
        )
    for col in (
        "DAILY_AGGREGATED_PREDICTION",
        "DAILY_AGGREGATED_PREDICTION_WINDOW",
    ):
        if col not in daily_data.columns:
            continue
        daily_data[col] = daily_data[col].astype("Int64")
    return data, daily_data


def split_training_timeseries(data: pd.DataFrame) -> list:
    """Performs reindexing of a dataframe and splitting of dataframe on rows of
    NaNs. Returns a list containing the splitted dataframe.

    Args:
        data (pd.DataFrame): Dataframe to be reindexed and splitted.

    Returns:
        list: List containing DataFrames
    """
    data_list = []
    data = resample_hourly(data)
    if len(data) != 0:
        data_index = ModelEvaluator()._generate_time_segments(data.notna().any(axis=1))
        for item in data_index:
            data_list.append(data.iloc[item[0] : item[1]])
    return data_list


def infer_datetimeindex(data: pd.DataFrame, series: pd.Series) -> pd.Index:
    """Infers relevant datetimeindex for series from dataframe.
    If series is shorter length than dataframe, the difference in length is likely due to the usage of the LSTMAE model where the first (window_size - 1) data points would have no predictions.
    This would result in a returned index which excludes these points.
    Else, it returns index of data itself.

    Args:
        data (pd.DataFrame): Dataframe with a relevant DatetimeIndex
        series (pd.Series): Series which index is to be overwritten

    Returns:
        pd.DatetimeIndex: DatetimeIndex to be set for series.
    """
    n_series, n_data = len(series), len(data.index)
    assert (
        n_series <= n_data
    ), "Series length longer than DataFrame, can't infer DateTimeindex"
    if n_series < n_data:
        return data.index[-n_series:]
    return data.index


def return_zero_for_non_operating(
    operating_state_col: str, preds: float
) -> Union[int, float]:
    """Converts predictions to 0 when operating_state_col == 0.

    Args:
        state_col (str): Operating State of Asset.
        preds (float): Prediction Value.

    Returns:
        Union[int,float]: Prediction value according to operating_state_col.
    """
    return 0.0 if operating_state_col == 0 else preds


def floor_predictions_to_zero_for_nonoperating(
    data: pd.DataFrame, preds: np.array
) -> np.array:
    """Extract floored predictions from model's initial predictions. There should be no
    positive predictions when asset is not operating.

    Args:
        data (pd.DataFrame): Ground truth data from dataloader
        preds (np.array): Predictions from model

    Returns:
        np.array: Floored predictions
    """
    placeholder_data = data.copy()
    placeholder_data["preds"] = pd.Series(
        preds, index=placeholder_data.index[-len(preds) :]
    )
    placeholder_data["new_preds"] = placeholder_data.apply(
        lambda x: return_zero_for_non_operating(x["Asset_Operating"], x.preds), axis=1
    )
    return placeholder_data["new_preds"][-len(preds) :]
