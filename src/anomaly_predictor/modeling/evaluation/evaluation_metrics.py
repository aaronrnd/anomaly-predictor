from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


class ModelEvaluator:
    """Evaluates model given predictions and ground truth."""

    def __init__(self, return_mode: str = "pointwise"):
        """Initializes parameter for ModelEvaluator which influences type of metrics
        returned when main method `evaluate` is called.

        Args:
            return_mode (str, optional): Expects values "pointwise" or "overlap".
                Defaults to "pointwise".
        """
        self.return_mode = return_mode

    def evaluate(
        self, ground_truth: pd.Series, predictions: pd.Series
    ) -> Tuple[float, float, float]:
        """Evaluates the recall, f1_score, false positive rate of predictions
        against ground truth.

        Args:
            ground_truth (pd.Series): Actual labels of whether point is anomalous
                (or forecasted to be if curve shifting was done).
            predictions (pd.Series): Predictions on whether point is anomalous
                (or forecasted to be if curve shifting was done).

        Returns:
            Tuple[float, float, float]: a tuple representing the calculated recall,
                f1_score, false positive rate.
        """
        recall = self._recall(ground_truth, predictions, return_mode=self.return_mode)
        f1 = self._f1_score(ground_truth, predictions, return_mode=self.return_mode)
        fpr = self._fpr(ground_truth, predictions, return_mode=self.return_mode)
        return recall, f1, fpr

    def _recall(
        self,
        ground_truth: pd.Series,
        predictions: pd.Series,
        return_mode: str = "pointwise",
    ) -> float:
        """Evaluates the recall of predictions against ground truth.

        Args:
            ground_truth (pd.Series): Actual labels of whether point is anomalous
                (or forecasted to be if curve shifting was done).
            predictions (pd.Series): Predictions on whether point is anomalous
                (or forecasted to be if curve shifting was done).

        Returns:
            float: recall
        """
        if return_mode == "pointwise":
            return recall_score(ground_truth, predictions, zero_division=0)

        elif return_mode == "overlap":
            true_pos, _, _, false_neg = self._overlap_segment_pipeline(
                ground_truth, predictions
            )
            try:
                return true_pos / (true_pos + false_neg)
            except:
                return 0

    def _f1_score(
        self,
        ground_truth: pd.Series,
        predictions: pd.Series,
        return_mode: str = "pointwise",
    ) -> float:
        """Evaluates the f1_score of predictions against ground truth.

        Args:
            ground_truth (pd.Series): Actual labels of whether point is anomalous
                (or forecasted to be if curve shifting was done).
            predictions (pd.Series): Predictions on whether point is anomalous
                (or forecasted to be if curve shifting was done).

        Returns:
            float: f1 score
        """
        if return_mode == "pointwise":
            return f1_score(ground_truth, predictions, zero_division=0)
        elif return_mode == "overlap":
            true_pos, false_pos, _, false_neg = self._overlap_segment_pipeline(
                ground_truth, predictions
            )
            try:
                return (2 * true_pos) / (2 * true_pos + false_pos + false_neg)
            except:
                return 0

    def _fpr(
        self,
        ground_truth: pd.Series,
        predictions: pd.Series,
        return_mode: str = "pointwise",
    ) -> float:
        """Evaluates the false positive rate of predictions against ground truth.

        Args:
            ground_truth (pd.Series): Actual labels of whether point is anomalous
                (or forecasted to be if curve shifting was done).
            predictions (pd.Series): Predictions on whether point is anomalous
                (or forecasted to be if curve shifting was done).

        Returns:
            float: false positive rate
        """
        if return_mode == "pointwise":
            true_neg, false_pos, _, _ = confusion_matrix(
                ground_truth, predictions, labels=[0, 1]
            ).ravel()
            return false_pos / (false_pos + true_neg)
        if return_mode == "overlap":
            _, false_pos, true_neg, _ = self._overlap_segment_pipeline(
                ground_truth, predictions
            )
            try:
                return false_pos / (false_pos + true_neg)
            except:
                return 0

    @staticmethod
    def _generate_time_segments(data: pd.DataFrame) -> list:
        """Generate list of tuple(s) of date ranges containing ranges of continuous labels.

        Args:
            data (pd.DataFrame): Timeseries DataFrame containing labels to be segmented.

        Returns:
            list: List of tuple(s) containing the start_date and end_date for each
                continuous segment for the given DataFrame.
        """
        date_range_tuple = []
        time, val = data.index, data.values
        splits = np.where(val[1:] != val[:-1])[0] + 1
        splits = np.concatenate(([0], splits, [len(val) - 1]))
        for i in range(len(splits) - 1):
            if val[splits[i]]:
                if time[splits[i + 1]] == data.index.max():
                    date_range_tuple.append((splits[i], splits[i + 1] + 1))
                else:
                    date_range_tuple.append((splits[i], splits[i + 1]))
        return date_range_tuple

    @staticmethod
    def _overlap(expected: tuple, observed: tuple) -> bool:
        """Determines if the expected and observed time range overlaps.

        Args:
            expected (tuple): A tuple of start and end time representing a continuous
                range of anomalous time points in the ground truth.
            observed (tuple): A tuple of start and end time representing a continuous
                range of time points predicted to be anomalous.

        Returns:
            bool: Whether the expected range overlaps with the observed range.
        """
        first = expected[0] - observed[1]
        second = expected[1] - observed[0]
        return first * second < 0

    @staticmethod
    def _get_true_negatives(list_ranges: list, last_index: int) -> int:
        """From a given list of identified index range determine the number of
        unidentified ranges, this unidentfied points would be the true negatives.

        Args:
            list_ranges (list): List of containing identified tp, fp, fn ranges.
            last_index (int): Last index in entire index range.

        Returns:
            int: True negatives in given range.
        """
        identified_range = []

        for ranges in list_ranges:
            identified_range.extend((ranges))
        identified_range.sort()

        # If there's no identified tp, fp or fn then the entire range should be true negative
        if len(identified_range) == 0:
            return 1

        # Check that last_index makes sense if not stop
        assert (
            identified_range[-1][1] <= last_index
        ), "Last index in identified_range is more than last_index."

        index = 0
        tn = 0
        if identified_range[-1][1] != last_index:
            tn += 1
        for start, end in identified_range:
            if start == 0:
                index = end
                continue
            else:
                if start - index > 0:
                    tn += 1
                index = end
        return tn

    def _overlap_segment(
        self, expected: list, observed: list, last_index: int
    ) -> Tuple[int, int, int, int]:
        """Compares predicted anomaly ranges with ground truth ranges and returns
        true postive, false postive, true negative and false negative counts.

        Args:
            expected (list): List of tuple(s) containing ground truth anomaly index ranges.
            observed (list): List of tuple(s) containing predicted anomaly index ranges.
            last_index (int): Last index in entire index range

        Returns:
            Tuple[int, int, int, int]: Tuple containing true postive, false postive,
                true negative, false negative counts.
        """
        tp, fp, fn = 0, 0, 0

        tp_range, fp_range, fn_range = [], [], []

        observed_copy = observed.copy()

        for expected_seq in expected:
            found = False
            for observed_seq in observed:
                if self._overlap(expected_seq, observed_seq):
                    if not found:
                        tp += 1
                        found = True
                        overlapping_seq = []
                        overlapping_seq.extend(expected_seq)
                        overlapping_seq.extend(observed_seq)
                        tp_range.append((min(overlapping_seq), max(overlapping_seq)))
                    if observed_seq in observed_copy:
                        observed_copy.remove(observed_seq)

            if not found:
                fn += 1
                fn_range.append(expected_seq)

        fp += len(observed_copy)
        fp_range.extend(observed_copy)

        tn = self._get_true_negatives([tp_range, fp_range, fn_range], last_index)

        return tp, fp, tn, fn

    def _overlap_segment_pipeline(
        self, ground_truth: pd.Series, predictions: pd.Series
    ) -> Tuple[int, int, int, int]:
        """Generates true positive, false positive, true negative and false negative
        counts given ground truth and predictions.

        Args:
            ground_truth (pd.Series): Actual labels of whether point is anomalous
                (or forecasted to be if curve shifting was done).
            predictions (pd.Series): Predictions on whether point is anomalous
                (or forecasted to be if curve shifting was done).
        Returns:
            Tuple[int, int, int, int]: Tuple containing true postive, false postive,
                true negative, false negative counts.
        """
        ground_truth_anomaly = self._generate_time_segments(ground_truth.to_frame())
        prediction_anomaly = self._generate_time_segments(predictions.to_frame())
        last_possible_index = ground_truth.index.get_loc(ground_truth.index.max())
        true_pos, false_pos, true_neg, false_neg = self._overlap_segment(
            ground_truth_anomaly, prediction_anomaly, last_possible_index + 1
        )
        return true_pos, false_pos, true_neg, false_neg
