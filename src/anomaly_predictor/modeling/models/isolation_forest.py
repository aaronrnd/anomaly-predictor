import numpy as np
from sklearn.ensemble import IsolationForest

from src.anomaly_predictor.modeling.models.base_model import BaseModel


class ABBIsolationForest(BaseModel):
    """IsolationForest based on sklearn's."""

    def __init__(self, params: dict = None):
        """Initialize isolationforest model, with params if provided. Else,
        default setting by sklearn.

        Args:
            params (dict, optional): Specified params for isolationforest.
            Defaults to None.
        """
        super().__init__()
        if params:
            self.model = IsolationForest(**params)
        else:
            self.model = IsolationForest()

    def fit(self, training_data: np.ndarray, training_label: np.ndarray = None):
        """Fits the instantiated model with training_data.

        Args:
            training_data (np.ndarray): Training Data.
            training_label (np.ndarray, optional): Training Labels. Defaults to None.
        """
        self.model.fit(training_data, training_label)

    def predict(self, inference_data: np.ndarray) -> np.ndarray:
        """Predict inference_data using fitted model. 0 implies inlier, 1
        implies outlier.

        Args:
            inference_data (np.ndarray): Data to predict on.

        Returns:
            np.ndarray: ndarray of prediction results.
        """
        preds = self.model.predict(inference_data)
        converted_preds = np.where(preds == -1, 1, 0)
        return converted_preds

    def predict_anomaly_score(
        self, inference_data: np.ndarray, threshold: float = None
    ) -> np.ndarray:
        """Predicts inference_data using fitted model. Returns Anomaly Score
        instead of 0 and 1.

        Args:
            inference_data (np.ndarray): Data to predict on.
            threshold (float, optional): Floors anomaly scores to 0 if their absolute
                value is smaller than threshold. Defaults to None.

        Returns:
            np.ndarray: ndarray of anomaly scores.
        """
        preds = self.model.score_samples(inference_data)
        if threshold:
            floored_preds = np.where(-preds > threshold, -preds, 0)
        else:
            floored_preds = -preds
        return floored_preds
