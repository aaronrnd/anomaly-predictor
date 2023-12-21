import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.anomaly_predictor.modeling.models.base_model import BaseModel
from src.anomaly_predictor.modeling.models.early_stopping import EarlyStopping
from src.anomaly_predictor.modeling.models.lstm_ae_architecture import (
    LSTMAEArchitecture,
)
from src.anomaly_predictor.utils import copy_logs

logger = logging.getLogger(__name__)


class LSTMAutoEncoder(BaseModel):
    """Long Short Term Memory AutoEncoder for anomaly detection and forecast."""

    def __init__(
        self,
        model_params: dict = None,
        lr: float = 1e-3,
        reconstruction_error_mode: str = "last_timepoint",
        pred_threshold: float = 0.5,
        max_reconstruction_error: float = 10,
        early_stopping_params: dict = None,
    ):
        """Initializes necessary parameters for model architecture and predictions.

        Args:
            model_params (dict, optional): parameters for LSTMAE's architecture.
                Defaults to None.
            lr (float, optional): model's learning rate. Defaults to 1e-3.
            reconstruction_error_mode (str, optional): determines if reconstruction
                error is computed on last  reconstructed timepoint or all of the
                reconstructed window. Defaults to "last_timepoint".
            pred_threshold (float, optional): Empirically determined threshold
                during training. If window slice has reconstruction error above
                threshold, then it will be predicted as 1 (anomalous) else 0.
                Defaults to 0.5.
            max_reconstruction_error (float, optional): Empirically determined value.
                This should be the maximum reconstruction error encountered during
                training. Defaults to 10.
            early_stopping_params (dict, optional): Parameters for early stopping.
                If None, no early stopping is applied. Defaults to None.
        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_params:
            self.model = LSTMAEArchitecture(**model_params, device=self.device).to(
                self.device
            )
        self.lr = lr
        self.rce_mode = reconstruction_error_mode
        self.pred_threshold = pred_threshold
        self.max_reconstruction_error = max_reconstruction_error
        if early_stopping_params:
            self.early_stopping = EarlyStopping(**early_stopping_params)
        else:
            self.early_stopping = None

    def get_params(self) -> dict:
        """Get parameters for model.

        Returns:
            dict: a dictionary of parameter names mapped to their values.
        """
        return {
            "model_architecture": self.model.get_params(),
            "lr": self.lr,
            "reconstruction_error_mode": self.rce_mode,
            "pred_threshold": float(self.pred_threshold),
            "max_reconstruction_error": float(self.max_reconstruction_error),
        }

    def save_model(self, model_dir: Path, model_name: str) -> None:
        """Exports model to model_dir and saves it under model_name, additionally
        exports model's parameters as model_params.json.

        Args:
            model_dir (Path): directory where model is stored.
            model_name (str): name which model is saved under.
        """
        torch.save(self.model.state_dict(), Path(Path(model_dir) / f"{model_name}.pt"))
        with open(Path(Path(model_dir) / "model_params.json"), "w") as filepath:
            json.dump(self.get_params(), filepath)

    def load_model(self, path: Path) -> None:
        """Loads model from path and ensures model works with available device.

        Args:
            path (Path): path where model is stored.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

    def fit(
        self,
        training_data: torch.utils.data.DataLoader,
        validation_data: torch.utils.data.DataLoader = None,
        n_epochs: int = 5,
        loss_reduction: str = "mean",
        model_dir: Path = None,
    ) -> dict:
        """Trains model on training data and evaluates during training on
        validation data if provided.

        Args:
            training_data (torch.utils.data.DataLoader): DataLoader in which
                each batch is of tensor size batch_size * lookback * n_features.
                Model training will be done on this data.
            validation_data (torch.utils.data.DataLoader, optional): DataLoader
                in which each batch is of tensor size batch_size * lookback * n_features.
                Validation loss will be evaluated on this data during training.
                Defaults to None.
            n_epochs (int, optional): Number of epochs for training. Defaults to 5.
            loss_reduction (str, optional): Loss function's reduction method.
                Accepts "sum" or "mean". Defaults to "mean".
            model_dir (Path, optional): Optional path to model output directory, to
                output training logs. Defaults to None.

        Returns:
            dict: dict of keys "train" and "val" if validation_data was provided.
                Value for each key is the list of mean training loss in each epoch.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss(reduction=loss_reduction).to(self.device)
        history = dict(train=[], val=[]) if validation_data else dict(train=[])

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            train_losses = []
            with tqdm(training_data, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch} / {n_epochs}")
                for batch in training_data:
                    tepoch.update(1)
                    self.model.zero_grad()

                    batch = batch.to(self.device)
                    reconstruction = self.model(batch)

                    loss = loss_func(reconstruction, batch.float())
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())
                    tepoch.set_postfix({"Train loss": loss.item()})
                train_loss = np.mean(train_losses)
                history["train"].append(train_loss)
                tepoch.set_postfix({"Mean Train loss": train_loss})

                if validation_data:
                    self.model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch in validation_data:

                            batch = batch.to(self.device)
                            reconstruction = self.model(batch)

                            loss = loss_func(reconstruction, batch.float())
                            val_losses.append(loss.item())

                        val_loss = np.mean(val_losses)
                        history["val"].append(val_loss)
                        tepoch.set_postfix(
                            {"Mean Train loss": train_loss, "Mean Val loss": val_loss}
                        )

                    if self.early_stopping is not None:
                        self.early_stopping(val_loss, self.model, epoch)

                        if self.early_stopping.early_stop:
                            break

            logger.info(f"Completed epoch {epoch} / {n_epochs}")
            if model_dir:
                copy_logs(model_dir)

        if self.early_stopping and validation_data is not None:
            checkpoint = torch.load(self.early_stopping.path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            tqdm.write(
                f"Min val loss of {checkpoint['val_loss']} achieved at epoch {epoch}"
            )
        return history

    def get_reconstruction_error(self, data: torch.utils.data.DataLoader) -> np.ndarray:
        """Given data, returns reconstruction error for each window.
        If self.rce_mode is "last_timepoint", error is computed only on the last
        row / timepoint of window. If mode is "all", error is computed on all
        points in window.

        Args:
            data (torch.utils.data.DataLoader): DataLoader in which
                each batch is of tensor size batch_size * lookback * n_features.

        Returns:
            np.ndarray: np.ndarray of reconstruction errors of length being
                no. of window slices.
        """
        errors = []
        loss_func = nn.MSELoss(reduction="none").to(self.device)

        self.model.eval()
        with torch.no_grad():
            for seq_true in data:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)

                error = loss_func(seq_pred, seq_true)
                if self.rce_mode == "all":
                    _, lookback, n_features = seq_true.shape
                    error = error.view(-1, lookback * n_features)
                    error = np.mean(error.cpu().numpy(), axis=1)
                elif self.rce_mode == "last_timepoint":
                    error = np.mean(error[:, -1, :].cpu().numpy(), axis=1)
                errors.append(error)
            errors = np.concatenate(errors)
            return errors

    def predict(
        self, data: torch.utils.data.DataLoader, recon_errors: np.ndarray = None
    ) -> np.ndarray:
        """Given data, data is reconstructed using model. If reconstruction error
        exceeds self.pred_threshold, point is predicted as anomalous.

        Args:
            data (torch.utils.data.DataLoader): DataLoader in which
                each batch is of tensor size batch_size * lookback * n_features.
            recon_errors (np.ndarray, optional): To be provided if reconstruction
                error is already gotten and hence not needed to be computed again.
                Defaults to None.

        Returns:
            np.ndarray: np.ndarray of length being no. of window slices, values being
                0 or 1 where 1 represents an anomaly prediction.
        """
        if recon_errors is None:
            reconstruction_error = self.get_reconstruction_error(data)
        else:
            reconstruction_error = recon_errors
        return np.where(reconstruction_error < self.pred_threshold, 0, 1)

    def predict_anomaly_score(
        self,
        data: torch.utils.data.DataLoader,
    ) -> np.ndarray:
        """Returns reconstruction error for each window.

        Args:
            data (torch.utils.data.DataLoader): DataLoader in which
                each batch is of tensor size batch_size * lookback * n_features.

        Returns:
            np.ndarray: np.ndarray of anomaly scores of length being no. of
                window slices.
        """
        return self.get_reconstruction_error(data)
