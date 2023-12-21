import logging
import math
from pathlib import Path
from typing import Type

import torch

from src.anomaly_predictor.modeling.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training if val loss doesn't improve beyond delta after given patience."""

    def __init__(
        self, patience: int = 5, min_delta: float = 0, path: Path = "checkpoint.pt"
    ):
        """Initializes necessary parameters for early stopping.

        Args:
            patience (int, optional): How long to wait after last time validation
                loss improved. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored quantity to
                qualify as an improvement. Defaults to 0.
            path (Path, optional): Path for the checkpoint to be saved to. Defaults
                to "checkpoint.pt".
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = min_delta
        self.path = path

    def __call__(self, val_loss: float, model: Type[BaseModel], epoch: int):
        """Enables class to behave like function when called like function.
        When called, checks that val loss has improved by delta against best
        score thus far. If not, it increments counter by 1. Early stop switch
        is True when counter exceeds patience or when val loss is nan / inf.

        Args:
            val_loss (float): val loss recorded at current epoch.
            model (Type[BaseModel]): subclass instance of BaseModel, e.g. LSTMAutoencoder.
            epoch (int): current epoch.
        """

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif math.isnan(score):
            self.early_stop = True
            logger.info("Early stopping triggered due to NaN value in val loss...")
        elif math.isinf(score):
            self.early_stop = True
            logger.info("Early stopping triggered due to infinity value in val loss...")
        else:
            self.best_score = score
            self.save_checkpoint(model, epoch)
            self.counter = 0

    def save_checkpoint(self, model: Type[BaseModel], epoch: int):
        """Saves model when validation loss decrease.

        Args:
            model (Type[BaseModel]): subclass instance of BaseModel, e.g. LSTMAutoencoder.
            epoch (int): current epoch.
        """
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_loss": -self.best_score,
                "epoch": epoch,
            },
            self.path,
        )
