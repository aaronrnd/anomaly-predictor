import json
from pathlib import Path

from joblib import load

from src.anomaly_predictor.utils import export_artifact


class BaseModel:
    """Parent class for models in code base."""

    def __init__(self):
        self.model = None
        self.params = None

    def get_params(self) -> dict:
        """Get parameters for model.

        Returns:
            dict: a dictionary of parameter names mapped to their values.
        """
        return self.model.get_params()

    def save_model(self, model_dir: Path, model_name: str) -> None:
        """Exports model to model_dir and saves it under model_name, additionally
        exports model's parameters as model_params.json.

        Args:
            model_dir (Path): directory where model is stored.
            model_name (str): name which model is saved under.
        """
        export_artifact(self.model, model_dir, model_name)

        with open(Path(Path(model_dir) / "model_params.json"), "w") as filepath:
            json.dump(self.get_params(), filepath)

    def load_model(self, path: Path) -> None:
        """Loads model from path.

        Args:
            path (Path): path where model is stored.
        """
        self.model = load(path)
