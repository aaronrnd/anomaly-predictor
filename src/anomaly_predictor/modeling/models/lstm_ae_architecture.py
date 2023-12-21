import torch
import torch.nn as nn


class LSTMAEArchitecture(nn.Module):
    """LSTM Autoencoder architecture which holds the encoder and decoder layers."""

    def __init__(
        self,
        n_features: int,
        hidden_size: list,
        n_layers: list,
        dropout: float,
        device: str,
    ):
        """Initializes the parameters of model architecture.

        Example of acceptable values for n_layers and hidden_size:
        {'n_features': 23,
        'hidden_size': [[32, 16, 8], [16, 32]],
        'n_layers': [3, 2],
        'dropout': 0}

        Args:
            n_features (int): The number of expected features in the input data
            hidden_size (list): The number of features in the hidden state for
                encoder and decoder layer. The expected length of this list is 2
                where hidden_size[0] is for encoder and hidden_size[1] is for decoder.
            n_layers (list): Number of recurrent layers for encoder and decoder.
                E.g., setting n_layers=2 would mean stacking two LSTMs together
                to form a stacked LSTM, with the second LSTM taking in outputs
                of the first LSTM and computing the final results. The expected
                length of this list is 2 where n_layers[0] is for encoder and
                n_layers[1] is for decoder.
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each LSTM layer except the last layer, with dropout
                probability equal to dropout.
            device (str): Expects values "cpu" or "cuda".
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.encoder = self._build_encoder_decoder(mode="encoder").to(device)
        self.decoder = self._build_encoder_decoder(mode="decoder").to(device)
        self.output_layer = nn.Linear(self.hidden_size[1][-1], self.n_features).to(
            device
        )

    def get_params(self):
        """Get parameters for model.

        Returns:
            dict: a dictionary of parameter names mapped to their values.
        """
        return {
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }

    def _build_encoder_decoder(self, mode: str) -> nn.ModuleList:
        """Builds encoder or decoder layer depending on mode.

        Args:
            mode (str): Accepts value "encoder" or "decoder".

        Returns:
            nn.ModuleList: List of nn.LSTM layers.
        """
        if mode == "encoder":
            hidden_size, n_layers = self.hidden_size[0], self.n_layers[0]
            current_input_size = self.n_features
        elif mode == "decoder":
            hidden_size, n_layers = self.hidden_size[1], self.n_layers[1]
            current_input_size = self.hidden_size[0][-1]

        assert len(hidden_size) == n_layers
        layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.LSTM(
                input_size=current_input_size,
                hidden_size=hidden_size[i],
                num_layers=1,
                batch_first=True,
            )
            current_input_size = hidden_size[i]
            layers.append(layer)

            if self.dropout > 0:
                if mode == "encoder" and i == (n_layers - 1):
                    continue
                layers.append(nn.Dropout(p=self.dropout))
        return layers

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass of data through the module's layers.

        Args:
            data (torch.Tensor): data of size (batch_size, lookback, n_features).

        Returns:
            (torch.Tensor): reconstructed data of size (batch_size, lookback, n_features).
        """
        lookback = data.shape[1]

        # forward pass through encoder
        output = data.float()
        for layer in self.encoder[:-1]:
            if isinstance(layer, nn.Dropout):
                output = layer(output)
            else:
                output, (_, _) = layer(output)
        _, (hidden_state, _) = self.encoder[-1](output)
        output = hidden_state.squeeze(dim=0)

        # forward pass through decoder
        output = output.unsqueeze(1).repeat(1, lookback, 1)

        for layer in self.decoder:
            if isinstance(layer, nn.Dropout):
                output = layer(output)
            else:
                output, (_, _) = layer(output)

        return self.output_layer(output)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Forward propogate the data through the encoded layer. Returns an
        encoded vector.

        Args:
            data (torch.Tensor): windowed data in Tensor.

        Returns:
            torch.Tensor: encoded data in single vector.
        """
        # forward pass through encoder
        output = data.float()
        for layer in self.encoder[:-1]:
            if isinstance(layer, nn.Dropout):
                output = layer(output)
            else:
                output, (_, _) = layer(output)
        _, (hidden_state, _) = self.encoder[-1](output)
        output = hidden_state.squeeze(dim=0)

        return output
