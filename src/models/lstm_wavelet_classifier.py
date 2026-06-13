import torch
from torch import Tensor, nn

from utils.data_specs import NUM_STATES


class LSTMWaveletClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        head_hidden_size: int = 128,
        num_classes: int = NUM_STATES,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
