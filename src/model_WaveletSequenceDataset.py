from typing import Literal

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import Dataset

from utils.channel_names import CHANNEL_NAMES, ChannelName


TransformedCWT = dict[ChannelName | Literal["codes"], ArrayLike]


class WaveletSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        transformed: TransformedCWT,
        seq_len: int = 640,
        stride: int = 1,
        drop_code: int = 0,
        label_offset: int = 1,
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        codes = np.asarray(transformed["codes"]).astype(np.int64)
        keep_mask = codes != drop_code
        mapped_labels = codes[keep_mask] - label_offset
        if mapped_labels.min(initial=0) < 0:
            raise ValueError("Mapped labels contain negative values; check drop_code/label_offset")

        channel_mats: list[np.ndarray] = []
        for ch in CHANNEL_NAMES:
            cwt = np.asarray(transformed[ch], dtype=np.float32)
            channel_mats.append(cwt[:, keep_mask])

        stacked = np.stack(channel_mats, axis=0)
        x = np.transpose(stacked, (2, 0, 1)).reshape(stacked.shape[2], -1)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(mapped_labels.astype(np.int64))
        self.seq_len = seq_len
        self.stride = stride

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("Feature/label length mismatch")

        if self.x.shape[0] < self.seq_len:
            self.num_sequences = 0
        else:
            self.num_sequences = 1 + (self.x.shape[0] - self.seq_len) // self.stride

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        x_seq = self.x[start:end]
        y = self.y[end - 1]
        return x_seq, y
