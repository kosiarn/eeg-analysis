import torch.testing
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, seq_len: int = 640):
        features = dataframe.drop(columns=['codes']).values.astype(np.float32)
        labels = dataframe['codes'].astype(int).values
        self.sequences = []
        self.sequence_labels = []
        for i in range(len(features) - seq_len):
            self.sequences.append(features[i:i+seq_len])
            self.sequence_labels.append(labels[i+seq_len-1])
        self.sequences = np.array(self.sequences)
        self.sequence_labels = np.array(self.sequence_labels)

    def __getitem__(self, idx: int):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.sequence_labels[idx])

    def __len__(self):
        return len(self.sequences)

# from preprocessing_DataReader import PreprocessingDataReader

# path = './data/physionet.org/files/eegmmidb/1.0.0'
# reader = PreprocessingDataReader(path=path)
# reader.load(patient=[1], experiment=[1, 2, 3])
# reader.normalize(norm_type="min-max")
# data_edf = reader.get()

# dataset = EEGDataset(data_edf)