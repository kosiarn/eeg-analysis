from torch.utils.data import Dataset
from preprocessing_DataReader import PreprocessingDataReader
from preprocessing_Transform import Transformer, Wavelet
from numpy.typing import ArrayLike
from typing import Literal, override
from utils.channel_names import ChannelName, CHANNEL_NAMES

class EEGDataLoader(Dataset[tuple[list[float], float]]):
    data: PreprocessingDataReader
    def __init__(self, data_dir: str):
        self.data = PreprocessingDataReader(path = data_dir)

    def __len__(self):
        return self.data.get().shape[0] # number of rows in result dataframe

    def __getitem__(self, index: int):
        return (
                self.data.get().iloc[index, :-1].tolist(),
                self.data.get().iloc[index, -1].item()
            )

    def load(self, patient: int, experiment: int):
        self.data.load(patient, experiment)
        self.data.normalize()


class WaveletTransformDataLoader(Dataset[tuple[list[list[float]], float]]):
     data: dict[ChannelName | Literal["codes"], ArrayLike] | None = None
     loader: PreprocessingDataReader
     scales: list[float]
     transformer: Transformer
     def __init__(self, data_dir: str, scales: list[float], wavelet: Wavelet):
        self.loader = PreprocessingDataReader(path = data_dir)
        self.scales = scales
        self.transformer = Transformer(wavelet)

     def __len__(self):
         return self.data['codes'].shape[0]

     def __getitem__(self, index: int):
        sample = []
        for channel in CHANNEL_NAMES:
            sample.append(
                self.data[channel][:,index].tolist()
            )
        return (
                sample,
                self.data["codes"][index].item()
            )

     def load(self, patient: int, experiment: int):
        self.loader.load(patient, experiment)
        self.loader.normalize()
        self.data = self.transformer.CWTTransform(self.loader.get(), self.scales, "all")


if __name__ == "__main__":
    loader = WaveletTransformDataLoader('../data/physionet.org/files/eegmmidb/1.0.0', [0.1, 0.2, 0.5, 0.9], Wavelet.CGAU4)
    loader.load(1, 1)
    print(loader.__getitem__(1))
