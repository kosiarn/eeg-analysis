import pywt
from enum import Enum
from utils.channel_names import ChannelName, CHANNEL_NAMES
from typing import Literal
from pandas import DataFrame
from numpy.typing import ArrayLike

class Wavelet(Enum):
    CGAU4 = "cgau4"

class Transformer:
    """The transformer class. It's capable of transforming a time series by chosen wavelet function."""
    wavelet: Wavelet

    def __init__(self, wavelet: Wavelet):
        """:param wavelet: Type of the wavelet function."""
        self.wavelet = wavelet
    
    def CWTTransform(
            self,
            data: DataFrame,
            scales: list[float],
            channels: list[ChannelName] | Literal["all"] = "all"
            ) -> dict[ChannelName | Literal["codes"], ArrayLike]:
        """
        Performs a wavelet transform on supplied data.
        :param data: A dataframe that consists of columns with names found in CHANNEL_NAMES and numerical values. It should also include a column named `codes`; it's returned without the transformation applied
        :param scales: a list of scales that will be applied to the wavelet function before transforming the data with it. Every row in channel's output corresponds to a value from this list; more info: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#pywt.cwt -> `scales` parameter
        :param channels: A list of channels that should undergo the transform. The output only consists of results for these channels and the `codes` column.
        """
        target_channels: list[ChannelName] = CHANNEL_NAMES if channels == "all" else channels
        transformed: dict[ChannelName | Literal["codes"], ArrayLike] = {}
        transformed["codes"] = data["codes"]
        for channel in target_channels:
            coef, _ = pywt.cwt(data[channel], scales, self.wavelet.value)
            transformed[channel] = coef
        return transformed

if __name__ == "__main__":
    import preprocessing_DataReader as data_reader
    data = data_reader.PreprocessingDataReader("data/physionet.org/files/eegmmidb/1.0.0")
    data.load([1], [1])
    transformer = Transformer(Wavelet.CGAU4)
    transformed = transformer.CWTTransform(data.get(), scales = [1,2,3], channels = "all")
    print(transformed.keys())
