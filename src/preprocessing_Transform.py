import pywt
from enum import Enum
import logging
from utils.channel_names import ChannelName, CHANNEL_NAMES
from typing import Literal
from pandas import DataFrame
from numpy.typing import ArrayLike
logger = logging.getLogger("waveletTransform")

class Wavelet(Enum):
    CGAU4 = "cgau4"

class Transformer:
    wavelet: Wavelet

    def __init__(self, wavelet: Wavelet):
        self.wavelet = wavelet
    
    def CWTTransform(
            self,
            data: DataFrame,
            scales: list[float],
            channels: list[ChannelName] | Literal["all"] = "all"
            ) -> dict[ChannelName | Literal["codes"], ArrayLike] | ValueError:
        target_channels: list[ChannelName] = CHANNEL_NAMES if channels == "all" else channels
        transformed: dict[ChannelName | Literal["codes"], ArrayLike] = {}
        transformed["codes"] = data["codes"]
        for channel in target_channels:
            coef, _freqs = pywt.cwt(data[channel], scales, self.wavelet.value)
            transformed[channel] = coef
        return transformed

