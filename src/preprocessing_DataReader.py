import numpy as np
import pandas as pd
import mne
from typing import Union, List
import os

class PreprocessingDataReader:
    def __init__(self, path: str, patient: Union[int|List[int]]) -> None:
        self.path = path
        self.patent = patient if isinstance(patient, list) else [patient]
        self.data = []

    def _get_path(self, patient: int) -> str:
        try:
            os.path.join(self.path, )
        except FileNotFoundError:
            pass

    def _read_edf_file(self):
        pass

    def load(self):
        pass

if __name__ == '__main__':
    path = '../data/physionet.org/files/eegmmidb/1.0.0'
    data = PreprocessingDataReader(path=path, patient=[1, 2])
