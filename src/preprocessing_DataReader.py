from tqdm import tqdm
from utils.channel_names import CHANNEL_NAMES
import numpy as np
import pandas as pd
import mne
from typing import Union, List
import os

class PreprocessingDataReader:
    def __init__(self, path: str, patient: Union[int|List[int]]) -> None:
        self.path = path
        self.patient = patient if isinstance(patient, list) else [patient]
        self.data = []

    def _get_path(self, patient: int) -> str:
        return os.path.join(self.path, f"S{patient:03d}")


    def _read_edf_file(self, file_path: int) -> pd.DataFrame:
        patient_path = self._get_path(file_path)
        if not os.path.exists(patient_path):
            print(f"Patient folder not found: {patient_path}")

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True)
            annotations = raw.annotations
            edf_data = pd.DataFrame(
                raw.get_data(),
                columns=CHANNEL_NAMES,
            )
            return edf_data
        except Exception as e:
            print(f"Could not read {file_path}: {e}")


    def load(self):
        pass

    def normalize(self):
        pass

if __name__ == '__main__':
    path = '../data/physionet.org/files/eegmmidb/1.0.0'
    data = PreprocessingDataReader(path=path, patient=[1, 2])
    patients_data = data._read_edf_file(1)
