from time import process_time
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

    @staticmethod
    def _get_codes(edf_data) -> list:
        annotations = edf_data.annotations
        codes = annotations.description
        time_array = np.array(
            [round(x, 10) for x in np.arange(0, len(edf_data) / 160, 0.00625)]
        )
        code_array = []
        counter = 0
        for timeVal in time_array:
            if timeVal in annotations.onset:
                counter += 1
            code_of_target = int(
                codes[counter - 1].replace("T", "")
            )
            code_array.append(code_of_target)
        return code_array

    def _read_edf_file(self, file_path: int) -> pd.DataFrame:
        patient_path = self._get_path(file_path)
        if not os.path.exists(patient_path):
            print(f"Patient folder not found: {patient_path}")

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True)
            edf_data = pd.DataFrame(
                raw.get_data(),
                columns=CHANNEL_NAMES,
            )
            # https://stackoverflow.com/questions/22649693/drop-rows-with-all-zeros-in-pandas-data-frame
            edf_data = edf_data[~(edf_data == 0).all(axis=1)]
            codes = self._get_codes(raw)
            edf_data["codes"] = np.array(codes)

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
    data.load()
