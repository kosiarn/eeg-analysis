from fontTools.qu2cu.qu2cu import elevate_quadratic
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from utils.channel_names import CHANNEL_NAMES
import numpy as np
import pandas as pd
import mne
from typing import Union, List
import os

class PreprocessingDataReader:
    """
    A class for reading, preprocessing, and normalizing EEG data from EDF files
    in the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB).

    Attributes:
        path (str): Base path to the EEG dataset directory.
        patient (List[int]): List of patient IDs to load.
        experiment (List[int]): List of experiment IDs to load.
        edf_data (pd.DataFrame): Combined EEG data with associated codes.
        data (List): Placeholder list, currently unused.
    """
    def __init__(self, path: str) -> None:
        """
        Construct the directory path for a specific patient.
        Args:
            patient (int): Patient ID.
        Returns:
            str: Path to the patient's data folder.
        """
        self.path = path
        self.patient = None
        self.experiment = None
        self.edf_data = pd.DataFrame()
        self.data = []

    def _get_path(self, patient: int) -> str:
        """
        Construct the file path for a specific patient's data.

        :param patient: Patient ID (e.g., 1 for S001).
        :return: Path to the patient's folder.
        """
        return os.path.join(self.path, f"S{patient:03d}")

    @staticmethod
    def _get_codes(edf_data) -> list:
        """
        Extracts event codes from EDF annotations and aligns them with EEG time samples.
        Available codes are: T0, T1, T2, T3

        :param edf_data: MNE Raw object containing EEG data and annotations.
        :return: List of stimulus codes aligned with EEG data.
        """
        annotations = edf_data.annotations
        codes = annotations.description
        n_samples = edf_data.n_times                # liczba próbek EEG
        sfreq = edf_data.info['sfreq']              # Częstotliwość 160Hz

        time_array = np.array([round(x, 10) for x in np.arange(0, n_samples / sfreq, 0.00625)])
        code_array = []
        counter = 0
        for timeVal in time_array:
            if timeVal in annotations.onset:
                counter += 1
            code_of_target = codes[counter - 1].replace("T", "")
            code_array.append(code_of_target)
        return code_array

    def _read_edf_file(self, file_path: str) -> pd.DataFrame:
        """
        Reads a single EDF file and extracts EEG data.

        :param file_path: Full path to the EDF file.
        :return: EEG data and codes in a DataFrame.
        """
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            edf_data = pd.DataFrame(
                raw.get_data().T,
                columns=CHANNEL_NAMES,
            )
            edf_data = edf_data.drop(
                set(edf_data.columns) - set(CHANNEL_NAMES)
            )
            codes = self._get_codes(raw)
            edf_data["codes"] = np.array(codes).T
            # https://stackoverflow.com/questions/22649693/drop-rows-with-all-zeros-in-pandas-data-frame
            edf_data = edf_data[~(edf_data == 0).all(axis=1)]

            return edf_data
        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    def load(self, patient: Union[int|List[int]], experiment: Union[int|List[int]]) -> pd.DataFrame:
        """
        Loads EEG data for one or more patients and experiments.

        :param patient: Patient ID or list of patient IDs to load.
        :param experiment: Experiment ID or list of experiment IDs to load.
        """
        self.patient = patient if isinstance(patient, list) else [patient]
        self.experiment = experiment if isinstance(experiment, list) else [experiment]
        experiments = [f"R{e:02d}" for e in self.experiment]

        for patient in self.patient:
            patient_path = self._get_path(patient)
            if not os.path.exists(patient_path):
                print(f"Patient folder not found: {patient_path}")
            for file in tqdm(os.listdir(patient_path), desc=f"Reading EDF rom patient: {patient:03d}"):
                if file.endswith(".edf") and any(exp in file for exp in experiments):
                    file_path = os.path.join(patient_path, file)
                    patient_data = self._read_edf_file(file_path)
                    self.edf_data = pd.concat([self.edf_data, patient_data], ignore_index=True)


    def normalize(self, norm_type: str = "min-max"):
        """
        Normalizes the EEG data using the specified method.

        :param norm_type: Normalization method; either 'min-max' or 'z-score'.
        :raises ValueError: If an unsupported normalization type is provided.
        """
        if self.edf_data.empty:
            print("No data loaded. Please call load() before normalize().")

        column_names = self.edf_data.columns
        if norm_type == "min-max":
            scaler = MinMaxScaler()
        elif norm_type == "z-score":
            scaler = StandardScaler()
        else:
            raise ValueError("Normalization type must be 'min-max' or 'z-score'.")

        self.edf_data[column_names] = scaler.fit_transform(self.edf_data[column_names])

    def get(self):
        """
        Returns the preprocessed and normalized EEG dataset.

        :return: Final EEG data with channels and stimulus codes.
        """
        return self.edf_data

if __name__ == '__main__':
    path = '../data/physionet.org/files/eegmmidb/1.0.0'
    data = PreprocessingDataReader(path=path)
    data.load(patient=[1], experiment=[1, 2, 3])
    data.normalize(norm_type="min-max")
    data_edf = data.get()
