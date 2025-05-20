from math import floor
import torch
from utils.channel_names import CHANNEL_NAMES
from utils.data_specs import NUM_STATES, NUM_PATIENTS
from utils.training import get_run_title, one_hot_encode_target
from dataloader import EEGDataLoader
from random import shuffle
from tqdm import tqdm

class WaveletFCNet(torch.nn.Module):
    """
    A fully connected neural network dedicated for classifying the patient's state based on EEG values put through a wavelet transform.
    """
    def __init__(self, num_channels: int, num_freqs: int):
        super(WaveletFCNet, self).__init__()
        self.fc1 = torch.nn.Linear(num_channels * num_freqs, num_channels)
        self.fc2 = torch.nn.Linear(num_channels, NUM_STATES)

    def forward(self, x: list[float]):
        x = torch.Tensor(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim = 0)
        return output

if __name__ == "__main__":
    wavelet_net = WaveletFCNet(len(CHANNEL_NAMES), 1)
    data = EEGDataLoader('../data/physionet.org/files/eegmmidb/1.0.0')
    
    optimizer = torch.optim.SGD(wavelet_net.parameters(), lr = 0.01, momentum = 0.9)
    criterion = torch.nn.CrossEntropyLoss()

    run_title = get_run_title("wavelet_fc")
    experiment_index = 5
    num_epochs = 1000
    patients = [patient + 1 for patient in range(NUM_PATIENTS - 1)]
    shuffle(patients)
    training_set = patients[:floor(0.7 * NUM_PATIENTS)]
    data.load(training_set, experiment_index)
    for epoch in range(num_epochs):
        wavelet_net.train()
        running_loss = 0.0
        print(f"Epoch {epoch}...")
        for (sample, target) in tqdm(data):
            ground_truth = one_hot_encode_target(target)
            predicted = wavelet_net(sample)
            loss = criterion(predicted, ground_truth)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 10 == 0:
            torch.save(wavelet_net.state_dict(), f"./models/{run_title}/checkpoint_e{epoch}")
