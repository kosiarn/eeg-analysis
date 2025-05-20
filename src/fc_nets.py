import torch
from utils.channel_names import CHANNEL_NAMES
from utils.data_specs import NUM_STATES
from dataloader import EEGDataLoader

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
    data.load(1, 1)
    test_sample = data.__getitem__(1)[0]
    output = wavelet_net(test_sample)
    print(output)
    
