import os
import pickle
import pathlib

import torch
import torch.nn.functional as F
import numpy as np

WESAD_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'raw', 'WESAD')


class WESADDataset(torch.utils.data.Dataset):

    def __init__(self, idx: int, sequence_length: int) -> None:
        super().__init__()
        self.seq_length = sequence_length
        self.X, self.Y = load_wesad(idx)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i + self.seq_length >= len(self):
            i = len(self) - self.seq_length

        return self.X[i:i+self.seq_length], self.Y[i:i+self.seq_length]
          

def load_wesad(idx):
    assert  2 < idx < 18
    with open(os.path.join(WESAD_PATH, f'S{idx}', f'S{idx}.pkl'), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    X = np.stack([
        data['signal']['chest']['ACC'],
        data['signal']['chest']['Resp'],
        data['signal']['chest']['EDA'],
        data['signal']['chest']['ECG'],
        data['signal']['chest']['EMG'],
        data['signal']['chest']['Temp'],
    ], axis=1)
    Y = data['signal']['chest'], data['label']
    X = X[(Y>0) & (Y<5)]
    X = torch.tensor((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    Y = Y[(Y>0) & (Y<5)]
    Y = F.one_hot(Y-1, num_classes=4)
    return X, Y
