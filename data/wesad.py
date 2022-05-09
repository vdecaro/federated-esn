import os
import pickle
import pathlib

import torch
import torch.nn.functional as F
import numpy as np

RAW_WESAD_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'raw', 'WESAD')
WESAD_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'processed', 'WESAD')

USERS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14", "15", "16", "17"]

class WESADDataset(torch.utils.data.Dataset):

    def __init__(self, idx: int, sequence_length: int) -> None:
        super().__init__()
        self.user = USERS[idx]

        u_path = os.path.join(WESAD_PATH, f'{self.user}.pkl')
        if not os.path.exists(u_path):
            self.user_data = self.preprocess()
        else:
            self.user_data = pickle.load(open(u_path, 'rb'))
        self.seq_length = sequence_length
        self.X, self.Y = self._to_sequence_chunks(self.seq_length)

    @property
    def seq_length(self):
        return self.seq_length

    @seq_length.setter
    def seq_length(self, new_length):
        print(f"Setting the length of the chunks in WESAD user {self.user}")
        self._to_sequence_chunks(new_length)
        self.seq_length = new_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
    
    def _to_sequence_chunks(self, length):
        X, Y = torch.split(self.u_data['X'], length, dim=0), torch.split(self.u_data['Y'], length, dim=0)
        if X[-1].shape[0] != length:
            X, Y = X[:-1], Y[:-1]

        return X, Y

    def preprocess(self, user):
        with open(os.path.join(RAW_WESAD_PATH, f'S{user}', f'S{user}.pkl'), 'rb') as f:
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
        u_dict = {'X': X, 'Y': Y}

        pickle.dump(u_dict, open(os.path.join(WESAD_PATH, 'users', f'{self.user}.pkl'), 'wb+'))

        return u_dict
