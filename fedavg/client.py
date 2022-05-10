import os
import ray

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data.seq_loader import seq_collate_fn

from .esn.readout import compute_ridge_matrices

@ray.remote
class FedAvgClient:

    def __init__(self, i):
        self.idx = i
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.c_path = None
        self.exp_path = None
        self.epochs = None
        self.lr = None

        self.dataset = None
        self.loader = None
        self.H_mat = None
        self.Y = None

        self.score_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()
        print(f"Client {i} created. Running on {self.device}.")
    
    def local_ip_update(self, reservoir_path: str):
        reservoir = torch.load(reservoir_path, map_location=self.device)
        with torch.no_grad():
            opt = torch.optim.SGD(reservoir.parameters(), lr=self.lr)
            print(f"Starting local IP for Client {self.idx}")
            for _ in range(self.epochs):
                prev_ip_a = reservoir.ip_a.data.clone()
                for x, y in self.loader:
                    reservoir(x.to(self.device))
                    opt.step()
                    reservoir.zero_grad()
                print(f"Diff for user {self.idx} == {((reservoir.ip_a.data - prev_ip_a)**2).mean()}") 
            print(f"Local IP of Client {self.idx} completed.")
        torch.save(reservoir, self.c_path)
        
        return self.c_path

    def compute_ab(self, reservoir_path: str):
        with torch.no_grad():
            reservoir = torch.load(reservoir_path, map_location=self.device)
            input = self.dataset.X.to(self.device)
            to_size = len(self.dataset)*self.dataset.seq_length
            self.H_mat = reservoir(input).reshape(to_size, -1)
            A, B = compute_ridge_matrices(self.H_mat, self.dataset.Y.reshape(to_size, -1).to(self.H_mat))
        return (A, B)
    
    def local_eval(self, readout_path = None):
        with torch.no_grad():
            readout = torch.load(readout_path, map_location=self.device)['W']
            Y_pred = torch.argmax(F.linear(self.H_mat, readout), -1).flatten()
            Y_true = torch.argmax(self.dataset.Y, dim=-1).flatten()
            n_samples = Y_true.size(0)
            acc = self.score_fn(Y_true, Y_pred)
        return (acc, n_samples)

    def _build(self, exp_path, config):
        self.exp_path = exp_path
        self.c_path = os.path.join(self.exp_path, 'clients', f'{self.idx}.pkl')
        self.epochs = config['EPOCHS']
        self.lr = config['ETA']
        if self.dataset is None:
            if config['DATASET'] == 'HHAR':
                from data.hhar import HHARDataset
                self.dataset = HHARDataset(self.idx)
            if config['DATASET'] == 'WESAD':
                from data.wesad import WESADDataset
                self.dataset = WESADDataset(self.idx)
        
        self.dataset.seq_length = config['SEQ_LENGTH']

        self.loader = DataLoader(self.dataset, batch_size=config['BATCH_SIZE'], collate_fn=seq_collate_fn)
        return True
