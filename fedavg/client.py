import os
import ray

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data.seq_loader import seq_collate_fn

from .esn.readout import compute_ridge_matrices
from .utils import empty_cache

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
        self.train_pred_lab = []
        self.Y = None

        self.score_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()
        print(f"Client {i} created. Running on {self.device}.")
    
    def local_ip_update(self, reservoir_path: str):
        reservoir = torch.load(reservoir_path, map_location=self.device)
        with torch.no_grad():
            opt = torch.optim.SGD(reservoir.parameters(), lr=self.lr)
            reservoir.train()
            for _ in range(self.epochs):
                for x, y in self.loader:
                    reservoir(x.to(self.device))
                    reservoir.ip_a.grad = F.normalize(reservoir.ip_a.grad, dim=0)
                    reservoir.ip_b.grad = F.normalize(reservoir.ip_b.grad, dim=0)
                    opt.step()
                    reservoir.zero_grad()
        torch.save(reservoir, self.c_path)
        empty_cache()
        return self.c_path

    def compute_ab(self, reservoir_path: str):
        with torch.no_grad():
            reservoir = torch.load(reservoir_path, map_location=self.device)
            A, B = None, None
            reservoir.eval()
            for x, y in self.loader:
                h = reservoir(x.to(self.device)).reshape((-1, reservoir.hidden_size))
                y_reshaped = y.reshape((-1, y.size(-1))).to(h)
                a_batch, b_batch = compute_ridge_matrices(h, y_reshaped)
                A = A + a_batch if A is not None else a_batch
                B = B + b_batch if B is not None else b_batch
                self.train_pred_lab.append((h.to('cpu'), y_reshaped.to('cpu')))
        empty_cache()
        return (A, B)
    
    def local_eval(self, readout_path = None):
        with torch.no_grad():
            readout = torch.load(readout_path, map_location=self.device)['W']
            acc, n_samples = 0, 0
            for h, y in self.train_pred_lab:
                Y_pred = torch.argmax(F.linear(h.to(self.device), readout), -1).flatten().to('cpu')
                Y_true = torch.argmax(y, dim=-1).flatten()
                curr_acc = self.score_fn(Y_true, Y_pred)
                curr_n_samples = Y_true.size(0)
                acc += curr_acc * curr_n_samples
                n_samples += curr_n_samples
            del self.train_pred_lab
            self.train_pred_lab = []
        empty_cache()
        return (acc / n_samples, n_samples)

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
