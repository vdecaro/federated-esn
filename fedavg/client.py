import os
import ray

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data.seq_loader import seq_collate_fn

from esn.readout import compute_ridge_matrices

@ray.remote(num_cpus=2, num_gpus=0.25)
class FedAvgClient(object):

    def __init__(self, i):
        self.idx = i
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.res_client_path = None
        self.exp_path = None
        self.epochs = None
        self.lr = None

        self.dataset = None
        self.loader = None
        self.H_mat = None
        self.Y = None

        self.score_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()
    
    @torch.no_grad
    def local_ip_update(self, reservoir_path: str):
        reservoir = torch.load(reservoir_path).to(self.device)

        opt = torch.optim.SGD(reservoir.parameters(), lr=self.lr)
        print(f"Starting local IP for Client {self.idx}")
        for _ in range(self.epochs):
            for batch in self.loader:
                reservoir(batch.to(self.device))
                opt.step()
                reservoir.zero_grad()   
        print(f"Local IP of Client {self.idx} completed.")
        torch.save(reservoir, os.path.join(self.exp_path, 'clients', f'{self.idx}.pkl'))
        
        return self.res_client_path

    @torch.no_grad
    def compute_ab(self, reservoir_path: str):
        reservoir = torch.load(reservoir_path).to(self.device)
        input = torch.stack(self.dataset.X, dim=1).to(self.device)
        self.H_mat = reservoir(input).reshape((len(self.dataset)*self.dataset.seq_length, reservoir.hidden_size))

        return compute_ridge_matrices(self.H_mat, self.dataset.Y)
    
    @torch.no_grad
    def local_eval(self, readout_path = None):
        readout = torch.load(readout_path)['W'].to(self.device)
        Y_pred = F.linear(self.H_mat, readout).flatten()
        Y_true = self.dataset.Y.flatten()
        n_samples = Y_true.size(0)
        return self.score_fn(Y_true, Y_pred), n_samples

    def _build(self, exp_path, config):
        self.exp_path = exp_path
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
