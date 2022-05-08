import os
import ray
import torch
import torch.nn.functional as F

from esn.readout import compute_ridge_matrices

@ray.remote(num_cpus=2, num_gpus=0.25)
class FedAvgClient(object):

    def __init__(self):
        self.idx = None
        self.res_client_path = None
        self.exp_path = None
        self.epochs = None
        self.batch_size = None
        self.lr = None

        self.train_X, self.train_H, self.train_Y = None
        self.batch_loader = None
    
    @torch.no_grad
    def local_ip_update(self, reservoir_path: str):
        reservoir = torch.load(reservoir_path)

        opt = torch.optim.SGD(reservoir.parameters(), lr=self.lr)
        print(f"Starting local IP for Client {self.idx}")
        for _ in range(self.epochs):
            for batch in self.batch_loader:
                reservoir(batch)
                opt.step()
                reservoir.zero_grad()   
        print(f"Local IP of Client {self.idx} completed.")
        torch.save(reservoir, self.res_client_path)
        
        return self.res_client_path

    @torch.no_grad
    def compute_ab(self, reservoir_path: str):
        reservoir = torch.load(reservoir_path)
        self.train_H = reservoir(self.train_X)
        return compute_ridge_matrices(self.train_H, self.train_Y)
    
    @torch.no_grad
    def local_eval(self, readout_path = None):
        readout = torch.load(readout_path)['W']
        train_pred = F.linear(self.train_H, readout)
        return self.score_fn(self.train_Y, train_pred)

    def _build(self, i, exp_path, config):
        self.exp_path = exp_path
        self.res_client_path = os.path.join(exp_path, 'clients', f'{i}.pkl')
        self.epochs = config['EPOCHS']
        self.batch_size = config['BATCH_SIZE']
        self.lr = config['ETA']
