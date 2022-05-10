import os

import ray
from ray import tune
import torch
import torch.nn.functional as F
import numpy as np

from .client import FedAvgClient
from .esn.reservoir import Reservoir
from .esn.readout import validate_readout

class FedAvgServer(tune.Trainable):

    def setup(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clients = [FedAvgClient.remote(i) for i in range(len(config['TRAIN_USERS']))]
        self.reservoir, self.res_path = None, None
        self.readout, self.read_path = None, None

        self.eval_data = None
        self.l2_values = None
        self.score_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()
        
        self._data_init(config)
        self._build(config)
    
    @torch.no_grad
    def step(self):
        client_refs = ray.get([c.local_update.remote(self.res_path) for c in self.clients])

        client_models = [torch.load(ref) for ref in client_refs]
        ip_a_c = torch.stack([m.ip_a.data.clone() for m in client_models], dim=0)
        ip_b_c = torch.stack([m.ip_b.data.clone() for m in client_models], dim=0)
        self.reservoir.ip_a.copy_(torch.mean(ip_a_c, dim=0))
        self.reservoir.ip_b.copy_(torch.mean(ip_b_c, dim=0))
        self.reservoir.to(self.device)
        torch.save(self.reservoir, self.res_path)
        
        clients_ab = ray.get([c.compute_ab.remote(self.res_path) for c in self.clients])

        client_a, client_b = zip(*clients_ab)
        a = torch.stack(client_a, dim=0).sum(0).to(self.device)
        b = torch.stack(client_b, dim=0).sum(0).to(self.device)
        eval_H = torch.cat([self.reservoir(eval_d.X) for eval_d in self.eval_data], 0).to(self.device), 
        eval_Y = torch.cat([eval_d.Y for eval_d in self.eval_data], 0).to(self.device)
        best_W, best_l2, best_score = validate_readout(a, b, eval_H, eval_Y, self.l2_values, self.score_fn)
        torch.save({'W': best_W, 'l2': best_l2}, self.read_path)

        client_eval = ray.get([c.local_eval.remote(self.read_path) for c in self.clients])
        acc_c, acc_n_samples = zip(*client_eval)
        train_acc = np.average(acc_c, weights=acc_n_samples)
        
        return {
            "train_score":  train_acc,
            "eval_score": best_score,
        }


    def _build(self, config):
        self.reservoir = Reservoir(
            input_size = config['INPUT_SIZE'],
            hidden_size=config['HIDDEN_SIZE'],
            activation='tanh',
            leakage=config['LEAKAGE'],
            input_scaling=config['INPUT_SCALING'],
            rho=config['RHO'],
            mu=config['MU'],
            sigma=config['SIGMA']
        )
        self.res_path = os.path.join(self.logdir, 'reservoir.pkl')
        torch.save(self.reservoir, self.res_path)

        self.l2 = config['L2']
        self.readout = torch.nn.Linear(
            in_features=self.reservoir.hidden_size,
            out_features=self.config['N_CLASSES'],
            bias=False
        )
        self.read_path = os.path.join(self.logdir, 'readout.pkl')

        _ = ray.get([c._build.remote(
            self.logdir,
            config
        )   
        for i, c in enumerate(self.clients)])
        self.client_refs = [
            os.path.join(self.logdir, 'clients', f'{i}.pkl') for i in range(config['N_CLIENTS'])
        ]
        
        for data in self.eval_data:
            data.seq_length = config['SEQ_LENGTH']
        
        return True
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        torch.save(self.reservoir, os.path.join(tmp_checkpoint_dir, 'reservoir.pkl'))
        torch.save(self.readout, os.path.join(tmp_checkpoint_dir, 'readout.pkl'))
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, checkpoint):
        self.reservoir = torch.load(os.path.join(checkpoint, 'reservoir.pkl'))
        self.readout = torch.load(os.path.join(checkpoint, 'readout.pkl'))

    def reset_config(self, new_config):
        return self._build(new_config)

    def data_init(self, config):
        if config['DATASET'] == 'WESAD':
            from data.wesad import WESADDataset
            data_constr = WESADDataset
        if config['DATASET'] == 'HHAR':
            from data.hhar import HHARDataset
            data_constr = HHARDataset
        
        self.eval_data = [data_constr(u) for u in config['VALIDATION_USERS']]
