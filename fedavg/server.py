import os
import ray
from ray import tune
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from data.seq_loader import seq_collate_fn

from .client import FedAvgClient
from .esn.reservoir import Reservoir
from .esn.readout import validate_readout

from .utils import empty_cache

class FedAvgServer(tune.Trainable):

    def setup(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Server running on {self.device}")
        self.n_clients = len(config['TRAIN_USERS'])
        self.clients = [FedAvgClient.options(num_cpus=1, num_gpus=config['GPU_SIZE']).remote(i) for i in range(self.n_clients)]
        self.reservoir, self.res_path = None, None
        self.readout, self.read_path = None, None
        self.mode = None

        self.eval_data = None
        self.loader = None
        self.l2 = None
        self.score_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()
        
        self._data_init(config)
        self._build(config)
    
    def step(self):
        with torch.no_grad():
            if self.mode == 'intrinsic_plasticity':
                print("Clients are starting the local update...")
                client_refs = ray.get([c.local_ip_update.remote(self.res_path) for c in self.clients])
                print("Clients completed the local update.")

                client_models = [torch.load(ref) for ref in client_refs]
                empty_cache()
                ip_a_c = torch.stack([m.ip_a.data.clone() for m in client_models], dim=0)
                ip_b_c = torch.stack([m.ip_b.data.clone() for m in client_models], dim=0)
                self.reservoir.ip_a.copy_(torch.mean(ip_a_c, dim=0))
                self.reservoir.ip_b.copy_(torch.mean(ip_b_c, dim=0))
            self.reservoir.to(self.device)
            torch.save(self.reservoir, self.res_path)
            
            print("Clients are starting the AB computation...")
            clients_ab = ray.get([c.compute_ab.remote(self.res_path) for c in self.clients])
            print("Clients completed the AB computation.")
            empty_cache()
            client_a, client_b = zip(*clients_ab)
            a = sum(client_a).to(self.device)
            b = sum(client_b).to(self.device)
            print("Server is starting the validation...")
            eval_data = []
            self.reservoir.eval()
            for loader in self.loaders:
                for eval_x, eval_y in loader:
                    res_appl = self.reservoir(eval_x.to(self.device)).reshape((-1, self.reservoir.hidden_size)).to('cpu')
                    y_reshaped = eval_y.reshape((-1, eval_y.size(-1)))
                    eval_data.append((res_appl, y_reshaped))
            empty_cache()

            self.readout, best_l2, best_score = validate_readout(a, b, eval_data, self.l2, self.score_fn)
            torch.save({'W': self.readout, 'l2': best_l2}, self.read_path)
            print("Server completed the validation.")
            empty_cache()

            print("Clients are starting the local evaluation...")
            client_eval = ray.get([c.local_eval.remote(self.read_path) for c in self.clients])
            acc_c, acc_n_samples = zip(*client_eval)
            train_acc = np.average(acc_c, weights=acc_n_samples)
            empty_cache()
            print("Clients completed the local evaluation.")
            
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
            sigma=config['SIGMA'],
            mode=config['MODE']
        )
        self.res_path = os.path.join(self.logdir, 'reservoir.pkl')
        torch.save(self.reservoir, self.res_path)

        self.l2 = config['L2']
        self.read_path = os.path.join(self.logdir, 'readout.pkl')
        self.mode = config['MODE']

        os.makedirs(os.path.join(self.logdir, 'clients'))
        _ = ray.get([c._build.remote(
            self.logdir,
            config
        )   
        for _, c in enumerate(self.clients)])
        
        self.client_refs = [
            os.path.join(self.logdir, 'clients', f'{i}.pkl') for i in range(self.n_clients)
        ]
        
        self.loaders = []
        for data in self.eval_data:
            data.seq_length = config['SEQ_LENGTH']
            self.loaders.append(DataLoader(data, batch_size=500, collate_fn=seq_collate_fn))
        
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

    def _data_init(self, config):
        if config['DATASET'] == 'WESAD':
            from data.wesad import WESADDataset
            data_constr = WESADDataset
        if config['DATASET'] == 'HHAR':
            from data.hhar import HHARDataset
            data_constr = HHARDataset
        
        self.eval_data = [data_constr(u) for u in config['VALIDATION_USERS']]
