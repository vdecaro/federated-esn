import argparse
import pickle
import os

import ray
from ray import tune
from yaml import parse
from exp import run_exp
from data.seq_loader import seq_collate_fn

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--mode', '-m', type=str)
parser.add_argument('--percentage', '-p', type=int, default=100)

USERS = {
    'WESAD': {
        'TRAIN': {
            25: [2, 5, 7],
            50: [2, 5, 7, 8, 11],
            75: [2, 5, 7, 8, 11, 13, 15],
            100: [2, 5, 7, 8, 11, 13, 15, 16, 17]
        }, 
        'VALIDATION': [3, 10, 14],
        'TEST': [4, 6, 9]
    },
    'HHAR':{
        'TRAIN': {
            25: [0, 1],
            50: [0, 1, 2],
            75: [0, 1, 2, 4],
            100: [0, 1, 2, 4, 7]
        },
        'VALIDATION': [5, 8],
        'TEST': [3, 6]
    }
}

def main():
    args = parser.parse_args()
    dataset, perc, mode = args.dataset, args.percentage, args.mode

    test_dir = f"experiments/{dataset}_{perc}_{mode}/{dataset}_test"
    if dataset == 'WESAD':
        from data.wesad import WESADDataset
        data_constr = WESADDataset
    if dataset == 'HHAR':
        from data.hhar import HHARDataset
        data_constr = HHARDataset
    test_data = [data_constr(u) for u in USERS[dataset]['TEST']]
    test_loaders = [DataLoader(
            d, 
            batch_size=500,
            collate_fn=seq_collate_fn
    ) for d in test_data]

    test_exp = tune.ExperimentAnalysis(test_dir, default_metric='eval_score', default_mode='max')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    acc_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()

    acc = {}
    for i, trial in enumerate(test_exp.trials):
        chk = test_exp.get_best_checkpoint(trial)
        reservoir = torch.load(os.path.join(chk.local_path, 'reservoir.pkl')).to(device).eval()
        readout = torch.load(os.path.join(chk.local_path, 'readout.pkl'))['W'].to(device).eval()
        trial_acc, trial_n_samples = 0, 0
        with torch.no_grad():
            for loader in test_loaders:
                for x, y in loader:
                    h = reservoir(x.to('cuda' if torch.cuda.is_available() else 'cpu')).reshape((-1, reservoir.hidden_size))
                    Y_pred = torch.argmax(F.linear(h, readout), -1).flatten().to('cpu')
                    Y_true = torch.argmax(y, dim=-1).flatten()
                    curr_acc = acc_fn(Y_true, Y_pred)
                    curr_n_samples = Y_true.size(0)
                    trial_acc += curr_acc * curr_n_samples
                    trial_n_samples += curr_n_samples
            acc[f'trial_{i}'] = trial_acc / trial_n_samples
    with open(f"experiments/{dataset}_{perc}_{mode}/test_res.pkl", 'wb+') as f:
        pickle.dump(acc, f)
    print(acc, "saved.")


if __name__ == '__main__':
    main()