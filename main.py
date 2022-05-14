import argparse
import pickle
import os

import ray
from ray import tune
from exp import run_exp
from data.seq_loader import seq_collate_fn

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--mode', '-m', type=str)
parser.add_argument('--percentage', '-p', type=int, default=100)
parser.add_argument('--gpu_trial', '-g', type=int, default=1)

USERS = {
    'WESAD': {
        'TRAIN': {
            25: [0, 3, 5],
            50: [0, 3, 5, 6, 9],
            75: [0, 3, 5, 6, 9, 10, 12],
            100: [0, 3, 5, 6, 9, 10, 12, 13, 14]
        }, 
        'VALIDATION': [1, 8, 11],
        'TEST': [2, 4, 7]
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

def get_config(name, perc, mode, test):
    if not test:
        d_users = USERS[name]
        if name == 'WESAD':
            config = {
                'DATASET': 'WESAD',
                'TRAIN_USERS': d_users['TRAIN'][perc],
                'VALIDATION_USERS': d_users['VALIDATION'],
                'SEQ_LENGTH': tune.choice([150, 350, 700]),
                'INPUT_SIZE': 8,
                'N_CLASSES': 4,

                'HIDDEN_SIZE': tune.choice([200, 250, 300]),
                'RHO': tune.uniform(0.3, 0.99),
                'LEAKAGE': tune.choice([0.1, 0.3, 0.5]),
                'INPUT_SCALING': tune.uniform(0.5, 1),
                'MU': 0,
                'SIGMA': tune.uniform(0.005, 0.15),
                'ETA': 1e-2,
                'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
                'BATCH_SIZE': 100,
                'EPOCHS': tune.choice([1, 3, 5]),
                'PATIENCE': 5,
                'MODE': mode
            }
        if name == 'HHAR':
            config = {
                'DATASET': 'HHAR',
                'TRAIN_USERS': d_users['TRAIN'][perc],
                'VALIDATION_USERS': d_users['VALIDATION'],
                'SEQ_LENGTH': tune.choice([100, 150, 200, 400]),
                'N_CLASSES': 6,
                'INPUT_SIZE': 6,

                'HIDDEN_SIZE': tune.choice([100, 200, 300, 400, 500]),
                'RHO': tune.uniform(0.3, 0.99),
                'LEAKAGE': tune.choice([0.1, 0.3, 0.5]),
                'INPUT_SCALING': tune.uniform(0.5, 1),
                'MU': 0,
                'SIGMA': tune.uniform(0.005, 0.15),
                'ETA': 1e-2,
                'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
                'BATCH_SIZE': 50,
                'EPOCHS': tune.choice([1, 3, 5]),
                'PATIENCE': 5,
                'MODE': mode
            }
    else:
        tune_exp = tune.ExperimentAnalysis(f"experiments/{name}_{perc}_{mode}/{name}_ms", default_metric='eval_score', default_mode='max')
        config = tune_exp.get_best_config()
        config['MODE'] = mode
    return config


def test_fn(dataset, perc, mode):

    test_dir = f"experiments/{dataset}_{perc}_{mode}/{dataset}_test"
    if dataset == 'WESAD':
        from data.wesad import WESADDataset
        data_constr = WESADDataset
    if dataset == 'HHAR':
        from data.hhar import HHARDataset
        data_constr = HHARDataset

    test_exp = tune.ExperimentAnalysis(test_dir, default_metric='eval_score', default_mode='max')
    config = test_exp.get_best_config()
    test_data = [data_constr(u) for u in USERS[dataset]['TEST']]
    for d in test_data:
        d.seq_length = config['SEQ_LENGTH']
    test_loaders = [DataLoader(
            d, 
            batch_size=500,
            collate_fn=seq_collate_fn
    ) for d in test_data]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    acc_fn = lambda Y, Y_pred: (torch.sum(Y == Y_pred)/Y.size(0)).item()

    acc = {}
    for i, trial in enumerate(test_exp.trials):
        chk = test_exp.get_best_checkpoint(trial)
        reservoir = torch.load(os.path.join(chk.local_path, 'reservoir.pkl')).to(device).eval()
        readout = torch.load(os.path.join(chk.local_path, 'readout.pkl')).to(device)
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


def main():
    args = parser.parse_args()
    dataset, perc, gt, mode = args.dataset, args.percentage, args.gpu_trial, args.mode
    test = False
    config = get_config(dataset, perc, mode, test)
    if test:
        config['SEQ_LENGTH'] = 400 if dataset == 'HHAR' else 700
    exp_dir = f"experiments/{config['DATASET']}_{perc}_{mode}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    ray.init()
    
    res = run_exp(config, exp_dir, gt, test)

    test = True
    config = get_config(dataset, perc, mode, test)
    res = run_exp(config, exp_dir, gt, test)
    test_fn(dataset, perc, mode)

if __name__ == '__main__':
    main()