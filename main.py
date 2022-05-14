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
parser.add_argument('--gpu_trial', '-g', type=int, default=1)
parser.add_argument('--test', '-t', action='store_true')

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


def main():
    args = parser.parse_args()
    dataset, perc, gt, mode, test = args.dataset, args.percentage, args.gpu_trial, args.mode, args.test
    config = get_config(dataset, perc, mode, test)
    exp_dir = f"experiments/{config['DATASET']}_{perc}_{mode}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    ray.init()
    res = run_exp(config, exp_dir, gt, test)

if __name__ == '__main__':
    main()