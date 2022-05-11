import argparse
import pickle
import os

import ray
from ray import tune
from yaml import parse
from exp import run_exp


parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--percentage', '-p', type=int, default=100)
parser.add_argument('--gpu_trial', '-g', type=int, default=1)


def get_config(name, perc):
    if name == 'WESAD':
        TEST_USERS = [1, 4, 7]
        TRAIN_USERS = {
            25: [2, 5, 7],
            50: [2, 5, 7, 8, 11],
            75: [2, 5, 7, 8, 11, 13, 15],
            100: [2, 5, 7, 8, 11, 13, 15, 16, 17]
        }
        config = {
            'DATASET': 'WESAD',
            'TRAIN_USERS': TRAIN_USERS[perc],
            'VALIDATION_USERS': [4, 10, 14],
            'SEQ_LENGTH': tune.choice([150, 350, 700]),
            'INPUT_SIZE': 8,
            'N_CLASSES': 4,

            'HIDDEN_SIZE': tune.choice([200, 250, 300]),
            'RHO': tune.uniform(0.5, 0.99),
            'LEAKAGE': tune.choice([0, 0.1, 0.5, 0.8, 0.9, 1]),
            'INPUT_SCALING': tune.uniform(0.5, 1),
            'MU': 0,
            'SIGMA': tune.uniform(0.005, 0.15),
            'ETA': 1e-2,
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'BATCH_SIZE': 50,
            'EPOCHS': tune.randint(1, 15),
            'PATIENCE': 5
        }
    if name == 'HHAR':
        TEST_USERS = [3, 6]
        TRAIN_USERS = {
            25: [0, 1],
            50: [0, 1, 2],
            75: [0, 1, 2, 4],
            100: [0, 1, 2, 4, 7]
        }
        config = {
            'DATASET': 'HHAR',
            'TRAIN_USERS': TRAIN_USERS[perc],
            'VALIDATION_USERS': [5, 8],
            'SEQ_LENGTH': tune.choice([100, 150, 200, 400]),
            'N_CLASSES': 6,
            'INPUT_SIZE': 6,

            'HIDDEN_SIZE': tune.choice([100, 200, 300, 400, 500]),
            'RHO': tune.uniform(0.5, 0.99),
            'LEAKAGE': tune.choice([0, 0.1, 0.5, 0.8, 0.9, 1]),
            'INPUT_SCALING': tune.uniform(0.5, 1),
            'MU': 0,
            'SIGMA': tune.uniform(0.005, 0.15),
            'ETA': 1e-2,
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'BATCH_SIZE': 50,
            'EPOCHS': tune.choice([1, 3, 5, 10, 15]),
            'PATIENCE': 5
        }
    return config, TEST_USERS

def main():
    dataset, perc, gt = parser.dataset, parser.percentage, parser.gpu_trial
    config = get_config(dataset, perc)
    exp_dir = f"experiments/{config['DATASET']}_{perc}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    ray.init()
    res = run_exp(config, exp_dir, gt)
    with open(os.path.join(exp_dir, 'dump_res.pkl'), 'wb+') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()