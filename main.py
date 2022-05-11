import argparse
import pickle
import os

import ray
from ray import tune
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
        conf = {
            'DATASET': 'WESAD',
            'TRAIN_USERS': TRAIN_USERS[perc],
            'VALIDATION_USERS': [4, 10, 14],
            'SEQ_LENGTH': tune.choice([100, 200, 300]),
            'INPUT_SIZE': 8,
            'N_CLASSES': 4,

            'HIDDEN_SIZE': tune.choice([100, 250]),
            'RHO': tune.quniform(0.1, 0.99, 0.05),
            'LEAKAGE': tune.quniform(0.1, 1.01, 0.1),
            'INPUT_SCALING': tune.quniform(0.1, 1.01, 0.1),
            'MU': tune.quniform(-0.5, 0.5, 0.1),
            'SIGMA': tune.quniform(0.1, 0.5, 0.05),
            'ETA': tune.qloguniform(1e-4, 1e-1, 5e-5),
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'BATCH_SIZE': 100,
            'PATIENCE': 10
        }
    if name == 'HAR':
        TEST_USERS = [3, 6]
        TRAIN_USERS = {
            25: [0, 1],
            50: [0, 1, 2],
            75: [0, 1, 2, 4],
            100: [0, 1, 2, 4, 7]
        }
        conf = {
            'DATASET': 'HHAR',
            'TRAIN_USERS': TRAIN_USERS[perc],
            'VALIDATION_USERS': [5, 8],
            'SEQ_LENGTH': tune.choice([100, 200, 300]),
            'N_CLASSES': 6,
            'INPUT_SIZE': 6,

            'HIDDEN_SIZE': tune.choice([100, 500]),
            'RHO': tune.quniform(0.1, 0.99, 0.05),
            'LEAKAGE': tune.quniform(0.1, 1.01, 0.1),
            'INPUT_SCALING': tune.quniform(0.1, 1.01, 0.1),
            'MU': tune.quniform(-0.5, 0.5, 0.1),
            'SIGMA': tune.quniform(0, 0.5, 0.5),
            'ETA': tune.qloguniform(1e-4, 1e-1, 5e-5),
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'BATCH_SIZE': 100,
            'PATIENCE': 10
        }
    return conf, TEST_USERS

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