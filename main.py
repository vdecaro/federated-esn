import argparse
import pickle
import os

import ray
from ray import tune
from yaml import parse
from exp import run_exp


parser = argparse.ArgumentParser()
parser.add_argument('dataset', )

def get_config(name):
    if name == 'WESAD':
        TEST_USERS = [1, 4, 7]
        config = {
            'DATASET': 'WESAD',
            'TRAIN_USERS': [2, 5, 7, 8, 11, 13, 15, 16, 17],
            'VALIDATION_USERS': [4, 10, 14],
            'SEQ_LENGTH': tune.choice([100, 200, 300]),
            'INPUT_SIZE': 8,
            'N_CLASSES': 4,

            'HIDDEN_SIZE': tune.choice([250, 300]),
            'RHO': tune.uniform(0.5, 0.99),
            'LEAKAGE': tune.quniform(0.1, 1, 0.1),
            'INPUT_SCALING': tune.uniform(0.5, 1),
            'MU': 0,
            'SIGMA': tune.uniform(0.005, 0.3),
            'ETA': 1e-5,
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'BATCH_SIZE': 100,
            'EPOCHS': tune.randint(1, 15),
            'PATIENCE': 5,
            'NORMALIZE': tune.choice([True, False])
        }
    if name == 'HHAR':
        TEST_USERS = [3, 6]
        config = {
            'DATASET': 'HHAR',
            'TRAIN_USERS': [0, 1, 2, 4, 7],
            'VALIDATION_USERS': [5, 8],
            'SEQ_LENGTH': tune.choice([100, 200, 300]),
            'N_CLASSES': 6,
            'INPUT_SIZE': 6,

            'HIDDEN_SIZE': tune.choice([100, 500]),
            'RHO': tune.quniform(0.1, 1, 0.05),
            'LEAKAGE': tune.quniform(0.1, 1, 0.1),
            'INPUT_SCALING': tune.quniform(0.1, 1, 0.05),
            'MU': tune.quniform(-0.5, 0.5, 0.1),
            'SIGMA': tune.quniform(0, 0.5, 0.05),
            'ETA': 1e-4,
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'BATCH_SIZE': 100,
            'EPOCHS': tune.randint(1, 15),
            'PATIENCE': 5
        }
    return config, TEST_USERS

def main():
    args = parser.parse_args()
    dataset = args.dataset
    config, test_users = get_config(dataset)
    exp_dir = f"experiments/{config['DATASET']}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    ray.init()
    res = run_exp(config, exp_dir)
    with open(os.path.join(exp_dir, 'dump_res.pkl'), 'wb+') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()