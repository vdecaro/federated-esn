import ray
from ray import tune

def get_config(name):
    if name == 'WESAD':
        TEST_USERS = [1, 4, 7]
        return {
            'DATASET': 'WESAD',
            'TRAIN_USERS': [2, 5, 7, 8, 11, 13, 15, 16, 17],
            'VALIDATION_USERS': [4, 10, 14],
            'N_CLASSES': 4,
            'INPUT_SIZE': 8,

            'HIDDEN_SIZE': tune.choice([100, 250]),
            'RHO': tune.quniform(0.1, 0.99, 0.05),
            'LEAKAGE': tune.quniform(0.1, 1.01, 0.1),
            'INPUT_SCALING': tune.quniform(0.1, 1.01, 0.1),
            'MU': tune.quniform(-0.5, 0.5, 0.1),
            'SIGMA': tune.quniform(0, 0.5, 0.5),
            'ETA': tune.qloguniform(1e-4, 1e-1, 5e-5),
            'BATCH_SIZE': 100,
        }
    if name == 'HAR':
        return {
            'DATASET': 'HAR',
            'TRAIN_USERS': [2, 5, 7, 8, 11, 13, 15, 16, 17],
            'VALIDATION_USERS': [4, 10, 14],
            'N_CLASSES': 4,
            'INPUT_SIZE': 8,

            'HIDDEN_SIZE': tune.choice([100, 250]),
            'RHO': tune.quniform(0.1, 0.99, 0.05),
            'LEAKAGE': tune.quniform(0.1, 1.01, 0.1),
            'INPUT_SCALING': tune.quniform(0.1, 1.01, 0.1),
            'MU': tune.quniform(-0.5, 0.5, 0.1),
            'SIGMA': tune.quniform(0, 0.5, 0.5),
            'ETA': tune.qloguniform(1e-4, 1e-1, 5e-5),
            'BATCH_SIZE': 100,
        }