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


if __name__ == '__main__':
    main()