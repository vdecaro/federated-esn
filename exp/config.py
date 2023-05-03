import os, json
from ray import tune
from typing import Dict

EXP_PATH = "/raid/decaro/experiments/esann2023"


def get_exp_dir(dataset: str):
    return os.path.join(EXP_PATH, dataset)


def get_hparams_config(
    dataset: str,
    perc: int,
):
    if dataset == "WESAD":
        from torch_esn.data.datasets.wesad import WESADDataset

        data_class = WESADDataset
    elif dataset == "HHAR":
        from torch_esn.data.datasets.hhar import HHARDataset

        data_class = HHARDataset

    config = {
        "data_config": {
            "dataset": dataset,
            "batch_size": 100,
        },
        "train_users": data_class.USERS["train"][perc],
        "eval_users": data_class.USERS["eval"],
        "test_users": data_class.USERS["test"],
        "reservoir": json.load(
            open(os.path.join(EXP_PATH, "reservoir_params", dataset, f"{perc}.json"))
        )["reservoir"],
        "l2": 0.00001,
        "perc_rec": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    }
    return config


def get_fed_args(config: Dict):
    n_train = len(config["train_users"])
    n_eval = len(config["eval_users"])
    n_test = len(config["test_users"])
    roles = (
        ["train" for _ in range(n_train)]
        + ["eval" for _ in range(n_eval)]
        + ["test" for _ in range(n_test)]
    )
    fed_args = {
        "n_clients_or_ids": config["train_users"]
        + config["eval_users"]
        + config["test_users"],
        "roles": roles,
    }
    fed_args.update(config["data_config"])

    return fed_args
