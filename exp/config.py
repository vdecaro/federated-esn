from ray import tune

from typing import Dict, Literal

EXP_PATH = ""


def get_exp_dir(method: str, dataset: str, perc: int):
    if method in ["ip", "ridge"] or "continual_ip" in method:
        exp_dir = f"{EXP_PATH}/centralized/{method}/{dataset}/{perc}"
    elif method in ["fedip", "incfed"] or "continual_fedip" in method:
        exp_dir = f"{EXP_PATH}/federated/{method}/{dataset}/{perc}"
    else:
        raise ValueError("Invalid experiment type.")

    return exp_dir


def get_hparams_config(method: str, dataset: str, perc: int):
    if "continual" not in method:
        if dataset == "WESAD":
            config = _wesad_experiment(perc)
        elif dataset == "HHAR":
            config = _hhar_experiment(perc)
        else:
            raise ValueError("Invalid experiment type.")

        return _load_method(config, method)
    else:
        return _get_continual_config(method, dataset, perc)


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
    if "continual" in config["template"]:
        fed_args["strategy"] = config["strategy"]

    return fed_args


def get_analysis(phase: Literal["model_selection", "retraining", "test"], exp_dir: str):
    return tune.ExperimentAnalysis(
        f"{exp_dir}/{phase}", default_metric="eval_acc", default_mode="max"
    )


def _load_method(config: Dict, method: str):
    config["template"] = method
    if "ip" in method:
        config["reservoir"]["net_gain_and_bias"] = True
        config["ip_args"] = {
            "mu": 0,
            "eta": 1e-2,
        }
        config["ip_args"]["sigma"] = tune.quniform(0.05, 1.05, 0.05)
        config["rounds"] = tune.choice([10, 12, 14, 16, 18, 20])
    if method == "fedip":
        config["ip_args"]["epochs"] = tune.choice([3, 5, 10])

    return config


def _get_continual_config(
    method: str,
    dataset: str,
    perc: int,
):
    template = "_".join(method.split("_")[:-1])
    strategy = method.split("_")[-1]
    if template == "continual_ip":
        ip_exp_dir = get_exp_dir("ip", dataset, perc)
    if template == "continual_fedip":
        ip_exp_dir = get_exp_dir("fedip", dataset, perc)

    config = get_analysis("model_selection", ip_exp_dir).get_best_config(
        "eval_acc", "max"
    )
    config["template"] = template
    config["strategy"] = strategy
    if dataset == "WESAD":
        config["n_contexts"] = 5
    elif dataset == "HHAR":
        config["n_contexts"] = 4
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    if strategy != "joint":
        if template == "continual_ip":
            rounds = config["rounds"] // 2
            config["rounds"] = tune.grid_search(list(range(rounds - 2, rounds + 3)))
        elif template == "continual_fedip":
            rounds = config["rounds"] // config["n_contexts"]
            config["rounds"] = tune.grid_search(
                list(range(rounds, rounds * (config["n_contexts"] - 1)))
            )

    config["l2"] = 1e-5

    return config


def _wesad_experiment(perc: int):
    from torch_esn.data.datasets.wesad import WESADDataset

    config = {
        "data_config": {
            "dataset": "WESAD",
            "batch_size": 100,
        },
        "train_users": WESADDataset.USERS["train"][perc],
        "eval_users": WESADDataset.USERS["eval"],
        "test_users": WESADDataset.USERS["test"],
        "reservoir": {
            "input_size": 8,
            "activation": "tanh",
            "hidden_size": tune.choice([200, 300, 400]),
            "rho": tune.quniform(0.3, 0.95, 0.05),
            "leakage": tune.quniform(0.1, 0.8, 0.05),
            "input_scaling": tune.quniform(0.5, 1, 0.05),
        },
        "l2": [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    }

    return config


def _hhar_experiment(perc: int):
    from torch_esn.data.datasets.hhar import HHARDataset

    config = {
        "data_config": {
            "dataset": "HHAR",
            "batch_size": 100,
        },
        "train_users": HHARDataset.USERS["train"][perc],
        "eval_users": HHARDataset.USERS["eval"],
        "test_users": HHARDataset.USERS["test"],
        "reservoir": {
            "input_size": 6,
            "activation": "tanh",
            "hidden_size": tune.choice([100, 200, 300, 400, 500]),
            "rho": tune.quniform(0.3, 0.95, 0.05),
            "leakage": tune.quniform(0.1, 0.8, 0.05),
            "input_scaling": tune.uniform(0.5, 1),
        },
        "l2": [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    }

    return config
