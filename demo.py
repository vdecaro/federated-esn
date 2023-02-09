import ray, argparse, torch
from exp.config import get_fed_args
from torch_esn.model.reservoir import Reservoir

from typing import Dict


def vanilla_demo(config: Dict):
    from torch_esn.wrapper.vanilla import VanillaESNWrapper

    trainer = VanillaESNWrapper(
        config["dataset"], config["train_users"], config["batch_size"]
    )
    evaluator = VanillaESNWrapper(
        config["dataset"], config["eval_users"], config["batch_size"]
    )
    reservoir = Reservoir(**config["reservoir"])
    if config["template"] == "ridge":
        readout, *_ = trainer.ridge_step(reservoir, config["l2"])

    elif config["template"] == "ip":
        mu, sigma, eta = (config["mu"], config["sigma"], config["eta"])
        print("Training IP")
        reservoir = trainer.ip_step(reservoir, mu, sigma, eta, 10)
        # torch.save(reservoir, "reservoir.pkl")
        print("Training ridge")
        readout, *_ = trainer.ridge_step(reservoir)
        print(evaluator.test_likelihood(reservoir, mu, sigma))

    results = evaluator.test_accuracy(readout, reservoir)

    print(results)


def continual_demo(config: Dict):
    from torch_esn.wrapper.continual import ContinualESNWrapper

    trainer = ContinualESNWrapper(
        config["dataset"],
        config["train_users"],
        config["batch_size"],
        config["strategy"],
    )
    evaluator = ContinualESNWrapper(
        config["dataset"],
        config["eval_users"],
        config["batch_size"],
        config["strategy"],
    )
    reservoir = Reservoir(**config["reservoir"])
    mu, eta, sigma = (config["mu"], config["sigma"], config["eta"])
    reservoir = trainer.ip_step(1, reservoir, mu, sigma, eta)
    readout, *_ = trainer.ridge_step(1, reservoir)
    results = [evaluator.test_accuracy(c, readout, reservoir) for c in range(4)]
    print(results)


def vanilla_federated_demo(config: Dict):
    from fedesn.vanilla import VanillaESNFederation

    federation = VanillaESNFederation(**get_fed_args(config))
    if config["template"] == "fedip":
        mu, eta, sigma, epochs = (
            config["mu"],
            config["sigma"],
            config["eta"],
            config["epochs"],
        )
        _ = federation.ip_ridge_train(config["reservoir"], mu, sigma, eta, epochs)
    elif config["template"] == "incfed":
        _ = federation.ridge_train(config["reservoir"], config["l2"])
    model = federation.pull_version()["model"]
    results = federation.test_accuracy("eval", model)
    print(results)
    federation.stop()


def continual_federated_demo(config: Dict):
    from fedesn.continual import ContinualESNFederation

    federation = ContinualESNFederation(**get_fed_args(config))
    reservoir = config["reservoir"]
    mu, sigma, eta, epochs = (
        config["mu"],
        config["sigma"],
        config["eta"],
        config["epochs"],
    )
    _ = federation.ip_train(1, reservoir, mu, sigma, eta, epochs)
    reservoir = federation.pull_version()["model"]["reservoir"]
    federation.stop()
    _ = federation.ridge_train(1, reservoir)
    model = federation.pull_version()["model"]
    results = [federation.test_accuracy("eval", c, model) for c in range(4)]
    print(results)


def get_config(template: str):
    from torch_esn.data.datasets.hhar import HHARDataset

    config = {
        "template": template,
        "data_config": {
            "dataset": "HHAR",
            "batch_size": 100,
        },
        "train_users": HHARDataset.USERS["train"][25],
        "eval_users": HHARDataset.USERS["eval"],
        "reservoir": {
            "input_size": 6,
            "activation": "tanh",
            "hidden_size": 400,
            "rho": 0.8,
            "leakage": 0.1,
            "input_scaling": 1,
        },
        "l2": [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    }
    if "ip" in template:
        config["reservoir"]["net_gain_and_bias"] = True
        config.update(
            {
                "mu": 0,
                "sigma": 0.5,
                "eta": 1e-2,
                "epochs": 1,
            }
        )
    if "continual" in template:
        template, strategy = template.split("_")[:-1], template.split("_")[-1]
        config["template"] = template
        config["strategy"] = strategy

    return config


parser = argparse.ArgumentParser()
parser.add_argument("method", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    method = args.method
    config = get_config(method)
    if method in ["ridge", "ip"]:
        vanilla_demo(config)
    elif method in ["fedip", "incfed"]:
        ray.init()  # address="ray://localhost:10001")
        vanilla_federated_demo(config)
    elif "continual_ip" in method:
        continual_demo(config)
    elif "continual_fedip" in method:
        ray.init()
        continual_federated_demo(config)
