import os, torch
import numpy as np
from ray import tune

from torch_esn.model.reservoir import Reservoir

from typing import Dict, Optional, Union


class VanillaESNTrainable(tune.Trainable):
    def setup(self, config: Dict):
        from torch_esn.wrapper.vanilla import VanillaESNWrapper

        self.method = config["template"]
        data_config = config["data_config"]
        wrapper_fn = lambda users: VanillaESNWrapper(
            dataset=data_config["dataset"],
            users=users,
            batch_size=data_config["batch_size"],
        )
        self.trainer = wrapper_fn(config["train_users"])
        self.evaluator = wrapper_fn(config["eval_users"])
        self.model = None

    def step(self):
        cfg = self.get_config()
        model = {"reservoir": Reservoir(**cfg["reservoir"]), "readout": None}
        to_return = {}
        if cfg["template"] == "ip":
            model["reservoir"] = self.trainer.ip_step(
                model["reservoir"], epochs=cfg["rounds"], **cfg["ip_args"]
            )
            likelihood = self.evaluator.test_likelihood(
                model["reservoir"], cfg["ip_args"]["mu"], cfg["ip_args"]["sigma"]
            )
            to_return["eval_score"] = likelihood

        if cfg["template"] in ["ridge", "ip"]:
            model["readout"], *_ = self.trainer.ridge_step(
                model["reservoir"], cfg["l2"]
            )
            results = self.evaluator.test_accuracy(model["readout"], model["reservoir"])
            model["readout"] = model["readout"][np.argmax(results)]
            accuracy = np.max(results)
            to_return["eval_acc"] = accuracy

        self.model = model
        return to_return

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        torch.save(self.model, os.path.join(checkpoint_dir, "model.pkl"))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: Union[Dict, str]):
        self.model = torch.load(os.path.join(checkpoint_dir, "model.pkl"))

    def reset_config(self, new_config):
        return True
