import os, torch
import numpy as np
from ray import tune

from torch_esn.model.reservoir import Reservoir
from fedesn.vanilla import VanillaESNFederation
from exp.config import get_fed_args

from typing import Dict, Optional, Union


class VanillaFedESNTrainable(tune.Trainable):
    def setup(self, config: Dict):
        self.federation = VanillaESNFederation(
            federation_id=self.trial_id, is_tune=True, **get_fed_args(config)
        )
        self.model = None

    def step(self):
        cfg = self.get_config()
        model = {"reservoir": Reservoir(**cfg["reservoir"])}
        to_return = {}
        if cfg["template"] == "fedip":
            mu, sigma = cfg["ip_args"]["mu"], cfg["ip_args"]["sigma"]
            self.federation.ip_train(model["reservoir"], **cfg["ip_args"])
            for _ in range(cfg["rounds"]):
                model = self.federation.pull_version()["model"]
            self.federation.stop()
            likelihood = self.federation.test_likelihood("eval", model, mu, sigma)
            to_return["eval_score"] = likelihood

        if cfg["template"] in ["incfed", "fedip"]:
            self.federation.ridge_train(model["reservoir"], cfg["l2"])
            model = self.federation.pull_version()["model"]
            self.federation.stop()
            results = self.federation.test_accuracy("eval", model)
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
