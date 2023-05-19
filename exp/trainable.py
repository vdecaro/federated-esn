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
        self.federation.ridge_train(model["reservoir"], cfg["l2"], cfg["perc_rec"])
        model = self.federation.pull_version()["model"]
        self.federation.stop()

        to_return = {}
        for k in ["full", "rand", "imp"]:
            eval_score = self.federation.test_accuracy(
                "eval",
                {"reservoir": model["reservoir"], "readout": model[k + "_readout"]},
            )
            train_score = self.federation.test_accuracy(
                "train",
                {"reservoir": model["reservoir"], "readout": model[k + "_readout"]},
            )
            test_score = self.federation.test_accuracy(
                "test",
                {"reservoir": model["reservoir"], "readout": model[k + "_readout"]},
            )
            to_return[k + "_train_score"] = train_score
            to_return[k + "_test_score"] = test_score
            to_return[k + "_eval_score"] = eval_score

        to_return["perc_chosen"] = model["perc_chosen"]

        self.model = model
        return to_return

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        torch.save(self.model, os.path.join(checkpoint_dir, "model.pt"))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: Union[Dict, str]):
        self.model = torch.load(os.path.join(checkpoint_dir, "model.pt"))

    def reset_config(self, new_config):
        return True
