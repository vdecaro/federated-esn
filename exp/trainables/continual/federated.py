import copy, torch, os, json
from ray import tune
from exp.config import get_fed_args

from torch_esn.model.reservoir import Reservoir
from fedesn.continual import ContinualESNFederation

from typing import Dict, Optional, Union


class ContinualFedESNTrainable(tune.Trainable):
    def setup(self, config: Dict):
        self.federation = ContinualESNFederation(
            federation_id=self.trial_id, is_tune=True, **get_fed_args(config)
        )
        self.model = [None for _ in range(config["n_contexts"])]
        self.metrics = [None for _ in range(config["n_contexts"])]

    def step(self):
        config = self.get_config()
        test_on = "eval" if config["phase"] == "model_selection" else "test"
        reservoir = Reservoir(**config["reservoir"])
        for i in range(config["n_contexts"]):
            print("Starting training for context", i)
            if config["strategy"] == "joint":
                reservoir.load_state_dict(
                    {
                        "net_a": torch.ones_like(reservoir.net_a),
                        "net_b": torch.zeros_like(reservoir.net_b),
                    },
                    strict=False,
                )

            if config["strategy"] == "joint":
                best_model, best_acc, patience = None, 0, 0
                self.federation.ip_ridge_train(
                    i, reservoir, l2=config["l2"], **config["ip_args"]
                )
                while patience < 3:
                    model = self.federation.pull_version()["model"]
                    acc = self.federation.test_accuracy("eval", -i, model)
                    print(f"Accuracy on context {i}:", acc)
                    if acc > best_acc:
                        best_model, best_acc = model, acc
                        patience = 0
                    else:
                        patience += 1
                self.federation.stop()
                model = best_model
            else:
                self.federation.ip_train(i, reservoir, **config["ip_args"])
                for _ in range(config["rounds"]):
                    reservoir = self.federation.pull_version()["model"]["reservoir"]
                self.federation.stop()

                self.federation.ridge_train(i, reservoir, config["l2"])
                model = self.federation.pull_version()["model"]
                self.federation.stop()

            self.model[i] = copy.deepcopy(model)
            self.metrics[i] = {
                "exp": [
                    self.federation.test_accuracy(test_on, c, model)
                    for c in range(config["n_contexts"])
                ],
                "stream": self.federation.test_accuracy(test_on, -i, model),
            }
        results = self.metrics[-1]["stream"]
        return {"eval_acc": results, "eval_score": results}

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        torch.save(self.model, os.path.join(checkpoint_dir, "model.pkl"))
        json.dump(
            self.metrics,
            open(os.path.join(checkpoint_dir, "metrics.json"), "w"),
            indent=4,
        )
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: Union[Dict, str]):
        self.model = torch.load(os.path.join(checkpoint_dir, "model.pkl"))
        self.metrics = json.load(
            open(os.path.join(checkpoint_dir, "metrics.json"), "r")
        )

    def reset_config(self, new_config):
        print("Trial complete. Resetting to next configuration...")
        self.model = [None for _ in range(new_config["n_contexts"])]
        self.metrics = [None for _ in range(new_config["n_contexts"])]
        print("Federation reset to new config.")
        return True
