import torch, copy, os, json
from ray import tune

from torch_esn.model.reservoir import Reservoir
from torch_esn.wrapper.continual import ContinualESNWrapper

from typing import Dict, Optional, Union


class ContinualESNTrainable(tune.Trainable):
    def setup(self, config: Dict):
        self.method = config["template"]
        data_config = config["data_config"]
        wrapper_fn = lambda users: ContinualESNWrapper(
            dataset=data_config["dataset"],
            users=users,
            batch_size=data_config["batch_size"],
            strategy=config["strategy"],
        )
        self.trainer = wrapper_fn(config["train_users"])
        self.evaluator = wrapper_fn(config["eval_users"])
        self.tester = wrapper_fn(
            config["eval_users"]
            if config["phase"] == "model_selection"
            else config["test_users"]
        )

        self.model = [None for _ in range(config["n_contexts"])]
        self.metrics = [None for _ in range(config["n_contexts"])]

    def step(self):
        config = self.get_config()
        n_ctx = config["n_contexts"]
        reservoir = Reservoir(**config["reservoir"])
        for i in range(n_ctx):
            print("Training on context", i)
            if config["strategy"] == "joint":
                reservoir.load_state_dict(
                    {
                        "net_a": torch.ones_like(reservoir.net_a),
                        "net_b": torch.zeros_like(reservoir.net_b),
                    },
                    strict=False,
                )
                best_reservoir, best_readout, best_acc, patience = None, None, 0, 0
                while patience < 5:
                    reservoir = self.trainer.ip_step(i, reservoir, **config["ip_args"])
                    readout, *_ = self.trainer.ridge_step(i, reservoir, l2=config["l2"])
                    acc = self.evaluator.test_accuracy(-i, readout, reservoir)
                    print(f"Accuracy on context {i}:", acc)
                    if acc > best_acc:
                        best_reservoir, best_readout, best_acc = (
                            copy.deepcopy(reservoir),
                            copy.deepcopy(readout),
                            acc,
                        )
                        patience = 0
                    else:
                        patience += 1
            else:
                reservoir = self.trainer.ip_step(
                    i, reservoir, epochs=config["rounds"], **config["ip_args"]
                )
                best_reservoir, best_readout, *_ = self.trainer.ridge_step(
                    i, reservoir, l2=config["l2"]
                )
            readout, reservoir = best_readout, best_reservoir
            self.model[i] = {
                "reservoir": copy.deepcopy(reservoir),
                "readout": copy.deepcopy(readout),
            }
            self.metrics[i] = {
                "exp": [
                    self.tester.test_accuracy(c, readout, reservoir)
                    for c in range(n_ctx)
                ],
                "stream": self.tester.test_accuracy(-i, readout, reservoir),
            }
        results = sum([m["stream"] for m in self.metrics]) / n_ctx
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
        print("ContinualESN reset to new config.")
        return True
