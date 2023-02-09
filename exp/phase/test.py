from typing import Optional
import torch, os, json
from ray import tune


def run(analysis: tune.ExperimentAnalysis, exp_dir: str, device: Optional[str] = None):
    results = []
    if "continual" not in exp_dir:
        from fedesn.vanilla import VanillaESNFederation

        config = analysis.get_best_config(metric="eval_score", mode="max")
        data_config = config["data_config"]
        tester = VanillaESNFederation(
            dataset=data_config["dataset"],
            batch_size=data_config["batch_size"],
            n_clients_or_ids=config["test_users"],
            roles=["test" for _ in config["test_users"]],
        )
        for trial in analysis.trials:
            model_path = analysis.get_best_checkpoint(
                trial, metric="eval_acc", mode="max", return_path=True
            )
            model = torch.load(os.path.join(model_path, "model.pkl"))
            results.append(tester.test_accuracy("test", model=model, device=device))
    else:
        for trial in analysis.trials:
            res_path = analysis.get_best_checkpoint(
                trial, metric="eval_acc", mode="max", return_path=True
            )
            result = json.load(open(os.path.join(res_path, "metrics.json")))
            results.append(result)
    json.dump(results, open(os.path.join(exp_dir, "results.json"), "w"), indent=4)
    print(results, "saved.")
