import os
from typing import Dict, Literal
from ray import air, tune
from ..trainables import get_trainable
from ..config import get_fed_args
from fedray.util.resources import get_resources_split


def run(
    phase: Literal["model_selection", "retraining"],
    exp_dir: str,
    config: Dict,
    gpus_per_trial: float,
):
    os.makedirs(exp_dir, exist_ok=True)

    reporter = tune.CLIReporter(
        metric_columns={
            "training_iteration": "#Iter",
            "eval_acc": "VL-Acc",
            "eval_score": "VL-Score",
        },
        infer_limit=3,
        metric="eval_score",
        mode="max",
    )

    if config["template"] in ["incfed", "fedip", "continual_fedip"]:
        fed_args = get_fed_args(config)
        n_nodes = 1 + len(fed_args["n_clients_or_ids"])
        resources = get_resources_split(
            n_nodes, num_cpus=2 + n_nodes, num_gpus=gpus_per_trial, is_tune=True
        )
    else:
        resources = {"cpu": 2, "gpu": gpus_per_trial}

    if "continual" in config["template"]:
        config["phase"] = phase

    if phase == "model_selection":
        num_samples = 100 if not "continual" in config["template"] else 1
    else:
        num_samples = 5

    tuner = tune.Tuner(
        tune.with_resources(get_trainable(config["template"]), resources),
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=num_samples, reuse_actors=True),
        run_config=air.RunConfig(
            name=phase,
            local_dir=exp_dir,
            stop=lambda trial_id, result: True,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="eval_score",
                checkpoint_frequency=1,
            ),
            verbose=1,
            progress_reporter=reporter,
        ),
    )
    return tuner.fit()
