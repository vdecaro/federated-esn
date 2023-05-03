import os
from typing import Dict, Literal
from ray import air, tune
from .trainable import VanillaFedESNTrainable
from .config import get_fed_args
from fedray.util.resources import get_resources_split


def run(
    perc: int,
    exp_dir: str,
    config: Dict,
    gpus_per_trial: float,
):
    os.makedirs(exp_dir, exist_ok=True)

    reporter = tune.CLIReporter(
        metric_columns={
            "training_iteration": "#Iter",
            "full_train_score": "FTR-Acc",
            "full_test_score": "FTS-Acc",
            "imp_train_score": "ITR-Acc",
            "imp_test_score": "ITS-Acc",
            "rand_train_score": "RTR-Acc",
            "rand_test_score": "RTS-Acc",
        },
        infer_limit=3,
    )

    fed_args = get_fed_args(config)
    n_nodes = 1 + len(fed_args["n_clients_or_ids"])
    resources = get_resources_split(
        n_nodes, num_cpus=2 + n_nodes, num_gpus=gpus_per_trial, is_tune=True
    )

    num_samples = 10

    tuner = tune.Tuner(
        tune.with_resources(VanillaFedESNTrainable, resources),
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=num_samples, reuse_actors=True),
        run_config=air.RunConfig(
            name=str(perc),
            local_dir=exp_dir,
            stop=lambda trial_id, result: True,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_frequency=1,
            ),
            verbose=1,
            progress_reporter=reporter,
        ),
    )
    return tuner.fit()
