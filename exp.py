import os

from collections import defaultdict
from typing import Dict
from ray import tune
from ray.tune.stopper import Stopper

from fedavg.server import FedAvgServer

def run_exp(config, exp_dir):
    early_stopping = TrialNoImprovementStopper(metric='eval_score', 
                                               mode='max', 
                                               patience_threshold=config['PATIENCE'])
    reporter = tune.CLIReporter(metric_columns={
                                    'training_iteration': '#Iter',
                                    'train_score': 'TR-Score',
                                    'eval_score': 'VL-Score', 
                                },
                                parameter_columns={'EPOCHS': 'EPOCHS', 'SIGMA': 'SIGMA', 'NORMALIZE': 'Norm', 'RHO': 'RHO', 'LEAKAGE': 'alpha'},
                                infer_limit=3,
                                metric='eval_score',
                                mode='max')

    n_clients = len(config['TRAIN_USERS'])
    
    return tune.run(
        FedAvgServer,
        name=f"{config['DATASET']}_ms",
        stop=early_stopping,
        local_dir=exp_dir,
        config=config,
        num_samples=25,
        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 1, "GPU": 1/(n_clients+1)} for _ in range(n_clients+1)]),
        keep_checkpoints_num=1,
        checkpoint_score_attr='eval_score',
        checkpoint_freq=1,
        max_failures=5,
        progress_reporter=reporter,
        verbose=1,
        reuse_actors=True
    )
    
class TrialNoImprovementStopper(Stopper):

    def __init__(self,
                 metric: str,
                 mode: str = None,
                 patience_threshold: int = 10):
        self._metric = metric
        self._mode = mode
        self._patience_threshold = patience_threshold

        self._trial_patience = defaultdict(lambda: 0)
        if mode == 'min':
            self._trial_best = defaultdict(lambda: float('inf')) 
        else:
            self._trial_best = defaultdict(lambda: -float('inf'))

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)

        better = (self._mode == 'min' and metric_result < self._trial_best[trial_id]) or \
            (self._mode == 'max' and metric_result > self._trial_best[trial_id])

        if better:
            self._trial_best[trial_id] = metric_result
            self._trial_patience[trial_id] = 0
        else:
            self._trial_patience[trial_id] += 1
        
        return self._trial_patience[trial_id] >= self._patience_threshold
    
    def stop_all(self):
        return False
