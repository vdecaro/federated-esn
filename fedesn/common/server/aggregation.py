from collections import defaultdict
from copy import deepcopy
from typing import Dict, List


class FedAggregator(object):
    def __init__(self):
        self.expected_clients = []
        self.residual_ids = []
        self.state = defaultdict(lambda: 0)

    def setup(self, client_ids: List[str]):
        self.expected_clients = deepcopy(client_ids)
        self.received_clients = []
        self.state = defaultdict(lambda: 0)

    def __call__(self, msg):
        if msg.sender_id not in self.expected_clients:
            raise ValueError(
                f"Message received from client {msg.sender_id}, not included in the \
                    expected clients."
            )
        self.received_clients.append(msg.sender_id)
        self.update(msg.body)

    def update(self, client_dict: Dict):
        raise NotImplementedError

    def compute(self):
        return self.state

    @property
    def ready(self):
        return all(
            [(expected in self.received_clients) for expected in self.expected_clients]
        )


class FedAvgAggregator(FedAggregator):
    def update(self, client_dict: Dict):
        n_samples = client_dict.pop("n_samples")
        self.state["n_samples"] += n_samples
        for k in client_dict:
            self.state[k] += client_dict[k] * n_samples

    def compute(self):
        n_samples = self.state["n_samples"]
        return {k: self.state[k] / n_samples for k in self.state}


class SumAggregator(FedAggregator):
    def update(self, client_dict: Dict):
        for k in client_dict:
            self.state[k] += client_dict[k]
