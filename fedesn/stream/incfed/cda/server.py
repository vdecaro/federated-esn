import fedray

from torch_esn.model.reservoir import Reservoir
from torch_esn.optimization.ridge_regression import solve_ab_decomposition

from typing import Dict, List


@fedray.remote
class CDAIncFedServer(fedray.node.FedRayNode):
    def build(
        self, reservoir_params: Dict, l2: List[float], update_every_n: int, **kwargs
    ):
        self.reservoir, self.readout = Reservoir(**reservoir_params), None
        self.l2 = l2
        self.update_every_n = update_every_n

    def run(self):
        self.send("model", {"model": self.reservoir})
        client_updates = {
            client_id: {"A": 0, "B": 0, "n_updates": 0} for client_id in self.neighbors
        }
        n_updates = 0
        while True:
            msg = self.receive()
            client_updates[msg.sender_id]["A"] += msg.body["A"]
            client_updates[msg.sender_id]["B"] += msg.body["B"]
            client_updates[msg.sender_id]["n_updates"] += 1
            n_updates += 1
            if n_updates % self.update_every_n == 0:
                A = sum([client_updates[c_id]["A"] for c_id in client_updates])
                B = sum([client_updates[c_id]["B"] for c_id in client_updates])
                self.readout = solve_ab_decomposition(A, B)
                self.update_version(
                    {"reservoir": self.reservoir, "readout": self.readout}
                )
