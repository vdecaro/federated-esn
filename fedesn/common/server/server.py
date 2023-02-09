import fedray, torch

from fedray.node import FedRayNode
from torch_esn.model.reservoir import Reservoir
from torch_esn.optimization.ridge_regression import solve_ab_decomposition
from .aggregation import FedAvgAggregator, SumAggregator

from typing import Dict, List, Literal, Optional, Union


@fedray.remote
class FedESNServer(FedRayNode):
    def train(
        self,
        method: Literal["ip", "ridge", "both"],
        reservoir: Union[Reservoir, Dict],
        **kwargs,
    ):
        if isinstance(reservoir, Dict):
            reservoir = Reservoir(**reservoir)

        if method in ["ip", "both"]:
            while True:
                self.ip_round(reservoir, with_versioning=(method == "ip"))
                if method == "both":
                    self.ridge_round(reservoir, **kwargs)
        elif method == "ridge":
            self.ridge_round(reservoir, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")

    def ip_round(self, reservoir: Reservoir, with_versioning: bool = False):
        ip_aggregator = FedAvgAggregator()
        self.send("reservoir", {"model": reservoir})
        ip_aggregator.setup(self.neighbors)
        while not ip_aggregator.ready:
            ip_aggregator(self.receive())
        ip_params = ip_aggregator.compute()
        reservoir.load_state_dict(ip_params, strict=False)
        if with_versioning:
            self.update_version(reservoir=reservoir.cpu())

    def ridge_round(
        self,
        reservoir: Reservoir,
        l2: Optional[List[float]] = None,
        prev_A: Optional[torch.Tensor] = None,
        prev_B: Optional[torch.Tensor] = None,
        with_versioning: bool = True,
    ):

        ridge_aggregator = SumAggregator()
        self.send("reservoir", {"model": reservoir})
        ridge_aggregator.setup(self.neighbors)
        while not ridge_aggregator.ready:
            ridge_aggregator(self.receive())
        ab_dict = ridge_aggregator.compute()
        A = ab_dict["A"] + prev_A if prev_A is not None else ab_dict["A"]
        B = ab_dict["B"] + prev_B if prev_B is not None else ab_dict["B"]
        if l2 is None or isinstance(l2, float):
            readout = solve_ab_decomposition(A, B, l2=l2).cpu()
        else:
            readout = [solve_ab_decomposition(A, B, curr_l2).cpu() for curr_l2 in l2]
        if with_versioning:
            self.update_version(
                reservoir=reservoir.cpu(),
                readout=readout,
                A=ab_dict["A"].cpu(),
                B=ab_dict["B"].cpu(),
            )
