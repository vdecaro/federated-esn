import torch
from fedray.federation import ClientServerFederation
from .client import VanillaFedESNClient
from fedesn.common.server import FedESNServer
from torch_esn.model.reservoir import Reservoir

from ray.util.placement_group import PlacementGroup
from typing import Dict, List, Literal, Optional, Union


class VanillaESNFederation(ClientServerFederation):
    def __init__(
        self,
        dataset: str,
        batch_size: int,
        n_clients_or_ids: Union[int, List[str]],
        roles: List[str],
        server_id: str = "server",
        resources: Union[str, PlacementGroup] = "uniform",
        federation_id: str = "",
        is_tune: bool = False,
        bundle_offset: int = 0,
    ) -> None:
        super().__init__(
            server_template=FedESNServer,
            client_template=VanillaFedESNClient,
            n_clients_or_ids=n_clients_or_ids,
            roles=roles,
            server_config={},
            client_config={
                "dataset": dataset,
                "batch_size": batch_size,
            },
            server_id=server_id,
            resources=resources,
            federation_id=federation_id,
            is_tune=is_tune,
            bundle_offset=bundle_offset,
        )

    def ridge_train(
        self,
        reservoir: Union[Reservoir, Dict],
        l2: Optional[List[float]] = None,
        perc_rec: Optional[float] = 1.0,
        prev_A: Optional[torch.Tensor] = None,
        prev_B: Optional[torch.Tensor] = None,
    ) -> None:
        return super().train(
            server_args={
                "method": "ridge",
                "reservoir": reservoir,
                "l2": l2,
                "prev_A": prev_A,
                "prev_B": prev_B,
            },
            client_args={"method": "ridge", "perc_rec": perc_rec},
        )

    def ip_train(
        self,
        reservoir: Union[Reservoir, Dict],
        mu: float,
        sigma: float,
        eta: float,
        epochs: int,
    ) -> None:
        return super().train(
            server_args={
                "method": "ip",
                "reservoir": reservoir,
            },
            client_args={
                "method": "ip",
                "mu": mu,
                "sigma": sigma,
                "eta": eta,
                "epochs": epochs,
            },
        )

    def ip_ridge_train(
        self,
        reservoir: Union[Reservoir, Dict],
        mu: float,
        sigma: float,
        eta: float,
        epochs: int,
        l2: Optional[List[float]] = None,
        prev_A: Optional[torch.Tensor] = None,
        prev_B: Optional[torch.Tensor] = None,
    ) -> None:
        return super().train(
            server_args={
                "method": "both",
                "reservoir": reservoir,
                "l2": l2,
                "prev_A": prev_A,
                "prev_B": prev_B,
            },
            client_args={
                "method": "both",
                "mu": mu,
                "sigma": sigma,
                "eta": eta,
                "epochs": epochs,
            },
        )

    def test_accuracy(
        self,
        phase: Literal["train", "eval", "test"],
        model: Dict,
        device: Optional[str] = None,
    ) -> float:
        return super().test(phase, metric="accuracy", model=model, device=device)

    def test_likelihood(
        self,
        phase: Literal["train", "eval", "test"],
        model: Dict,
        mu: float,
        sigma: float,
        device: Optional[str] = None,
    ) -> float:
        return super().test(
            phase, metric="likelihood", model=model, mu=mu, sigma=sigma, device=device
        )
