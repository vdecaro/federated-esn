import fedray
from fedray.node import FedRayNode
from torch_esn.model.reservoir import Reservoir

from typing import Dict, Literal, Optional, Tuple

from torch_esn.wrapper.vanilla import VanillaESNWrapper
from torch_esn.optimization.ridge_regression import compress_ridge_matrices


@fedray.remote
class VanillaFedESNClient(FedRayNode):
    def build(self, dataset: str, batch_size: int) -> None:
        self.wrapper = VanillaESNWrapper(dataset, self.id, batch_size)

    def train(self, method: Literal["ip", "ridge", "both"], **kwargs):
        if method in ["ip", "both"]:
            while True:
                self.ip_round(
                    kwargs["mu"], kwargs["sigma"], kwargs["eta"], kwargs["epochs"]
                )
                if method == "both":
                    self.ridge_round(kwargs["perc_rec"])
        elif method == "ridge":
            self.ridge_round(kwargs["perc_rec"])

        else:
            raise ValueError(f"Unknown training method: {method}")

    def ip_round(self, mu: float, sigma: float, eta: float, epochs: int):
        reservoir: Reservoir = self.receive().body["model"]
        reservoir = self.wrapper.ip_step(
            reservoir=reservoir, mu=mu, sigma=sigma, eta=eta, epochs=epochs
        )
        state_dict = reservoir.state_dict()
        n_samples = self.wrapper.get_dataset_size()
        self.send(
            header="ip_update",
            body={
                "net_a": state_dict["net_a"].cpu(),
                "net_b": state_dict["net_b"].cpu(),
                "n_samples": n_samples,
            },
        )

    def ridge_round(self, perc_rec: float):
        reservoir: Reservoir = self.receive().body["model"]
        A, B = self.wrapper.ridge_step(reservoir, l2=None, with_readout=False)
        imp_A, imp_B, imp_m_A, imp_m_B = compress_ridge_matrices(
            A, B, perc_rec, alpha=1.0
        )
        n_chosen = imp_m_A[0].sum()
        n = imp_m_A[0].shape[0]

        rand_A, rand_B, rnd_m_A, rnd_m_B = compress_ridge_matrices(
            A, B, n_chosen / n, alpha=0.0
        )

        self.send(
            "ridge_matrices",
            {
                "full_A": A.cpu(),
                "full_B": B.cpu(),
                "rand_A": rand_A.cpu(),
                "rand_B": rand_B.cpu(),
                "imp_A": imp_A.cpu(),
                "imp_B": imp_B.cpu(),
                "perc_chosen": n_chosen / n,
            },
        )

    def test(
        self,
        phase: str,
        model: Dict,
        metric: Literal["accuracy", "likelihood"],
        device: Optional[str] = None,
        **kwargs,
    ) -> Tuple[float, int]:
        if metric == "accuracy":
            readout, reservoir = model["readout"], model["reservoir"]
            metric = self.wrapper.test_accuracy(
                readout=readout, reservoir=reservoir, device=device
            )
        elif metric == "likelihood":
            reservoir = model["reservoir"]
            mu, sigma = kwargs["mu"], kwargs["sigma"]
            metric = self.wrapper.test_likelihood(
                reservoir=reservoir, mu=mu, sigma=sigma, device=device
            )

        n_samples = self.wrapper.get_dataset_size()
        return metric, n_samples
