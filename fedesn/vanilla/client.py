import fedray
from fedray.node import FedRayNode
from torch_esn.model.reservoir import Reservoir

from typing import Dict, Literal, Optional, Tuple

from torch_esn.wrapper.vanilla import VanillaESNWrapper


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
                    self.ridge_round()
        elif method == "ridge":
            self.ridge_round()

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

    def ridge_round(self):
        reservoir: Reservoir = self.receive().body["model"]
        A, B = self.wrapper.ridge_step(reservoir, with_readout=False)
        self.send("ridge_matrices", {"A": A, "B": B})

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
