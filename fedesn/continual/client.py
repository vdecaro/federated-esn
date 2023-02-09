import fedray
from fedray.node import FedRayNode
from torch_esn.model.reservoir import Reservoir

from typing import Dict, Literal, Optional, Tuple

from torch_esn.wrapper.continual import ContinualESNWrapper


@fedray.remote
class ContinualFedESNClient(FedRayNode):
    def build(
        self,
        dataset: str,
        batch_size: int,
        strategy: Literal["naive", "joint", "replay"],
    ) -> None:

        self.wrapper = ContinualESNWrapper(dataset, self.id, batch_size, strategy)

    def train(self, method: Literal["ip", "ridge"], context: int, **kwargs):

        if method in ["ip", "both"]:
            while True:
                self.ip_round(
                    context,
                    kwargs["mu"],
                    kwargs["sigma"],
                    kwargs["eta"],
                    kwargs["epochs"],
                )
                if method == "both":
                    self.ridge_round(context)

        elif method == "ridge":
            self.ridge_round(context)

        else:
            raise ValueError(f"Unknown training method: {method}")

    def ip_round(self, context: int, mu: float, sigma: float, eta: float, epochs: int):

        reservoir: Reservoir = self.receive().body["model"]
        reservoir = self.wrapper.ip_step(
            context=context,
            reservoir=reservoir,
            mu=mu,
            sigma=sigma,
            eta=eta,
            epochs=epochs,
        )
        state_dict = reservoir.state_dict()
        n_samples = self.wrapper.get_dataset_size(context, with_strategy=True)
        self.send(
            header="ip_update",
            body={
                "net_a": state_dict["net_a"].cpu(),
                "net_b": state_dict["net_b"].cpu(),
                "n_samples": n_samples,
            },
        )

    def ridge_round(self, context: int):
        reservoir: Reservoir = self.receive().body["model"]
        A, B = self.wrapper.ridge_step(
            context=context, reservoir=reservoir, with_readout=False
        )
        self.send("ridge_matrices", {"A": A, "B": B})

    def test(
        self,
        phase: str,
        context: int,
        model: Dict,
        metric: Literal["accuracy", "likelihood"],
        device: Optional[str] = None,
        **kwargs,
    ) -> Tuple[float, int]:

        if metric == "accuracy":
            readout, reservoir = model["readout"], model["reservoir"]
            metric = self.wrapper.test_accuracy(
                context=context, readout=readout, reservoir=reservoir, device=device
            )
        elif metric == "likelihood":
            reservoir = model["reservoir"]
            mu, sigma = kwargs["mu"], kwargs["sigma"]
            metric = self.wrapper.test_likelihood(
                context=context, reservoir=reservoir, mu=mu, sigma=sigma, device=device
            )

        n_samples = self.wrapper.get_dataset_size(context)
        return metric, n_samples
