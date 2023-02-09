import time, random, torch, fedray

from torch_esn.model.reservoir import Reservoir
from torch_esn.optimization.ridge_regression import (
    compute_ridge_matrices,
    validate_readout,
    solve_ab_decomposition,
)

from .memory.beta_drift_aware_buffer import BetaDriftDetectionMemory

from torch_esn.data.datasets.digit5.dataset import Digit5
from torch_esn.data.datasets.digit5.benchmark import digit5_benchmark

from typing import Dict, Literal, Optional, Tuple

DATA_DIR = None


@fedray.remote
class CDAIncFedClient(fedray.node.FedRayNode):
    def build(
        self,
        dataset: str,
        n_experiences: int,
        delta_window: int,
        sensitivity: float,
        threshold: float,
        min_samples_train: float,
        **kwargs
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.delta = delta_window
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.min_samples_train = min_samples_train

        self.train_data, self.test_data = Digit5(root=DATA_DIR, split="train"), Digit5(
            root=DATA_DIR, split="test"
        )
        self.train_data.apply_local_cluster_partition()
        self.benchmark = digit5_benchmark(
            self.train_data, self.test_data, n_experiences=n_experiences
        )

        self.score_fn = lambda Y, Y_pred: (
            torch.sum(Y.argmax(-1).flatten() == Y_pred.argmax(-1).flatten()) / Y.size(0)
        ).item()

    def run(self):
        msg = self.receive()
        reservoir: Reservoir = msg.body["reservoir"].to(self.device)
        readout: torch.Tensor = msg.body["readout"].to(self.device)
        short_term = BetaDriftDetectionMemory(
            self.delta, self.sensitivity, self.threshold, self.min_samples_train
        )
        A, B = 0, 0
        drift_occurred = -1
        for train_exp in self.benchmark.train_stream:
            for x, y in train_exp.dataset:
                time.sleep(0.5 * random.random())
                h = reservoir(x.to(self.device))
                y_pred = h @ readout
                short_term.put((h.to("cpu"), y), confidence=torch.max(y_pred))
                if drift_occurred == -1:
                    drift_occurred = short_term.test_drift(short_term)

                else:
                    if len(short_term[drift_occurred:]) > self.min_samples_train:
                        loader = short_term.get_loader(drift_occurred, batch_size=50)
                        drift_A, drift_B = compute_ridge_matrices(loader, reservoir)
                        self.send("ridge_matrices", {"A": drift_A, "B": drift_B})
                        A, B = A + drift_A, B + drift_B
                        readout = solve_ab_decomposition(A, B)

    def test(
        self,
        phase: Literal["train", "eval", "test"],
        model: Dict,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
    ) -> Tuple[float, int]:
        accuracy = validate_readout(
            model["readout"],
            self.loader,
            self.score_fn,
            model["reservoir"].eval(),
            device,
        )
        return (self.id, accuracy, len(self.dataset) * self.dataset.seq_length)
