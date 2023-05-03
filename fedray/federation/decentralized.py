import ray, threading
import numpy as np
from fedray._private.communication.broker import _get_or_create_broker

from fedray.node import FedRayNode, VirtualNode
from fedray.federation.generic import Federation

from ray.util.placement_group import PlacementGroup

from typing import Dict, List, Literal, Type, Union


class DecentralizedFederation(Federation):
    def __init__(
        self,
        node_template: Type[FedRayNode],
        n_nodes_or_ids: Union[int, List[str]],
        roles: List[str],
        topology: Union[str, np.ndarray],
        node_config: Union[Dict, List[Dict]],
        resources: Union[str, PlacementGroup] = "uniform",
        federation_id: str = "",
        is_tune: bool = True,
        bundle_offset: int = 0,
    ) -> None:

        if isinstance(n_nodes_or_ids, int):
            node_ids = [f"node_{i}" for i in range(n_nodes_or_ids)]
        else:
            node_ids = n_nodes_or_ids

        nodes = [
            VirtualNode(
                node_template,
                node_id,
                federation_id,
                role,
                node_config[i] if isinstance(node_config, list) else node_config,
            )
            for i, (node_id, role) in enumerate(zip(node_ids, roles))
        ]

        super(DecentralizedFederation, self).__init__(
            nodes, topology, resources, federation_id, is_tune, bundle_offset
        )

    def train(self, train_args: Union[Dict, List[Dict]], blocking: bool = False):
        if self._broker is None:
            self._broker = _get_or_create_broker(
                self._pg, self._fed_id, self._bundle_offset
            )
        train_nodes = []
        for i, node in enumerate(self._nodes, start=1 + self._bundle_offset):
            if "train" in node.role:
                if not node.built:
                    node.build(i, self._pg)
                train_nodes.append(node)

        ray.get(
            self._broker.link_nodes.remote(
                [node.id for node in train_nodes], self._topology
            )
        )
        ray.get([node.handle._setup_train.remote() for node in train_nodes])
        train_args = [
            (train_args[i] if isinstance(train_args, List) else train_args)
            for i, _ in enumerate(train_nodes)
        ]
        self._runtime = threading.Thread(
            target=ray.get,
            args=[[node.handle._train.remote(**train_args[i]) for node in train_nodes]],
            daemon=True,
        )
        self._runtime.start()
        if blocking:
            self._runtime.join()

    def test(self, phase: Literal["train", "eval", "test"], **kwargs) -> List:
        test_nodes = []
        for i, node in enumerate(self._nodes, start=1 + self._bundle_offset):
            if node.role is not None and phase in node.role:
                test_nodes.append(node)
                if node.handle is None:
                    node.build(i, self._pg)
        remotes = [node.handle.test.remote(phase, **kwargs) for node in test_nodes]

        results = ray.get(remotes)
        values, weights = zip(*results)
        return np.average(values, weights=weights, axis=0)
