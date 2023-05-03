import ray, threading
import numpy as np
import ray

from ray.util.placement_group import PlacementGroup
from fedray._private.communication.broker import _get_or_create_broker
from fedray.node import FedRayNode, VirtualNode
from fedray.federation.generic import Federation

from typing import Dict, List, Literal, Optional, Type, Union


class ClientServerFederation(Federation):
    def __init__(
        self,
        server_template: Type[FedRayNode],
        client_template: Type[FedRayNode],
        n_clients_or_ids: Union[int, List[str]],
        roles: List[str],
        server_config: Dict = {},
        client_config: Dict = {},
        server_id: str = "server",
        resources: Union[str, PlacementGroup] = "uniform",
        federation_id: str = "",
        is_tune: bool = False,
        bundle_offset: int = 0,
    ) -> None:

        if isinstance(n_clients_or_ids, int):
            c_ids = [f"client_{i}" for i in range(n_clients_or_ids)]
        else:
            c_ids = n_clients_or_ids

        nodes = [
            VirtualNode(
                server_template, server_id, federation_id, "train", server_config
            )
        ]
        for c_id, role in zip(c_ids, roles):
            nodes.append(
                VirtualNode(client_template, c_id, federation_id, role, client_config)
            )

        super(ClientServerFederation, self).__init__(
            nodes, "star", resources, federation_id, is_tune, bundle_offset
        )

    def train(self, server_args: Dict, client_args: Dict, blocking: bool = False):
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

        server_args = [server_args]
        client_args = [
            client_args[i] if isinstance(client_args, List) else client_args
            for i, _ in enumerate(train_nodes[1:])
        ]
        train_args = server_args + client_args

        self._runtime_remotes = [
            node.handle._train.remote(**train_args[i])
            for i, node in enumerate(train_nodes)
        ]
        self._runtime = threading.Thread(
            target=ray.get, args=[self._runtime_remotes], daemon=True
        )
        self._runtime.start()
        if blocking:
            self._runtime.join()

    def test(self, phase: Literal["train", "eval", "test"], **kwargs) -> List:
        test_nodes = []
        for i, node in enumerate(self._nodes[1:], start=2 + self._bundle_offset):
            if phase in node.role:
                test_nodes.append(node)
                if node.handle is None:
                    node.build(i, self._pg)
        remotes = [node.handle.test.remote(phase, **kwargs) for node in test_nodes]

        results = ray.get(remotes)
        values, weights = zip(*results)
        return np.average(values, weights=weights, axis=0)

    def pull_version(
        self,
        node_ids: Union[str, List[str]] = "server",
        timeout: Optional[float] = None,
    ):
        return super().pull_version(node_ids, timeout)

    @property
    def server(self):
        return self._nodes[0].handle

    def reset(self, server_config: Dict, client_config: Dict):
        outcomes = [self.server.reset.remote(**server_config)]
        for client in self._nodes[1:]:
            outcomes.append(client.handle.reset.remote(**client_config))
        return all(ray.get(outcomes))
