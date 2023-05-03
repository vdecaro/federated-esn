import ray, threading
import numpy as np

from fedray.node import VirtualNode

from fedray._private.communication.broker import _get_or_create_broker
from fedray._private.communication.message import Message
from fedray.util.resources import get_resources_split
from ray.util.placement_group import PlacementGroup

from typing import Dict, List, Literal, Optional, Union


class Federation(object):
    def __init__(
        self,
        nodes: List[VirtualNode],
        topology: Union[str, np.ndarray],
        resources: Union[str, PlacementGroup] = "uniform",
        federation_id: str = "",
        is_tune: bool = False,
        bundle_offset: int = 0,
    ):

        self._fed_id = federation_id
        self._name = "supervisor"
        self._nodes: List[VirtualNode] = nodes
        self._topology: Union[str, np.ndarray] = topology

        if not is_tune:
            if isinstance(resources, str):
                self._pg = get_resources_split(
                    len(self._nodes), split_strategy=resources
                )
            else:
                self._pg = resources
        else:
            self._pg = ray.util.get_current_placement_group()
        self._bundle_offset = 1 + bundle_offset if is_tune else bundle_offset

        self._broker = None
        self._state: Literal["IDLE", "RUNNING"] = "IDLE"
        self._runtime_remotes: List[ray.ObjectRef] = None
        self._runtime: threading.Thread = None

    def __getitem__(self, node_id: str):
        for node in self._nodes:
            if node.id == node_id:
                return node.handle
        raise ValueError(f"Identifier {node_id} not found in process.")

    def train(self, blocking: bool = False, **train_args):
        raise NotImplementedError

    def test(self, phase: Literal["train", "eval", "test"], **kwargs) -> List:
        raise NotImplementedError

    def pull_version(
        self, node_ids: Union[str, List[str]], timeout: Optional[float] = None
    ):
        to_pull = [node_ids] if isinstance(node_ids, str) else node_ids
        to_pull = [
            node.handle._pull_version.remote()
            for node in self._nodes
            if node.id in to_pull
        ]

        if timeout is None:
            new_versions = ray.get(to_pull)
            return new_versions[0] if len(to_pull) == 1 else new_versions
        else:
            new_versions, _ = ray.wait(to_pull, timeout=timeout)
            if len(new_versions) == 0:
                return None
            else:
                return new_versions[0] if len(to_pull) == 1 else new_versions

    def send(self, header: str, body: Dict, to: Optional[Union[str, List[str]]] = None):
        """_summary_
        Args:
            header (str): _description_
            body (Dict): _description_
            to (Optional[Union[str, List[str]]], optional): _description_.
                Defaults to None.
        """
        if isinstance(to, str):
            to = [to]

        msg = Message(header=header, sender_id=self._name, body=body)
        ray.get([self._broker.publish.remote(msg, to)])

    def stop(self) -> None:
        ray.get(
            [
                node.handle.stop.remote()
                for node in self._nodes
                if node.built and "train" in node.role
            ]
        )
        self._runtime.join()
        self._state = "IDLE"

    @property
    def running(self) -> bool:
        return (
            self._state == "RUNNING"
            and self._runtime is not None
            and self._runtime.is_alive()
        )

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def node_ids(self) -> List[str]:
        return [node.id for node in self._nodes]

    @property
    def resources(self) -> Dict[str, Dict[str, Union[int, float]]]:
        res_arr = self._pg.bundle_specs
        resources = {
            "all": {"CPU": res_arr[0]["CPU"], "GPU": 0},
            "broker": {"CPU": res_arr[0]["CPU"]},
        }
        for i, node in enumerate(self._nodes, start=1):
            resources[node.id] = res_arr[i]
            resources["all"]["CPU"] += res_arr[i]["CPU"]
            if "GPU" in res_arr[i]:
                resources["all"]["GPU"] += res_arr[i]["GPU"]

        return resources
