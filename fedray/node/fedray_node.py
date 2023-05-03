import ray, copy, time

from fedray._private.communication.broker import FedRayBroker, Message
from fedray._private.communication.queue import Queue
from fedray._private.exceptions import EndProcessException

from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, Union


class FedRayNode(object):
    def __init__(self, node_id: str, role: str, federation_id: str = "", **kwargs):
        """_summary_
        Args:
            node_id (str): _description_
            role (str): _description_
        Raises:
            ValueError: _description_
        """

        # Node hyperparameters
        self._fed_id: str = federation_id
        self._id: str = node_id
        self._role: str = role

        # Communication interface
        self._broker: FedRayBroker = None
        self._message_queue: Queue = None

        # Node's version
        self._version: int = 0
        self._version_buffer: Queue = None
        self._node_metrics: Dict[str, Any] = {}

        # Buildup function
        self._node_config = kwargs
        self.build(**kwargs)

    def build(self, **kwargs):
        """_summary_"""
        pass

    def _setup_train(self):
        """_summary_"""
        if self._broker is None:
            self._broker = ray.get_actor(f"{self._fed_id}/broker")
        self._message_queue = Queue()
        self._version = 0
        self._version_buffer = Queue()
        return True

    def _train(self, **train_args):
        try:
            self.train(**train_args)
        except EndProcessException:
            print(f"Node {self.id} is exiting.")

        return self._node_metrics

    def train(self, **train_args) -> Dict:
        """_summary_
        Raises:
            NotImplementedError: _description_
        Returns:
            Dict: _description_
        """
        raise NotImplementedError

    def test(self, phase: Literal["train", "eval", "test"], **kwargs):
        """_summary_
        Args:
            phase (Literal[&#39;train&#39;, &#39;eval&#39;, &#39;test&#39;]):
                _description_
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def send(self, header: str, body: Dict, to: Optional[Union[str, List[str]]] = None):
        """_summary_
        Args:
            header (str): _description_
            body (Dict): _description_
            to (Optional[Union[str, List[str]]], optional): _description_. Defaults to None.
        """
        if isinstance(to, str):
            to = [to]

        msg = Message(header=header, sender_id=self._id, body=body)
        ray.get([self._broker.publish.remote(msg, to)])

    def receive(self, timeout: Optional[float] = None) -> Message:
        """_summary_
        Args:
            timeout (Optional[float], optional): _description_. Defaults to None.
        Raises:
            EndProcessException: _description_
        Returns:
            Message: _description_
        """
        try:
            msg = self._message_queue.get(timeout=timeout)
        except Queue.Empty:
            msg = None

        if msg is not None and msg.header == "STOP":
            raise EndProcessException
        return msg

    def update_version(self, **kwargs):
        """_summary_
        Args:
            **kwargs: _description_
        Raises:
            NotImplementedError: _description_
        """
        to_save = {k: copy.deepcopy(v) for k, v in kwargs.items()}
        version_dict = {
            "id": self.id,
            "n_version": self.version,
            "timestamp": time.time(),
            "model": to_save,
        }
        self._version_buffer.put(version_dict)
        self._version += 1

    def stop(self):
        """_summary_"""
        self._message_queue.put(Message("STOP"), index=0)

    def enqueue(self, msg: ray.ObjectRef):
        """_summary_
        Args:
            msg (ray.ObjectRef): _description_
        Returns:
            _type_: _description_
        """
        self._message_queue.put(msg)
        return True

    def _invalidate_neighbors(self):
        """_summary_"""
        del self.neighbors

    def _pull_version(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._version_buffer.get(block=True)

    @property
    def id(self) -> str:
        """_summary_
        Returns:
            str: _description_
        """
        return self._id

    @property
    def version(self) -> int:
        """_summary_
        Returns:
            int: _description_
        """
        return self._version

    @property
    def is_train_node(self) -> bool:
        """_summary_
        Returns:
            bool: _description_
        """
        return "train" in self._role.split("-")

    @property
    def is_eval_node(self) -> bool:
        """_summary_
        Returns:
            bool: _description_
        """
        return "eval" in self._role.split("-")

    @property
    def is_test_node(self) -> bool:
        """_summary_
        Returns:
            bool: _description_
        """
        return "test" in self._role.split("-")

    @cached_property
    def neighbors(self) -> List[str]:
        """_summary_
        Returns:
            List[str]: _description_
        """
        return ray.get(self._broker.get_neighbors.remote(self.id))
