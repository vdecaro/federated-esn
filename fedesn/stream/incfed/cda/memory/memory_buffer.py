import torch
from torch.utils.data import DataLoader
from typing import Any, Callable, List, Optional, Tuple, Union


class OverloadedMemoryBuffer(object):
    def __init__(self, capacity: int = None) -> None:
        self._capacity = capacity
        self._q = []
        self._q_stats = {}

    def __len__(self):
        return len(self._q)

    def __getitem__(self, idx: int):
        return self._q[idx]

    def put(
        self, samples: Union[Tuple[Any, int], List[Tuple[Any, int]]], **kwargs
    ) -> None:

        if self._capacity is not None and len(self) + len(samples) >= self._capacity:
            n_discard = len(self) + len(samples) - self._capacity
        else:
            n_discard = 0

        samples = samples if isinstance(samples, List) else [samples]

        self._q += samples
        self._q = self._q[n_discard:]
        for k in kwargs:
            stat = kwargs[k] if isinstance(kwargs[k], List) else [kwargs[k]]
            self._q_stats[k] += stat
            self._q_stats[k] = self._q[n_discard:]

    def flush(self):
        if idx is None:
            idx = len(self)
            self._q = []
            self._q_stats = {k: [] for k in self._q_stats}

    def get_loader(self, idx, batch_size: int):
        return DataLoader(
            self._q[idx:], batch_size=batch_size, collate_fn=_memory_collate_fn
        )

    @property
    def samples(self):
        return self._q

    @property
    def get_stat(self, name: str) -> List:
        return self._q_stats[name]

    @property
    def is_empty(self):
        return self._q == []

    @property
    def capacity(self):
        return self._capacity


def _memory_collate_fn() -> Callable:
    def _aux(memory_zipped: List):
        x, y = zip * (memory_zipped)
        return torch.stack(x), torch.stack(y)

    return _aux
