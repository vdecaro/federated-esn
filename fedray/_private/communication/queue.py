import copyreg, threading, pickle, queue, time
from queue import Queue as _Queue
from queue import Full

# Make Queue a new-style class, so it can be used with copy_reg
class Queue(_Queue, object):
    Empty = queue.Empty
    Full = queue.Full

    def put(self, item, block=True, timeout=None, index=None):
        """Put an item into the queue.
        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until a free slot is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot
        is immediately available, else raise the Full exception ('timeout'
        is ignored in that case).
        """
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            self._put(item, index)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def _put(self, item, index) -> None:
        if index is None:
            self.queue.append(item)
        else:
            self.queue.insert(index, item)


def pickle_queue(q):
    q_dct = q.__dict__.copy()
    del q_dct["mutex"]
    del q_dct["not_empty"]
    del q_dct["not_full"]
    del q_dct["all_tasks_done"]
    return Queue, (), q_dct


def unpickle_queue(state):
    q = state[0]()
    q.mutex = threading.Lock()
    q.not_empty = threading.Condition(q.mutex)
    q.not_full = threading.Condition(q.mutex)
    q.all_tasks_done = threading.Condition(q.mutex)
    q.__dict__ = state[2]
    return q


copyreg.pickle(Queue, pickle_queue, unpickle_queue)
