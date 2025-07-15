import logging
import multiprocessing as mp
import pickle
import struct
from multiprocessing import shared_memory
from multiprocessing.managers import BaseManager
from typing import Dict, List

logger = logging.getLogger(__name__)

"""
This class will be use in scheduler and dp controller
If this class is placed in the dp controller,
it will cause circular references, so it is placed in a separate file.
"""


class DPBalanceMeta:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self._manager = mp.Manager()
        self.mutex = self._manager.Lock()

        init_local_tokens = [0] * self.num_workers
        init_onfly_info = [self._manager.dict() for _ in range(self.num_workers)]

        self.shared_state = self._manager.Namespace()
        self.shared_state.local_tokens = self._manager.list(init_local_tokens)
        self.shared_state.onfly_info = self._manager.list(init_onfly_info)

    def destructor(self):
        # we must destructor this class manually
        self._manager.shutdown()

    def get_shared_onfly(self) -> List[Dict[int, int]]:
        return [dict(d) for d in self.shared_state.onfly_info]

    def set_shared_onfly_info(self, data: List[Dict[int, int]]):
        self.shared_state.onfly_info = data

    def get_shared_local_tokens(self) -> List[int]:
        return list(self.shared_state.local_tokens)

    def set_shared_local_tokens(self, data: List[int]):
        self.shared_state.local_tokens = data

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_manager"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._manager = None
