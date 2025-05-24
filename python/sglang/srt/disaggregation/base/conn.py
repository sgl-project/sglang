from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from typing import Callable
import torch


class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str
    gpu_id: int


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfers states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...
    
    @property
    def is_support_asnyc(self):
        return False
    
    def prepare_batch(self, sch : "Scheduler", batch : "ScheduleBatch"):
        raise NotImplementedError("the prepare_batch method is used in async mode, which is not implemented in BaseKVManager")
    
    def mark_layer_ready(self, layer_id : int):
        raise NotImplementedError("the mark_layer_ready method is used in async mode, which is not implemented in BaseKVManager")
    
    def insert_layer_callbacks(self, model : torch.nn.Module):
        def hack_forward_func(layer : torch.nn.Module, ori_func : Callable, layer_id : int):
            def wrapper_forward(*args, **kwargs):
                rst = ori_func(*args, **kwargs)
                self.mark_layer_ready(layer_id)
                layer._call_time += 1
                return rst
            return wrapper_forward
        if not hasattr(model, "_hacked_by_kv_manager") or not model._hacked_by_kv_manager:
            for layer_id, layer in enumerate(model.model.layers):
                ori_forward = layer.forward
                layer._call_time = 0
                layer.forward = hack_forward_func(layer, ori_forward, layer_id)
            self._layer_sizes = len(model.model.layers)
            model._hacked_by_kv_manager = True


class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self, mgr: BaseKVManager, bootstrap_addr: str, bootstrap_room: int
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Notify the decoder server about the kv indices length and aux index
        """
        ...

    @abstractmethod
    def send(self, kv_indices: npt.NDArray[np.int64]):
        """
        Send the kv cache at the given kv indices to the decoder server
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails
        """
        ...


class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None):
        """
        Notify the prefill server about the kv indices and aux index
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails
        """
        ...


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, port: int): ...
