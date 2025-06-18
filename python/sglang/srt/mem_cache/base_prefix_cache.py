from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
else:
    Req = Any  # Placeholder for Req type when not type checking


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices: Indices of the data on the device matched by the prefix.
        last_device_node: The last node on the device that was matched.
        last_host_node: The last node on the host that was matched.
                        Note that if the prefix cache does not support host cache,
                        this **must** be the same as `last_device_node`.
        host_indices_length: Length of the indices on the host, if applicable.
                             This is `None` if the prefix cache does not support host cache.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    host_indices_length: int = 0


class LoadHostResult(NamedTuple):
    """Result of loading from host to device.

    Attributes:
        new_device_indices: New indices on the device after loading.
        last_device_node: The last node on the device after loading.
    """

    new_device_indices: torch.Tensor
    new_last_device_node: Any


class BasePrefixCache(ABC):
    """Cache can be indexed by either rid or key."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        pass

    @abstractmethod
    def cache_finished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    def init_load_back(
        self,
        last_host_node: Any,
        host_indices_length: int,
    ) -> LoadHostResult:
        """
        Initial the load from host to device.
        Caller should provide the device indices that will be loaded into.
        The device node and host node should be returned from `match_prefix` call.
        If the prefix cache does support host cache, this method should be overridden.
        Otherwise, it should not be called and will raise a NotImplementedError.

        Args:
            last_host_node: The last node on the host that was matched.
            host_indices_length: length of the indices on the host.
        """
        raise NotImplementedError(
            "init_load_host should be overridden if the prefix cache supports host cache."
        )

    def ready_to_load_host_cache(self) -> Any:
        """
        This method is called after the prefill batch is prepared,
        and all the initialized load from host operations will be executed after this call.
        If the prefix cache supports host cache, this method must be overridden.
        Otherwise, it is just a no-op.
        """

    def check_host_cache(self) -> Any:
        """
        Check about host cache status and try to update evictable memory.
        This will be called before the prefill batch is prepared.
        If the prefix cache supports host cache, this method must be overridden.
        Otherwise, it is just a no-op.
        """

    @abstractmethod
    def evict(self, num_tokens: int):
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any):
        pass

    @abstractmethod
    def dec_lock_ref(self, node: Any):
        pass

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def take_events(self):
        return []
