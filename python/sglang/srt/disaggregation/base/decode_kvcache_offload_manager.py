from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager(ABC):
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        model_config: ModelConfig,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        self.model_config = model_config
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.tree_cache = tree_cache
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    @abstractmethod
    def offload_kv_cache(self, req: "Req") -> bool: ...

    @abstractmethod
    def check_offload_progress(self): ...

    @abstractmethod
    def ongoing_size(self) -> int: ...
