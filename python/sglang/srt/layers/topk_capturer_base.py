import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    import numpy as np

    return int(np.prod(t.shape)) * t.dtype.itemsize


class BaseDeviceCache:
    def __init__(
        self, max_batch_size: int, num_layers: int, topk_size: int, device: str
    ):
        self.buffer = torch.zeros(
            (max_batch_size, num_layers, topk_size),
            dtype=torch.int32,
            device=device,
        )
        self.num_layers = num_layers
        self.topk_size = topk_size

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        batch = topk_indices.shape[0]
        topk_dim = min(topk_indices.shape[1], self.topk_size)
        self.buffer[:batch, layer_id, :topk_dim] = topk_indices[:, :topk_dim]

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)


class BaseHostCache:
    def __init__(self, num_tokens: int, num_layers: int, topk_size: int):
        self.buffer = torch.zeros(
            (num_tokens, num_layers, topk_size),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.topk_size = topk_size

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)


class BaseTopkCapturer:
    def __init__(
        self,
        num_tokens: int,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
    ):
        self.num_layers = num_layers
        self.topk_size = topk_size

        self.host_cache = BaseHostCache(num_tokens, num_layers, topk_size)
        self.device_cache = BaseDeviceCache(
            max_batch_size, num_layers, topk_size, device
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        self.device_cache.capture(layer_id, topk_indices)

    def _sync_to_host(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ):
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_rank,
            get_dp_local_info,
            is_dp_attention_enabled,
        )

        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = forward_batch.out_cache_loc.shape[0]

        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ].cpu()

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ) -> torch.Tensor:
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
        )
        return self.host_cache.buffer[cache_pool_idx]

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ):
        self._sync_to_host(forward_batch, can_run_graph, cuda_graph_batch)

    def is_enabled(self) -> bool:
        return True


class BaseTopkCapturerNoop:
    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        pass

    def get_topk(
        self, req_pool_idx: int, seqlen: int, req_to_token_pool: ReqToTokenPool
    ):
        return None

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ):
        pass

    def is_enabled(self) -> bool:
        return False
