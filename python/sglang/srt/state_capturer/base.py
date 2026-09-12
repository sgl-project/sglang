import dataclasses
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    HostTensorAllocator,
    _cuda_host_unregister,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class BaseDeviceCache:
    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
    ):
        self.buffer = torch.zeros(
            (max_batch_size, num_layers, topk_size),
            dtype=torch.int32,
            device=device,
        )
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        batch = topk_indices.shape[0]
        self.buffer[:batch, layer_id, :] = topk_indices

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"DeviceCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"size={size_mb:.2f} MB"
        )


class BaseHostCache:
    def __init__(
        self, num_tokens: int, num_layers: int, topk_size: int, name: str, device: str
    ):
        alloc = ALLOC_MEMORY_FUNCS[device]
        self.buffer = alloc(
            (num_tokens, num_layers, topk_size),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
            allocator=HostTensorAllocator(),
        )
        self.buffer.zero_()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()
        # L1 rows use reusable KV slots; L2 rows use stable HiCache host slots.
        self.valid = torch.zeros(num_tokens, dtype=torch.bool, device="cpu")
        self.must_be_valid = torch.zeros(num_tokens, dtype=torch.bool, device="cpu")
        self._hicache_rows = {}

    def store(self, cache_pool_idx: torch.Tensor, values: torch.Tensor) -> None:
        """Publish routed-expert rows for the current KV-slot owners."""
        cache_pool_idx = cache_pool_idx.cpu()
        self.buffer[cache_pool_idx] = values.cpu()
        self.valid[cache_pool_idx] = True

    def backup_to_hicache(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> None:
        """Save L1 rows under the L2 KV host-slot identity."""
        device_indices = device_indices.cpu()
        host_indices = host_indices.cpu()
        if len(device_indices) != len(host_indices):
            raise ValueError("HiCache device/host routed-expert index length mismatch")
        # HiCache transfers full KV pages. Some page slots are layout padding
        # and therefore have no forward-time routed-expert capture; preserve
        # their initialized rows instead of rejecting the whole page.
        rows = self.buffer[device_indices].clone()
        for host_idx, row in zip(host_indices.tolist(), rows):
            self._hicache_rows[host_idx] = row

    def restore_from_hicache(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ) -> bool:
        """Remap L2 rows to freshly allocated KV slots without stale reads."""
        host_indices = host_indices.cpu()
        device_indices = device_indices.cpu()
        self.must_be_valid[device_indices] = True
        if len(host_indices) != len(device_indices):
            raise ValueError("HiCache host/device routed-expert index length mismatch")

        # Invalidate first so a missing sidecar never exposes a prior owner.
        self.valid[device_indices] = False
        missing = [idx for idx in host_indices.tolist() if idx not in self._hicache_rows]
        if missing:
            logger.error(
                "Missing routed-expert HiCache rows for host slots %s", missing[:16]
            )
            return False

        rows = torch.stack([self._hicache_rows[idx] for idx in host_indices.tolist()])
        self.buffer[device_indices] = rows
        self.valid[device_indices] = True
        return True

    def invalidate_hicache(self, host_indices: torch.Tensor) -> None:
        """Forget L2 sidecar rows when host slots acquire non-L2 content."""
        for host_idx in host_indices.cpu().tolist():
            self._hicache_rows.pop(host_idx, None)

    def invalidate(self, cache_pool_idx: torch.Tensor) -> None:
        cache_pool_idx = cache_pool_idx.cpu()
        self.must_be_valid[cache_pool_idx] = True
        self.valid[cache_pool_idx] = False

    def clear(self) -> None:
        """Clear both reusable HostCache rows and HiCache L2 sidecar rows."""
        self.buffer.zero_()
        self.valid.zero_()
        self.must_be_valid.zero_()
        self._hicache_rows.clear()

    def clear_hicache(self) -> None:
        """Backward-compatible alias for callers added with HiCache support."""
        self.clear()

    def destroy(self):
        if self.buffer is None:
            return
        if _is_cuda or _is_hip:
            _cuda_host_unregister(self.buffer)
        self.buffer = None

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_gb = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"HostCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"size={size_gb:.2f} GB"
        )


@dataclasses.dataclass
class TopkCaptureOutput:
    """Holds GPU tensors captured during forward for overlap scheduling.
    map_device_tensors() D2H-copies them before copy_done.record() (may run on
    the dedicated result-copy stream); finalize() runs after copy_done.synchronize().
    """

    out_cache_loc: torch.Tensor
    topk: torch.Tensor
    host_cache: BaseHostCache

    def map_device_tensors(self, fn):
        # Device-tensor fields only; caller injects the copy+safety primitive
        # (see GenerationBatchResult.copy_to_cpu).
        self.out_cache_loc = fn(self.out_cache_loc)
        self.topk = fn(self.topk)

    def finalize(self):
        self.host_cache.store(self.out_cache_loc, self.topk)


class BaseTopkCapturer:
    def __init__(
        self,
        num_tokens: int,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
        device_topk_size: Optional[int] = None,
    ):
        """device_topk_size defaults to topk_size; pass a different value when
        the device buffer needs extra columns (e.g. fused shared experts) that
        are dropped before writing to host_cache via [:topk_size] truncation.
        """
        self.num_layers = num_layers
        self.topk_size = topk_size

        self.host_cache = BaseHostCache(
            num_tokens, num_layers, topk_size, name=name, device=device
        )
        self.device_cache = BaseDeviceCache(
            max_batch_size,
            num_layers,
            device_topk_size if device_topk_size is not None else topk_size,
            device,
            name=name,
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        self.device_cache.capture(layer_id, topk_indices)

    def destroy(self):
        self.host_cache.destroy()

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        """Return the device_cache slice for this forward batch, GPU-resident.

        Default assumes per-rank-local capture: each rank writes [:local_num_tokens)
        to its own device_cache. Subclasses with global-tensor capture semantics
        (e.g. shared cuda graph buffer indexed by dp_rank) should override and
        consume can_run_graph / cuda_graph_batch.
        """
        del can_run_graph, cuda_graph_batch  # reserved for subclass override
        num_tokens = forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[:num_tokens, :, : self.topk_size]

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ) -> torch.Tensor:
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        start_len = min(start_len, seqlen - 1)
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start_len : seqlen - 1]
            .cpu()
            .clone()
        )
        invalid_mask = (
            self.host_cache.must_be_valid[cache_pool_idx]
            & ~self.host_cache.valid[cache_pool_idx]
        )
        if invalid_mask.any():
            invalid = cache_pool_idx[invalid_mask].tolist()
            raise RuntimeError(
                "Routed-expert data is unavailable for KV slots "
                f"{invalid[:16]}; refusing to return stale HostCache rows."
            )
        return self.host_cache.buffer[cache_pool_idx]

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        """If no_copy_to_cpu is True, return a TopkCaptureOutput holding GPU tensors so
        the overlap thread can do non-blocking D2H + finalize itself. Otherwise sync
        D2H inline and return None (legacy non-overlap path).
        """
        slice_gpu = self._get_local_slice(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        if no_copy_to_cpu:
            # Clone before the next overlapping forward reuses these buffers.
            return TopkCaptureOutput(
                out_cache_loc=forward_batch.out_cache_loc.clone(),
                topk=slice_gpu.clone(),
                host_cache=self.host_cache,
            )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.store(out_cache_loc_cpu, slice_gpu)
        return None
