import logging
import os
import threading
import time
from abc import ABC
from queue import Queue
from typing import Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    def __init__(
        self, model_config: ModelConfig, max_running_requests: int, device: str
    ) -> None:
        self.buffer = torch.zeros(
            (
                max(
                    get_global_server_args().chunked_prefill_size, max_running_requests
                ),
                model_config.hf_text_config.num_hidden_layers,
                model_config.hf_text_config.num_experts_per_tok,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
    ) -> None:
        self.num_tokens = num_tokens
        self.buffer = torch.zeros(
            (
                num_tokens,
                model_config.hf_text_config.num_hidden_layers,
                model_config.hf_text_config.num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def set_experts_buffer(self, layer_id: int, loc: torch.Tensor, top_k: torch.Tensor):
        self.buffer[layer_id, loc, :] = top_k.to(device="cpu", non_blocking=True)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
        use_storage_backup: bool = False,
        storage_backup_path: str = "",
        req_to_token_pool: Optional[ReqToTokenPool] = None,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                device=device,
                use_storage_backup=use_storage_backup,
                storage_backup_path=storage_backup_path,
                req_to_token_pool=req_to_token_pool,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def sync_fwd_experts_buffer_DtoH(self, batch: int, loc: torch.Tensor):
        raise NotImplementedError

    def sync_fwd_experts_buffer_host_to_storage(
        self,
        req_pool_idx: int,
        seqlen: int,
        rid: str,
    ):
        raise NotImplementedError

    def wait_for_all_inflight_backups(self):
        pass

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
        use_storage_backup: bool = False,
        storage_backup_path: str = "",
        req_to_token_pool: Optional[ReqToTokenPool] = None,
    ):

        self.host_cache = _RoutedExpertsHostCache(model_config, num_tokens)

        self.device_cache = _RoutedExpertsDeviceCache(
            model_config, max_running_requests, device
        )

        self.use_storage_backup = use_storage_backup
        self.storage_backup_path = storage_backup_path
        self.req_to_token_pool = req_to_token_pool
        # write host storage to disk async
        if self.use_storage_backup:
            os.makedirs(self.storage_backup_path, exist_ok=True)
            self.stop_event = threading.Event()
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.sync_to_disk_thread = threading.Thread(
                target=self.sync_to_disk_thread_func, daemon=True
            )
            self.backup_queue = Queue()
            self.sync_to_disk_queue = Queue()
            self.lock = threading.Lock()
            self.inflight_count = 0
            self.backup_thread.start()
            self.sync_to_disk_thread.start()

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def sync_fwd_experts_buffer_DtoH(
        self, device_loc: torch.Tensor, cpu_loc: torch.Tensor
    ):
        if self.use_storage_backup:
            self.wait_for_all_inflight_backups()

        batch = device_loc.shape[0]

        self.host_cache.buffer[cpu_loc] = self.device_cache.buffer[:batch].to(
            device="cpu", non_blocking=True
        )

    # backup to storage:
    # scheduler -> backup_thread -> sync_to_disk_thread
    def sync_fwd_experts_buffer_host_to_storage(
        self,
        req_pool_idx: int,
        seqlen: int,
        rid: str,
    ):
        cache_idx = (
            self.req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1]
            .cpu()
            .clone()
        )
        self.backup_queue.put((rid, cache_idx))
        with self.lock:
            self.inflight_count += 1

    def wait_for_all_inflight_backups(self):
        while True:
            with self.lock:
                if self.inflight_count == 0:
                    break
            time.sleep(0.001)

    def backup_thread_func(self):
        while True:
            rid, cache_idx = self.backup_queue.get()
            routed_experts = self.get_host_cache().buffer[cache_idx].clone()
            self.sync_to_disk_queue.put((rid, routed_experts))
            with self.lock:
                self.inflight_count -= 1

    # TODO: use object store instead of local disk
    # Notice: this may create a lot of small files on disk. Make sure the storage can handle it and
    # remove the files when no longer needed.
    def sync_to_disk_thread_func(self):
        while True:
            rid, routed_experts = self.sync_to_disk_queue.get()
            torch.save(
                routed_experts,
                os.path.join(self.storage_backup_path, f"{rid}_routed_experts.pt"),
            )

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
        )
        return self.get_host_cache().buffer[cache_pool_idx]

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def sync_fwd_experts_buffer_DtoH(
        self, device_loc: torch.Tensor, cpu_loc: torch.Tensor
    ):
        pass

    def get_host_cache(self):
        pass

    def get_device_cache(self):
        pass


_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer
