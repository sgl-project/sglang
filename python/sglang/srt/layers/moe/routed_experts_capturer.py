import dataclasses
import logging
from abc import ABC
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_dp_local_info,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


@dataclasses.dataclass
class RoutedExpertsOutput:
    """Holds GPU tensors captured during forward for overlap scheduling.
    Call copy_to_cpu() inside forward stream (before copy_done.record()),
    then finalize() after copy_done.synchronize().
    """

    out_cache_loc: torch.Tensor
    routed_experts: torch.Tensor
    host_cache: "_RoutedExpertsHostCache"
    routed_expert_weights: Optional[torch.Tensor] = None

    def copy_to_cpu(self):
        self.out_cache_loc = self.out_cache_loc.to("cpu", non_blocking=True)
        self.routed_experts = self.routed_experts.to("cpu", non_blocking=True)
        if self.routed_expert_weights is not None:
            self.routed_expert_weights = self.routed_expert_weights.to(
                "cpu", non_blocking=True
            )

    def finalize(self):
        self.host_cache.buffer[self.out_cache_loc] = self.routed_experts
        if (
            self.routed_expert_weights is not None
            and self.host_cache.weights_buffer is not None
        ):
            self.host_cache.weights_buffer[self.out_cache_loc] = (
                self.routed_expert_weights
            )


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
        enable_weights: bool = False,
    ) -> None:
        # Multiply by ``dp_size`` so that the buffer can hold the full
        # concatenated batch across DP attention ranks. ``_get_local_range``
        # indexes into ``[attention_dp_rank * cuda_graph_batch, ...)`` so the
        # highest touched index scales with ``dp_size``.
        dp_size = get_global_server_args().dp_size
        buffer_first_dim = max(
            get_global_server_args().chunked_prefill_size * dp_size,
            max_running_requests * dp_size,
        )
        self.buffer = torch.zeros(
            (
                buffer_first_dim,
                num_hidden_layers,
                num_experts_per_tok + num_fused_shared_experts,
            ),
            dtype=torch.int32,
            device=device,
        )
        # Optional parallel float32 buffer for routing softmax weights.
        # Shape mirrors ``self.buffer`` so the same indexing logic applies.
        self.weights_buffer: Optional[torch.Tensor] = None
        if enable_weights:
            self.weights_buffer = torch.zeros(
                (
                    buffer_first_dim,
                    num_hidden_layers,
                    num_experts_per_tok + num_fused_shared_experts,
                ),
                dtype=torch.float32,
                device=device,
            )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        size = get_tensor_size_bytes(self.buffer)
        if self.weights_buffer is not None:
            size += get_tensor_size_bytes(self.weights_buffer)
        return size

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def capture_fwd_routed_expert_weights(
        self, layer_id: int, topk_weights: torch.Tensor
    ):
        """Store routing weights (softmax probabilities) for the given layer.

        No-op when the optional ``weights_buffer`` was not allocated.
        """
        if self.weights_buffer is None:
            return
        assert layer_id is not None, (
            "capturing routing expert weights but get layer_id None"
        )
        batch, k = topk_weights.shape
        cols = min(k, self.weights_buffer.shape[2])
        self.weights_buffer[:batch, layer_id, :cols] = topk_weights[:, :cols]

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        enable_weights: bool = False,
    ) -> None:
        self.num_tokens = num_tokens
        self.buffer = torch.zeros(
            (
                num_tokens,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.weights_buffer: Optional[torch.Tensor] = None
        if enable_weights:
            self.weights_buffer = torch.zeros(
                (
                    num_tokens,
                    num_hidden_layers,
                    num_experts_per_tok,
                ),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        size = get_tensor_size_bytes(self.buffer)
        if self.weights_buffer is not None:
            size += get_tensor_size_bytes(self.weights_buffer)
        return size

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
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
        enable_weights: bool = False,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                device=device,
                enable_weights=enable_weights,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        raise NotImplementedError

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError

    def capture_weights(self, layer_id: int, topk_weights: torch.Tensor):
        """Capture routing weights for the current layer.

        No-op unless the capturer was created with ``enable_weights=True``.
        """
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def get_routed_expert_weights(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        """Return host-cached routing weights for a request, or ``None``."""
        raise NotImplementedError

    def on_forward_end(
        self, forward_batch, can_run_graph, cuda_graph_batch, no_copy_to_cpu=False
    ) -> Optional[RoutedExpertsOutput]:
        raise NotImplementedError

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
        num_fused_shared_experts: int,
        device: str,
        enable_weights: bool = False,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.enable_weights = enable_weights

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            enable_weights=enable_weights,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
            enable_weights=enable_weights,
        )

    def _get_local_range(self, forward_batch, can_run_graph, cuda_graph_batch):
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
            return local_start_pos, local_start_pos + local_num_tokens
        else:
            return 0, forward_batch.out_cache_loc.shape[0]

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        local_start_pos, local_end_pos = self._get_local_range(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ].cpu()
        if (
            self.device_cache.weights_buffer is not None
            and self.host_cache.weights_buffer is not None
        ):
            self.host_cache.weights_buffer[out_cache_loc_cpu] = (
                self.device_cache.weights_buffer[
                    local_start_pos:local_end_pos, :, : self.num_experts_per_tok
                ].cpu()
            )

    def _prepare_routed_experts_output(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ) -> RoutedExpertsOutput:
        local_start_pos, local_end_pos = self._get_local_range(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        weights = None
        if self.device_cache.weights_buffer is not None:
            weights = self.device_cache.weights_buffer[
                local_start_pos:local_end_pos, :, : self.num_experts_per_tok
            ]
        return RoutedExpertsOutput(
            out_cache_loc=forward_batch.out_cache_loc,
            routed_experts=self.device_cache.buffer[
                local_start_pos:local_end_pos, :, : self.num_experts_per_tok
            ],
            host_cache=self.host_cache,
            routed_expert_weights=weights,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def capture_weights(self, layer_id: int, topk_weights: torch.Tensor):
        self.device_cache.capture_fwd_routed_expert_weights(layer_id, topk_weights)

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

    def get_routed_expert_weights(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        host = self.get_host_cache()
        if host.weights_buffer is None:
            return None
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
        )
        return host.weights_buffer[cache_pool_idx]

    def on_forward_end(
        self, forward_batch, can_run_graph, cuda_graph_batch, no_copy_to_cpu=False
    ) -> Optional[RoutedExpertsOutput]:
        if no_copy_to_cpu:
            return self._prepare_routed_experts_output(
                forward_batch=forward_batch,
                can_run_graph=can_run_graph,
                cuda_graph_batch=cuda_graph_batch,
            )
        else:
            self._sync_fwd_experts_buffer_DtoH(
                forward_batch=forward_batch,
                can_run_graph=can_run_graph,
                cuda_graph_batch=cuda_graph_batch,
            )
            return None

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def capture_weights(self, layer_id: int, topk_weights: torch.Tensor):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def get_routed_expert_weights(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        return None

    def on_forward_end(
        self, forward_batch, can_run_graph, cuda_graph_batch, no_copy_to_cpu=False
    ) -> Optional[RoutedExpertsOutput]:
        return None

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


def extract_routed_experts_from_meta_info(data):
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32
    )
    return routed_experts
