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


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
        enable_capture_dsa_topk_indices: bool = False,
        num_dsa_topk_indices: int = 2048,
    ) -> None:
        self.buffer = torch.zeros(
            (
                max(
                    get_global_server_args().chunked_prefill_size
                    * get_global_server_args().dp_size,
                    max_running_requests,
                ),
                num_hidden_layers,
                num_experts_per_tok + num_fused_shared_experts,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.dsa_topk_indices_buffer = None
        if enable_capture_dsa_topk_indices:
            self.dsa_topk_indices_buffer = torch.zeros(
                (
                    max(
                        get_global_server_args().chunked_prefill_size
                        * get_global_server_args().dp_size,
                        max_running_requests,
                    ),
                    num_hidden_layers,
                    num_dsa_topk_indices,
                ),
                dtype=torch.int32,
                device=device,
            )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def get_dsa_topk_indices_buffer_size_bytes(self):
        assert hasattr(self, "dsa_topk_indices_buffer")
        return get_tensor_size_bytes(self.dsa_topk_indices_buffer)

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def capture_fwd_dsa_topk_indices(
        self, layer_id: int, dsa_topk_indices: torch.Tensor
    ):
        assert layer_id is not None, "capturing dsa topk indices but get layer_id None"
        batch, _ = dsa_topk_indices.shape
        self.dsa_topk_indices_buffer[:batch, layer_id, :] = dsa_topk_indices

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )
        if self.dsa_topk_indices_buffer is not None:
            dsa_topk_indices_size_MB = (
                self.get_dsa_topk_indices_buffer_size_bytes() / _MB
            )
            logger.info(
                f"DSA topk indices device buffer allocated. #shape: {tuple(self.dsa_topk_indices_buffer.shape)}, size: {dsa_topk_indices_size_MB:.2f} MB"
            )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        enable_capture_dsa_topk_indices: bool = False,
        num_dsa_topk_indices: int = 2048,
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
        self.dsa_topk_indices_buffer = None
        if enable_capture_dsa_topk_indices:
            self.dsa_topk_indices_buffer = torch.zeros(
                (
                    num_tokens,
                    num_hidden_layers,
                    num_dsa_topk_indices,
                ),
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)

    def get_dsa_topk_indices_buffer_size_bytes(self):
        assert hasattr(self, "dsa_topk_indices_buffer")
        return get_tensor_size_bytes(self.dsa_topk_indices_buffer)

    def set_experts_buffer(self, layer_id: int, loc: torch.Tensor, top_k: torch.Tensor):
        self.buffer[layer_id, loc, :] = top_k.to(device="cpu", non_blocking=True)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )
        if self.dsa_topk_indices_buffer is not None:
            dsa_topk_indices_size_GB = (
                self.get_dsa_topk_indices_buffer_size_bytes() / _GB
            )
            logger.info(
                f"DSA topk indices host buffer allocated. #tokens: {self.num_tokens}, size: {dsa_topk_indices_size_GB:.2f} GB"
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
        enable_capture_dsa_topk_indices: bool = False,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                device=device,
                enable_capture_dsa_topk_indices=enable_capture_dsa_topk_indices,
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

    def capture_dsa_topk_indices(self, layer_id: int, dsa_topk_indices: torch.Tensor):
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def get_dsa_topk_indices(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
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
        enable_capture_dsa_topk_indices: bool = False,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok

        self.enable_capture_dsa_topk_indices = enable_capture_dsa_topk_indices
        if enable_capture_dsa_topk_indices:
            self.num_dsa_topk_indices = model_config.hf_text_config.index_topk
        else:
            self.num_dsa_topk_indices = None

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            enable_capture_dsa_topk_indices=self.enable_capture_dsa_topk_indices,
            num_dsa_topk_indices=self.num_dsa_topk_indices,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
            enable_capture_dsa_topk_indices=self.enable_capture_dsa_topk_indices,
            num_dsa_topk_indices=self.num_dsa_topk_indices,
        )

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            # handle with cuda graph padding
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = forward_batch.out_cache_loc.shape[0]

        # FIXME: sync explicitly here, overlap scheduler breaks here.
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ].cpu()

        if self.enable_capture_dsa_topk_indices:
            self.host_cache.dsa_topk_indices_buffer[out_cache_loc_cpu] = (
                self.device_cache.dsa_topk_indices_buffer[
                    local_start_pos:local_end_pos, :, : self.num_dsa_topk_indices
                ].cpu()
            )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)

    def capture_dsa_topk_indices(self, layer_id: int, dsa_topk_indices: torch.Tensor):
        if self.enable_capture_dsa_topk_indices:
            self.device_cache.capture_fwd_dsa_topk_indices(layer_id, dsa_topk_indices)

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

    def get_dsa_topk_indices(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        if self.enable_capture_dsa_topk_indices:
            cache_pool_idx = (
                req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
            )
            return self.get_host_cache().dsa_topk_indices_buffer[cache_pool_idx]

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        self._sync_fwd_experts_buffer_DtoH(
            forward_batch=forward_batch,
            can_run_graph=can_run_graph,
            cuda_graph_batch=cuda_graph_batch,
        )

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

    def capture_dsa_topk_indices(self, layer_id: int, dsa_topk_indices: torch.Tensor):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def get_dsa_topk_indices(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
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


def extract_routed_experts_from_meta_info(data):
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32
    )
    return routed_experts
