import logging
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
from sglang.srt.layers.topk_capturer_base import (
    BaseDeviceCache,
    BaseHostCache,
    TopkCaptureOutput,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class RoutedExpertsCapturer:
    """Capturer for routed experts with host buffer.

    Unlike IndexerTopkCapturer (per-rank-local capture), this captures into a
    global device buffer indexed by DP rank, so _get_local_range computes a
    DP-rank-aware slice when DP attention is enabled.
    """

    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["RoutedExpertsCapturer"]:
        if not enable:
            return None
        return RoutedExpertsCapturer(
            model_config,
            num_tokens=num_tokens,
            max_running_requests=max_running_requests,
            num_fused_shared_experts=num_fused_shared_experts,
            device=device,
        )

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok

        server_args = get_global_server_args()
        max_batch_size = max(
            server_args.chunked_prefill_size * server_args.dp_size,
            max_running_requests,
        )

        # Device cache holds the full topk_ids including any fused shared experts
        # columns. Host cache (and the user-facing return) drops the shared columns
        # via the [:num_experts_per_tok] truncation in on_forward_end.
        self.host_cache = BaseHostCache(
            num_tokens=num_tokens,
            num_layers=self.num_hidden_layers,
            topk_size=self.num_experts_per_tok,
            name="routed_experts",
        )
        self.device_cache = BaseDeviceCache(
            max_batch_size=max_batch_size,
            num_layers=self.num_hidden_layers,
            topk_size=self.num_experts_per_tok + self.num_fused_shared_experts,
            device=device,
            name="routed_experts",
        )

    def _get_local_range(self, forward_batch, can_run_graph, cuda_graph_batch):
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
            return local_start_pos, local_start_pos + local_num_tokens
        else:
            return 0, forward_batch.out_cache_loc.shape[0]

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture(layer_id, topk_ids)

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        cache_pool_idx = req_to_token_pool.req_to_token[req_pool_idx][
            : seqlen - 1
        ].cpu()
        return self.host_cache.buffer[cache_pool_idx]

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        local_start_pos, local_end_pos = self._get_local_range(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        slice_gpu = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ]
        if no_copy_to_cpu:
            return TopkCaptureOutput(
                out_cache_loc=forward_batch.out_cache_loc,
                topk=slice_gpu,
                host_cache=self.host_cache,
            )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = slice_gpu.cpu()
        return None


_global_expert_capturer: Optional[RoutedExpertsCapturer] = None


def get_global_experts_capturer() -> Optional[RoutedExpertsCapturer]:
    return _global_expert_capturer


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):
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
