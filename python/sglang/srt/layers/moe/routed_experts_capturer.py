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
from sglang.srt.layers.topk_capturer_base import BaseTopkCapturer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args


class RoutedExpertsCapturer(BaseTopkCapturer):
    """Capturer for routed experts with host buffer.

    Routed experts share a global device buffer across DP ranks (indexed by
    dp_rank), so `_get_local_slice` overrides the default to apply DP-rank-aware
    slicing. The device cache also holds extra columns for any fused shared
    experts; the host cache and user-facing return drop them via the
    [:topk_size] truncation.
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
        topk_size = model_config.hf_text_config.num_experts_per_tok
        num_layers = model_config.hf_text_config.num_hidden_layers

        server_args = get_global_server_args()
        # FIXME: spec decoding is not accounted for here. The device buffer can
        # overflow when max_running_requests * num_verify_tokens exceeds
        # chunked_prefill_size * dp_size.
        max_batch_size = max(
            server_args.chunked_prefill_size * server_args.dp_size,
            max_running_requests,
        )

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=topk_size,
            device=device,
            name="routed_experts",
            device_topk_size=topk_size + num_fused_shared_experts,
        )

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]


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
