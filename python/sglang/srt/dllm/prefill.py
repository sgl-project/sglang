"""Eligibility for a prompt-only forward that produces only KV-cache state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


def supports_kv_only_prefill(model_runner: ModelRunner) -> bool:
    """Use the same launch-time gate for eager execution and graph capture."""
    if not envs.SGLANG_ENABLE_DLLM_KV_ONLY_PREFILL.get():
        return False
    args = model_runner.server_args
    model = model_runner.model
    return (
        getattr(model, "supports_dllm_kv_only_prefill", False)
        and model_runner.device == "cuda"
        and torch.version.hip is None
        and model_runner.model_config.dtype == torch.bfloat16
        and model.quant_config is None
        and not model_runner.is_multimodal
        and not model_runner.is_draft_worker
        and args.dllm_algorithm == "LowConfidence"
        and args.dllm_fdfo
        and args.tp_size == args.pp_size == args.dp_size == args.attn_cp_size == 1
        and args.attention_backend == "flashinfer"
        and args.moe_runner_backend == "triton"
        and not args.enable_lora
        and not args.enable_torch_compile
        and not args.enable_two_batch_overlap
        and not args.enable_return_hidden_states
        and args.return_hidden_states_mode is None
        and not args.enable_return_aux_hidden_states
        and not args.enable_return_routed_experts
        and not args.enable_return_indexer_topk
        and args.speculative_algorithm is None
    )


def can_use_kv_only_prefill(
    model_runner: ModelRunner, forward_batch: ForwardBatch, batch: ScheduleBatch
) -> bool:
    """Call only after the algorithm has identified complete prompt blocks."""
    return (
        supports_kv_only_prefill(model_runner)
        and forward_batch.forward_mode.is_dllm_extend()
        and not forward_batch.return_logprob
        and forward_batch.capture_hidden_mode in (None, CaptureHiddenMode.NULL)
        and not forward_batch.return_hidden_states_before_norm
        and not forward_batch.return_pooled_hidden_states
        and not forward_batch.is_prefill_only
        and forward_batch.beam is None
        and not any(
            req.return_routed_experts or req.return_indexer_topk for req in batch.reqs
        )
    )
