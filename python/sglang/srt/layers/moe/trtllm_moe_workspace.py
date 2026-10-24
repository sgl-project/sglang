"""Caller-owned scratch for sequential FlashInfer TRT-LLM MoE launches."""

import logging

import torch

from sglang.srt.runtime_context import (
    get_buffer,
    get_context,
    get_parallel,
    get_schedule,
    max_prefill_buffer_tokens,
)

logger = logging.getLogger(__name__)


class TrtllmMoeWorkspace:
    """Cache sizing queries and share scratch across layers on the same stream.

    Size queries and buffer allocation may run during capture. Keep every buffer
    alive without resizing or replacing it, so captured graphs retain stable
    addresses.
    """

    def __init__(self, max_rows: int):
        self.max_rows = max_rows
        self.sizes = {}
        self.buffers = {}

    def get(self, quant_mode: str, kwargs: dict):
        x = kwargs["hidden_states"]
        if (
            not kwargs.get("do_finalize", True)
            or x.shape[0] == 0
            or max(x.shape[0], kwargs.get("tune_max_num_tokens", 8192)) > self.max_rows
        ):
            return None

        from flashinfer import fused_moe

        sizer = getattr(fused_moe, "trtllm_fused_moe_workspace_size", None)
        if sizer is None:
            return None

        options = {
            "quant_mode": quant_mode,
            "local_num_experts": kwargs["local_num_experts"],
            "num_fused_shared_experts": kwargs.get("num_fused_shared_experts") or 0,
            "use_shuffled_weight": kwargs.get("use_shuffled_weight", True),
            "weight_layout": kwargs.get("weight_layout"),
            "use_routing_scales_on_input": kwargs.get(
                "use_routing_scales_on_input", False
            ),
            "use_per_token_scale": kwargs.get("per_token_scale") is not None,
            "valid_hidden_size": kwargs.get("valid_hidden_size"),
            "valid_intermediate_size": kwargs.get("valid_intermediate_size"),
            "device": x.device,
        }
        if kwargs.get("activation_type") is not None:
            options["activation_type"] = int(kwargs["activation_type"])
        problem = (
            self.max_rows,
            x.shape[1] * (2 if quant_mode == "nvfp4" else 1),
            kwargs["intermediate_size"],
            kwargs["num_experts"],
            kwargs["top_k"],
        )
        size_key = (problem, tuple(options.items()))
        if size_key not in self.sizes:
            try:
                self.sizes[size_key] = int(sizer(*problem, **options))
            except (RuntimeError, ValueError) as exc:
                logger.warning("TRT-LLM MoE workspace sizing failed: %s", exc)
                self.sizes[size_key] = 0
        nbytes = self.sizes[size_key]
        if nbytes <= 0:
            return None
        stream = torch.cuda.current_stream(x.device).cuda_stream
        buffer_key = (x.device, stream, nbytes)
        if buffer_key not in self.buffers:
            self.buffers[buffer_key] = torch.empty(
                nbytes, dtype=torch.uint8, device=x.device
            )
            logger.info(
                "Reserved %.2f GiB TRT-LLM MoE workspace for up to %d tokens",
                nbytes / (1 << 30),
                self.max_rows,
            )
        return self.buffers[buffer_key]


def _make_workspace():
    schedule = get_schedule()
    rows = (
        max(
            max_prefill_buffer_tokens() or schedule.max_prefill_tokens,
            schedule.max_running_requests or 0,
        )
        * get_parallel().attn_dp_size
    )
    # Include the power-of-two autotuning bucket, even for a non-power-of-two
    # prefill chunk. Larger unchunked/speculative batches use per-call scratch.
    return TrtllmMoeWorkspace(1 << (max(1, rows) - 1).bit_length())


def get_trtllm_moe_workspace(quant_mode: str, **kwargs) -> torch.Tensor | None:
    """Return reusable scratch, or None when per-call allocation is needed.

    Callers should omit workspace_buffer when this returns None, preserving
    compatibility with FlashInfer versions without caller-owned scratch.
    """
    if not kwargs.get(
        "do_finalize", True
    ) or not get_context().is_config_namespace_published("schedule"):
        return None
    workspace = get_buffer("trtllm_moe_workspace", _make_workspace)
    return workspace.get(quant_mode, kwargs)
