"""Opt-in deterministic NPU HCCL buffer sizing; importing never initializes a device.

References:
- CANN 9.1 (sglang NPU default) HCCL_BUFFSIZE: default 200 MB when unset, must
  be positive, and every communication domain allocates send + receive buffers
  (~2x the value).
  https://www.hiascend.com/document/detail/en/canncommercial/910/maintenref/envvar/envref_07_0080.html
- torch_npu reads HCCL_BUFFSIZE in OptionsManager.cpp (default 200, > 0) and
  per-group hccl_buffer_size pg options override it.
- DeepEP Ascend buffer formulas (sglang expert_parallelism docs); the fixed
  102 MB + 4 MB window offsets and the 2x send/receive factor are identical
  across A2 (910B), A3 and Ascend 950 in the sgl-kernel-npu tilings:
    combine streaming on:  2 * (106MB + PER_ROUND_TOKENS * 3 * hidden * topk) + PADDING
    combine streaming off: 2 * (106MB + TOTAL_SEQ_LEN * 2 * hidden * topk) + PADDING
  where PER_ROUND_TOKENS is DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS (default
  8192), combine streaming is DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ (default
  off), and dispatch/combine rounds carry three resp. two hidden-sized tensors
  per selected expert.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
MIB = 1 << 20
# DeepEP Ascend fixed per-direction overhead: 102 MB framework + 4 MB metadata,
# and the documented PADDING_BUFFSIZE recommendation is 20 MB.
DEEPEP_FIXED_OVERHEAD_MB = 106
DEEPEP_PADDING_MB = 20
DEEPEP_MIN_BUFFSIZE_MB = 2 * DEEPEP_FIXED_OVERHEAD_MB + DEEPEP_PADDING_MB
# Default of DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS; valid range [32, 8192].
DEEPEP_PER_ROUND_TOKENS = 8192
_TRUTHY = ("1", "true", "yes")


@dataclass(frozen=True)
class NpuTuningContext:
    """Model and parallelism facts that determine representative HCCL sizes."""

    hidden_size: int
    num_experts_per_tok: int | None
    tp_size: int
    ep_size: int
    chunked_prefill_size: int
    max_prefill_tokens: int
    dtype: str


def representative_prefill_rows(context: NpuTuningContext) -> int:
    if context.max_prefill_tokens <= 0:
        raise ValueError("max_prefill_tokens must be positive for NPU autotuning")
    if context.chunked_prefill_size > 0:
        return min(context.chunked_prefill_size, context.max_prefill_tokens)
    return context.max_prefill_tokens


def _quantize_mb(raw_mb, quantum_mb, min_mb, max_mb):
    if min(quantum_mb, min_mb) <= 0 or max_mb < min_mb:
        raise ValueError("rounding quantum and bounds must be positive")
    return min(max_mb, max(min_mb, math.ceil(raw_mb / quantum_mb) * quantum_mb))


def hccl_buffer_mb(
    token_rows,
    hidden_size,
    element_size,
    headroom=1.25,
    quantum_mb=32,
    min_mb=64,
    max_mb=1024,
):
    """Size one collective payload: rows * hidden * element bytes plus headroom."""
    if min(token_rows, hidden_size, element_size) <= 0:
        raise ValueError("HCCL dimensions must be positive")
    if not math.isfinite(headroom) or headroom < 1:
        raise ValueError("HCCL headroom must be a multiplier >= 1")
    raw = token_rows * hidden_size * element_size * headroom / MIB
    return _quantize_mb(raw, quantum_mb, min_mb, max_mb)


def deepep_normal_buffer_mb(
    token_rows,
    hidden_size,
    topk,
    per_round_tokens=DEEPEP_PER_ROUND_TOKENS,
    ant_moving=False,
    combine_long_seq=False,
    quantum_mb=32,
    min_mb=DEEPEP_MIN_BUFFSIZE_MB,
    max_mb=4096,
):
    """Size the DeepEP Ascend normal-mode buffer from the documented formulas.

    With combine streaming on, rounds carry three hidden-sized tensors per
    expert (dispatch plus double-buffered combine); with it off (default), the
    full sequence must fit with two tensors per expert. The 2x covers
    send + receive.
    """
    if min(token_rows, hidden_size, topk, per_round_tokens) <= 0:
        raise ValueError("DeepEP dimensions must be positive")
    per_round = min(token_rows, per_round_tokens) * 3 * hidden_size * topk
    full_seq = token_rows * 2 * hidden_size * topk
    if combine_long_seq:
        payload = per_round
    elif ant_moving:
        payload = max(full_seq, per_round)
    else:
        payload = full_seq
    raw = 2 * (DEEPEP_FIXED_OVERHEAD_MB + payload / MIB) + DEEPEP_PADDING_MB
    return _quantize_mb(raw, quantum_mb, min_mb, max_mb)


def context_from_server_args(server_args):
    from sglang.srt.arg_groups.model_override_base import model_config_of

    model = model_config_of(server_args)
    hf = model.hf_text_config
    return NpuTuningContext(
        hf.hidden_size,
        getattr(hf, "num_experts_per_tok", None),
        server_args.tp_size,
        server_args.ep_size,
        server_args.chunked_prefill_size,
        server_args.max_prefill_tokens,
        str(model.dtype),
    )


def _deepep_round_config():
    """Read the DeepEP ant-moving env pair and the combine-streaming flag."""
    raw = os.environ.get("DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS")
    try:
        prt = min(DEEPEP_PER_ROUND_TOKENS, max(32, int(raw))) if raw else DEEPEP_PER_ROUND_TOKENS
    except ValueError:
        prt, raw = DEEPEP_PER_ROUND_TOKENS, None
    combine = os.environ.get("DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ", "0").lower() in _TRUTHY
    return prt, raw is not None, combine


def _decide(name, automatic, reason, dry):
    """Apply explicit-wins precedence, log the decision, set unless dry-run."""
    if name in os.environ:
        source, value = "explicit", os.environ[name]
    else:
        source, value = "deterministic-auto", automatic
    logger.info(
        "NPU autotune: %s=%s source=%s dry_run=%s %s",
        name,
        value,
        source,
        dry,
        reason,
    )
    if not dry and source == "deterministic-auto" and value is not None:
        os.environ[name] = str(value)


def initialize(server_args):
    """Resolve automatic HCCL sizing before scheduler/HCCL initialization."""
    auto_hccl = envs.SGLANG_NPU_AUTO_HCCL_BUFFSIZE.get()
    auto_deepep = envs.SGLANG_NPU_AUTO_DEEPEP_HCCL_BUFFSIZE.get()
    if not (auto_hccl or auto_deepep):
        return
    context = context_from_server_args(server_args)
    rows = representative_prefill_rows(context)
    dry = envs.SGLANG_NPU_AUTOTUNE_DRY_RUN.get()
    if auto_hccl:
        element = 4 if context.dtype in ("float32", "torch.float32") else 2
        _decide(
            "HCCL_BUFFSIZE",
            hccl_buffer_mb(
                rows,
                context.hidden_size,
                element,
                envs.SGLANG_NPU_HCCL_HEADROOM.get(),
                envs.SGLANG_NPU_HCCL_QUANTUM_MB.get(),
                envs.SGLANG_NPU_HCCL_MIN_MB.get(),
                envs.SGLANG_NPU_HCCL_MAX_MB.get(),
            ),
            f"collective_bytes={rows * context.hidden_size * element} "
            f"tp={context.tp_size} dtype={context.dtype}; HCCL default if unset",
            dry,
        )
    if auto_deepep:
        if context.num_experts_per_tok:
            prt, ant_moving, combine_long = _deepep_round_config()
            _decide(
                "DEEPEP_HCCL_BUFFSIZE",
                deepep_normal_buffer_mb(
                    rows,
                    context.hidden_size,
                    context.num_experts_per_tok,
                    prt,
                    ant_moving,
                    combine_long,
                    envs.SGLANG_NPU_DEEPEP_HCCL_QUANTUM_MB.get(),
                    envs.SGLANG_NPU_DEEPEP_HCCL_MIN_MB.get(),
                    envs.SGLANG_NPU_DEEPEP_HCCL_MAX_MB.get(),
                ),
                f"rows={rows} hidden={context.hidden_size} "
                f"topk={context.num_experts_per_tok} per_round_tokens={prt} "
                f"combine_long_seq={combine_long} ep={context.ep_size}; "
                "documented DeepEP normal-mode formulas",
                dry,
            )
        else:
            logger.info(
                "NPU autotune: DEEPEP_HCCL_BUFFSIZE=None source=default "
                "dry_run=%s model has no num_experts_per_tok; "
                "DeepEP sizing not applicable",
                dry,
            )
