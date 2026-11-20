"""Default deterministic NPU HCCL buffer sizing; importing never initializes a device.

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
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)
MIB = 1 << 20
# DeepEP Ascend fixed per-direction overhead: 102 MB framework + 4 MB metadata,
# and the documented PADDING_BUFFSIZE recommendation is 20 MB.
DEEPEP_FIXED_OVERHEAD_MB = 106
DEEPEP_PADDING_MB = 20
DEEPEP_MIN_BUFFSIZE_MB = 2 * DEEPEP_FIXED_OVERHEAD_MB + DEEPEP_PADDING_MB
# Default of DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS; valid range [32, 8192].
DEEPEP_PER_ROUND_TOKENS = 8192
DEEPEP_MAX_BUFFSIZE_MB = 4096


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
    max_mb=DEEPEP_MAX_BUFFSIZE_MB,
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
    if min(quantum_mb, min_mb) <= 0 or max_mb < min_mb:
        raise ValueError("rounding quantum and bounds must be positive")
    required = math.ceil(max(raw, min_mb) / quantum_mb) * quantum_mb
    details = (
        f"rows={token_rows} hidden={hidden_size} topk={topk} "
        f"per_round_tokens={per_round_tokens} ant_moving={ant_moving} "
        f"combine_long_seq={combine_long_seq} raw_MiB={raw:g} "
        f"required_MiB={required} cap_MiB={max_mb}"
    )
    if required > max_mb:
        raise ValueError(
            f"Automatic DeepEP HCCL sizing exceeds its limit: {details}. "
            "Set DEEPEP_HCCL_BUFFSIZE explicitly or reduce the workload."
        )
    logger.info("NPU autotune: DeepEP normal-mode requirement %s", details)
    return required


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
    tokens_name = "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS"
    rounds_name = "DEEPEP_NORMAL_LONG_SEQ_ROUND"
    tokens, rounds = os.environ.get(tokens_name), os.environ.get(rounds_name)
    if (tokens is None) != (rounds is None):
        raise ValueError(f"{tokens_name} and {rounds_name} must be set together")
    if tokens is not None:
        if not all(
            re.fullmatch(r"[ \t\n\r\v\f]*[+-]?[0-9]+", v) for v in (tokens, rounds)
        ):
            raise ValueError(f"{tokens_name} and {rounds_name} must be integers")
        per_round, round_count = int(tokens), int(rounds)
        if not (32 <= per_round <= 8192 and 1 <= round_count <= 256):
            raise ValueError(
                f"{tokens_name} must be in [32, 8192] and {rounds_name} in [1, 256]"
            )
        if per_round * round_count > 131072:
            raise ValueError(f"{tokens_name} * {rounds_name} must not exceed 131072")
    else:
        per_round = DEEPEP_PER_ROUND_TOKENS
    combine_name = "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ"
    combine = os.environ.get(combine_name, "0")
    if not re.fullmatch(r"[0-9]+", combine) or int(combine) > 2147483647:
        raise ValueError(f"{combine_name} must be a nonnegative 32-bit integer")
    return per_round, tokens is not None, bool(int(combine))


def initialize(server_args):
    """Resolve automatic HCCL sizing before scheduler/HCCL initialization."""
    # Snapshot overrides before the general heuristic fills HCCL_BUFFSIZE.
    hccl = os.environ.get("HCCL_BUFFSIZE")
    deepep = os.environ.get("DEEPEP_HCCL_BUFFSIZE")
    if hccl is not None:
        logger.info("NPU autotune: HCCL_BUFFSIZE=%s MiB source=explicit", hccl)
    if deepep is not None:
        logger.info("NPU autotune: DEEPEP_HCCL_BUFFSIZE=%s MiB source=explicit", deepep)
    elif hccl is not None:
        logger.info(
            "NPU autotune: DEEPEP_HCCL_BUFFSIZE=%s MiB source=inherited HCCL_BUFFSIZE",
            hccl,
        )
    if hccl is not None:
        return

    context = context_from_server_args(server_args)
    rows = representative_prefill_rows(context)
    element = 4 if context.dtype in ("float32", "torch.float32") else 2
    proposed = {"HCCL_BUFFSIZE": hccl_buffer_mb(rows, context.hidden_size, element)}
    if deepep is None:
        backend, mode = server_args.moe_a2a_backend, server_args.deepep_mode
        if (
            context.num_experts_per_tok
            and backend == "deepep"
            and context.ep_size > 1
            and mode in ("normal", "auto")
        ):
            prt, ant_moving, combine = _deepep_round_config()
            try:
                proposed["DEEPEP_HCCL_BUFFSIZE"] = deepep_normal_buffer_mb(
                    rows,
                    context.hidden_size,
                    context.num_experts_per_tok,
                    prt,
                    ant_moving,
                    combine,
                )
            except ValueError as exc:
                raise ValueError(
                    f"{exc} DEEPEP_NORMAL_LONG_SEQ_ROUND="
                    f"{os.environ.get('DEEPEP_NORMAL_LONG_SEQ_ROUND', '1')}"
                ) from exc
        else:
            logger.info(
                "NPU autotune: DeepEP sizing skipped; not applicable: "
                "topk=%s backend=%s ep=%s mode=%s",
                context.num_experts_per_tok,
                backend,
                context.ep_size,
                mode,
            )
    # Validation must finish before either value becomes visible to workers.
    for name, value in proposed.items():
        logger.info(
            "NPU autotune: %s=%s MiB source=automatic rows=%s hidden=%s "
            "dtype=%s tp=%s ep=%s",
            name,
            value,
            rows,
            context.hidden_size,
            context.dtype,
            context.tp_size,
            context.ep_size,
        )
    os.environ.update({name: str(value) for name, value in proposed.items()})
