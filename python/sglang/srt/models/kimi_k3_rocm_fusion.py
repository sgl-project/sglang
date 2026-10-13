# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kimi-K3 ROCm producer-quant fusion.

A "producer fusion" folds the activation quant a linear would run into the
kernel that produces its input (RMSNorm, output gate), so decode issues one
launch instead of two and hands the linear ``(fp8, scale)`` directly.

Every gate here is ``_is_hip``-derived: where the AITER kernels are absent
``_k3_ptpc_fp8`` is False and each entry point degrades to the split path.
"""

import logging
from typing import Optional

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.utils_rocm import (
    accepts_group128_fp8_tuple,
    accepts_ptpc_fp8_tuple,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_logged_fusions: set[str] = set()


def _k3_log_once(key: str, msg: str, *args) -> None:
    """Report an enabled fusion once per process, not once per layer."""
    if key not in _logged_fusions:
        _logged_fusions.add(key)
        logger.info(msg, *args)


_is_hip = is_hip()
_use_aiter = envs.SGLANG_USE_AITER.get() and _is_hip
_k3_ptpc_fp8 = _is_hip and envs.SGLANG_ROCM_K3_PTPC_FP8.get()
_k3_ptpc_fp8_max_tokens = envs.SGLANG_ROCM_K3_PTPC_FP8_MAX_TOKENS.get()
_k3_ptpc_fp8_min_tokens = envs.SGLANG_ROCM_K3_PTPC_FP8_MIN_TOKENS.get()
_k3_ptpc_fp8_shared_down = _is_hip and envs.SGLANG_ROCM_K3_PTPC_FP8_SHARED_DOWN.get()


_k3_ptpc_fp8_prefill_min_tokens = envs.SGLANG_ROCM_K3_PTPC_FP8_PREFILL_MIN_TOKENS.get()


def _k3_ptpc_fp8_batch_ok(num_tokens: int) -> bool:
    return _k3_ptpc_fp8_min_tokens <= num_tokens <= _k3_ptpc_fp8_max_tokens


def _k3_ptpc_fp8_moe_gemm_ok(num_tokens: int) -> bool:
    """Batch gate for the latent-up and shared-down PTPC GEMMs, which also win
    again at prefill-chunk sizes."""
    return _k3_ptpc_fp8_batch_ok(num_tokens) or (
        0 < _k3_ptpc_fp8_prefill_min_tokens <= num_tokens
    )


def _k3_linear_accepts_ptpc_tuple(module: nn.Module) -> bool:
    """Whether a linear consumes ``(fp8, per_token_scale)`` without re-quant.

    Quark W8A8-FP8 (K3 AttnFP8 o_proj) uses ``per_token`` + ``per_channel``
    weights, not compressed-tensors ``strategy``. Both land in
    ``apply_fp8_linear`` / ``apply_fp8_ptpc_linear`` on ROCm.
    """
    return _k3_ptpc_fp8 and accepts_ptpc_fp8_tuple(module)


def _k3_attn_inproj(self_attn: nn.Module) -> Optional[nn.Module]:
    return (
        getattr(self_attn, "fused_qkvg_proj", None)
        or getattr(self_attn, "fused_qkv_a_proj_with_mqa", None)
        or getattr(self_attn, "q_a_proj", None)
        or getattr(self_attn, "qkv_proj", None)
    )


def _k3_inproj_needs_bf16_hidden(self_attn: nn.Module, num_tokens: int) -> bool:
    """True when a BF16 copy of the RMSNormed hidden is still consumed.

    Merged KDA ``qkvgbfa`` decode is a single GEMM, so the producer can emit
    only the FP8 tuple. The split ``[q,k,v,g]`` + tiny ``[f_a|b]`` path still
    reads BF16 hidden.
    """
    sizes = getattr(self_attn, "_qkvgbfa_sizes", None)
    limit = getattr(self_attn, "_qkvgbfa_bs_limit", 0) or 0
    if sizes is not None and 0 < num_tokens <= limit:
        return False
    return getattr(self_attn, "_bfa_w", None) is not None


def attn_input_quant_format(self_attn: nn.Module, num_tokens: int) -> str:
    """The quant_format the attention input read can emit for this layer's
    in-projection: ``"fp8"`` (group-128) or ``"fp8_per_token"`` (PTPC), or
    ``""`` for a plain BF16 read. The read fuses the input RMSNorm with that
    quantization, so the in-projection skips its own activation quant."""
    inproj = _k3_attn_inproj(self_attn)
    if (
        not _use_aiter
        or inproj is None
        or not _k3_ptpc_fp8_batch_ok(num_tokens)
        or _k3_inproj_needs_bf16_hidden(self_attn, num_tokens)
    ):
        return ""
    if accepts_group128_fp8_tuple(inproj):
        return "fp8"
    if _k3_linear_accepts_ptpc_tuple(inproj):
        return "fp8_per_token"
    return ""


def _k3_hidden_tensor(hidden_states):
    """The activation itself; a fired producer fusion makes it ``(fp8, scale)``."""
    return hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states


def _k3_hidden_num_tokens(hidden_states) -> int:
    return _k3_hidden_tensor(hidden_states).shape[0]


def _k3_fuse_kda_o_norm_ptpc(
    core_attn_out: torch.Tensor,
    *,
    norm_gate: torch.Tensor,
    o_norm: nn.Module,
    o_proj: nn.Module,
):
    """Fuse KDA's gated output norm with the PTPC quant ``o_proj`` expects.

    Returns the ``(fp8, per-token scale)`` tuple, or None when the split
    bf16 norm plus standalone quant has to run.
    """
    token_count = core_attn_out.shape[-3]
    if not (
        _use_aiter
        and _k3_ptpc_fp8_batch_ok(token_count)
        and _k3_linear_accepts_ptpc_tuple(o_proj)
    ):
        return None
    # The split path materializes gated RMSNorm in bf16 and then launches
    # dynamic per-token quant for o_proj. AITER's sigmoid variant quantizes
    # the fp32 gated value directly (no bf16 round trip; GSM8K-neutral) and
    # emits the tuple consumed directly by apply_fp8_ptpc_linear.
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.gated_rmsnorm_fp8_per_token_quant import (
        gated_rmsnorm_fp8_per_token_quant,
    )

    norm_input = core_attn_out.squeeze(0)
    quant_out = torch.empty(
        (token_count, norm_input.shape[-2] * norm_input.shape[-1]),
        dtype=aiter_dtypes.fp8,
        device=norm_input.device,
    )
    quant_scale = torch.empty(
        (token_count, 1), dtype=torch.float32, device=norm_input.device
    )
    gated_rmsnorm_fp8_per_token_quant(
        quant_out,
        quant_scale,
        norm_input,
        norm_gate,
        o_norm.weight,
        o_norm.eps,
        use_sigmoid=True,
    )
    _k3_log_once(
        "kda_onorm_ptpc",
        "KDA o_norm+PTPC producer fusion enabled (scheme=%s, tokens=%d)",
        type(o_proj.scheme).__name__,
        token_count,
    )
    return quant_out, quant_scale


def _k3_fuse_mla_gate_ptpc(x: torch.Tensor, *, gate: torch.Tensor, o_proj: nn.Module):
    """Fuse the MLA output gate with the PTPC quant ``o_proj`` expects.

    Returns the ``(fp8, per-token scale)`` tuple, or None when the gate has
    to be applied on its own.
    """
    if not _k3_linear_accepts_ptpc_tuple(o_proj):
        return None
    from sglang.kernels.ops.attention import mla_output_gate_fp8_quant_hip

    if not mla_output_gate_fp8_quant_hip.covered(x, gate):
        return None
    _k3_log_once("mla_gate_ptpc", "K3 MLA o_proj gate+PTPC producer fusion enabled")
    return mla_output_gate_fp8_quant_hip.kimi_k3_mla_output_gate_fp8_quant(x, gate)


