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
"""Kimi-K3 ROCm weight preparation for Quark checkpoints.

Quark serializes K3 as per-output-channel FP8, which is not the layout the
AITER kernels want. The conversions live here; ``kimi_k3.py`` calls every entry
point behind ``_is_hip`` and keeps the checkpoint's own layout otherwise.
"""

from typing import Optional

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3_rocm_fusion import _k3_log_once, _k3_ptpc_fp8


def _k3_channel_fp8_to_bf16(module: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Dequantize a per-output-channel FP8 weight to dense bf16.

    The aiter batched absorb GEMM takes ``w_scale`` as a single scalar (the
    Triton kernel documents it as "per-batch scale for WQ with shape (1,)"), so
    a per-channel vector cannot reach it. Requantizing to one per-tensor scale
    would fit, but the scale axis is the GEMM's *contraction* axis, and on real
    K3 weights that second quantization costs ~2.3% relative error against the
    checkpoint (worst channel ~3.8%). Dequantizing instead costs ~0.12% and
    lets the absorb run its bf16 path with w_scale left at the 1.0 default.

    Must run while dim 0 is still the channel axis weight_scale indexes, i.e.
    before the kv_b_proj head split.
    """
    weight_scale = module.weight_scale
    # Per-channel scale is 1D [out]; reshape so it broadcasts against [out, in].
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.view(-1, 1)
    return (weight.to(torch.float32) * weight_scale.to(torch.float32)).to(
        torch.bfloat16
    )


def _k3_kda_inproj_channel_fp8_scales(
    self_attn: nn.Module,
) -> Optional[list[torch.Tensor]]:
    """Per-channel FP8 scales of the three KDA input projections, if uniform.

    Quark stores self_attn.* as [out, in] e4m3 plus an [out] fp32 scale --
    already the PTPC layout -- so the merge can reuse them instead of
    re-quantizing a dequantized copy."""
    if not (
        _k3_ptpc_fp8
        and envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get()
        and self_attn.do_fuse_qkvbfg
        and self_attn.use_full_rank_gate
    ):
        return None
    scales = []
    for module in (self_attn.fused_qkvg_proj, self_attn.f_a_proj, self_attn.b_proj):
        weight = module.weight
        scale = getattr(module, "weight_scale", None)
        if (
            type(weight.data) is not torch.Tensor
            or weight.dim() != 2
            or weight.dtype != torch.float8_e4m3fn
            or not isinstance(scale, torch.Tensor)
            or scale.numel() != weight.shape[0]
        ):
            return None
        scales.append(scale.data.reshape(-1).float())
    return scales


def _k3_merge_kda_inproj_fp8(self_attn: nn.Module) -> bool:
    """Merge the per-channel FP8 KDA input projections into one PTPC GEMM."""
    scales = _k3_kda_inproj_channel_fp8_scales(self_attn)
    if scales is None:
        return False
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if not ptpc_fp8_aiter_hip.available():
        return False
    mods = (self_attn.fused_qkvg_proj, self_attn.f_a_proj, self_attn.b_proj)
    weights = [module.weight.data for module in mods]
    if len({weight.shape[1] for weight in weights}) != 1:
        return False
    sizes = [weight.shape[0] for weight in weights]
    # Same 8-row pad as the BF16 merge: it keeps every fused-output row
    # 16-byte aligned for the vectorized consumers of the split slices.
    pad = (-sum(sizes)) % 8
    if pad:
        weights.append(weights[0].new_zeros((pad, weights[0].shape[1])))
        scales.append(scales[0].new_ones(pad))
    (
        self_attn._qkvgbfa_fp8_w,
        self_attn._qkvgbfa_fp8_s,
        self_attn._qkvgbfa_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack_prequantized(
        torch.cat(weights, dim=0), torch.cat(scales)
    )
    self_attn._bfa_fa_size, self_attn._bfa_b_size = sizes[1], sizes[2]
    self_attn._qkvgbfa_sizes = [*self_attn.split_sizes, sizes[1], sizes[2], pad]
    ptpc_fp8_aiter_hip.warmup(
        self_attn._qkvgbfa_fp8_w,
        self_attn._qkvgbfa_fp8_s,
        self_attn._qkvgbfa_fp8_n,
        weights[0].shape[1],
    )
    _k3_prepare_f_b_tiny_gemm(self_attn)
    return True


def _k3_prepare_f_b_tiny_gemm(self_attn: nn.Module) -> None:
    """Dequant Quark PTPC ``f_b`` into the BF16 tiny-GEMM buffer.

    Decode ``f_b`` is ``[M, 128] @ [1536, 128].T``; tiny-GEMM covers it
    without the contiguous copy and group-quant the FP8 path needs.
    """
    if self_attn._bfa_f_b_w is not None:
        return
    weight = self_attn.f_b_proj.weight
    scale = getattr(self_attn.f_b_proj, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return
    # Online quantization leaves f_b dense (the fused KDA decode kernel wants a
    # bf16 weight), so the merged FP8 in-projection can reach here with nothing
    # to dequantize; the tiny GEMM takes that weight as is.
    is_prequantized = weight.dtype == torch.float8_e4m3fn
    if is_prequantized:
        if not isinstance(scale, torch.Tensor) or scale.numel() != weight.shape[0]:
            return
    elif weight.dtype != torch.bfloat16:
        return
    n, k = int(weight.shape[0]), int(weight.shape[1])
    from sglang.kernels.ops.gemm.kimi_k3 import _K3_TINY_GEMM_MAX_TOKENS

    if (n, k) not in _K3_TINY_GEMM_MAX_TOKENS:
        return
    self_attn._bfa_f_b_w = (
        (weight.data.float() * scale.data.reshape(-1, 1).float())
        .to(torch.bfloat16)
        .contiguous()
        if is_prequantized
        else weight.data.contiguous()
    )
    _k3_log_once(
        "kda_f_b_tiny_gemm", "K3 KDA f_b BF16 tiny-GEMM enabled (N=%d K=%d)", n, k
    )


def _k3_qkvgbfa_inproj(self_attn: nn.Module, hidden_states) -> Optional[torch.Tensor]:
    """Run the merged [q,k,v,g | f_a | b] projection, or None if uncovered."""
    if self_attn._use_qkvgbfa_ptpc_fp8(hidden_states):
        from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

        if isinstance(hidden_states, tuple):
            return ptpc_fp8_aiter_hip.run(
                hidden_states[0],
                self_attn._qkvgbfa_fp8_w,
                self_attn._qkvgbfa_fp8_s,
                self_attn._qkvgbfa_fp8_n,
                x_scale=hidden_states[1],
            )
        return ptpc_fp8_aiter_hip.run(
            hidden_states,
            self_attn._qkvgbfa_fp8_w,
            self_attn._qkvgbfa_fp8_s,
            self_attn._qkvgbfa_fp8_n,
        )
    # A prequantized merge has no dense carrier to hand the linear method.
    if self_attn._qkvgbfa_layer is None:
        return None
    return self_attn.fused_qkvg_proj.quant_method.apply(
        self_attn._qkvgbfa_layer, hidden_states, None
    )


def _k3_apply_f_b(self_attn: nn.Module, f_a: torch.Tensor) -> torch.Tensor:
    if self_attn._bfa_f_b_w is None:
        # f_a is a column slice of the merged projection and the FP8
        # activation quantizer asserts on contiguity; the tiny GEMM below
        # takes the view as is.
        return self_attn.f_b_proj(f_a.contiguous())[0]
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

    if f_a.stride(-1) != 1:
        f_a = f_a.contiguous()
    return kimi_k3_tiny_gemm(f_a, self_attn._bfa_f_b_w)


def k3_prepare_front_down_fp8(mlp: nn.Module) -> None:
    """Pack the fused front's latent down-projection for the PTPC FP8 path.

    Mirrors ``_prepare_moe_latent_mxfp4``'s split of the merged front weight --
    ``[gate_up | router | latent_down]`` -- keeping the router head BF16 (FP8
    logits move the top-k pick) and quantizing only the ``[3584, 7168]`` tail.

    The head view is shared with the MXFP4 path when both are packed; it is a
    slice of ``_front_w``, so this costs one FP8 copy of the down-projection and
    nothing else.
    """
    if not envs.SGLANG_ROCM_K3_MOE_LATENT_FP8.get() or not mlp.use_latent_moe:
        return
    if not (mlp._eligible_for_fused_front or getattr(mlp, "_eligible_for_partial_fused_front", False)):
        return
    if mlp._front_sizes is None or len(mlp._front_sizes) not in (2, 3):
        return

    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if not ptpc_fp8_aiter_hip.available():
        return
    head_rows = sum(mlp._front_sizes[:-1])
    down = mlp._front_w[head_rows:]
    if tuple(down.shape) != (3584, 7168) or down.dtype != torch.bfloat16:
        return

    if mlp._front_head is None:
        mlp._front_head = mlp._front_w[:head_rows]
    # Resolved once here rather than per forward: k3_use_front_down_fp8 runs on
    # every layer of every step, and this pack already happens after load.
    mlp._front_down_fp8_min_tokens = (
        envs.SGLANG_ROCM_K3_MOE_LATENT_FP8_MIN_TOKENS.get()
    )
    (
        mlp._front_down_fp8_w,
        mlp._front_down_fp8_s,
        mlp._front_down_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack(down.contiguous())
    # Kernel selection must happen outside cuda-graph capture.
    ptpc_fp8_aiter_hip.warmup(
        mlp._front_down_fp8_w,
        mlp._front_down_fp8_s,
        mlp._front_down_fp8_n,
        down.shape[1],
    )
    _k3_log_once(
        "k3_front_down_fp8",
        "K3 ROCm: latent front down-projection packed as PTPC FP8 "
        "(decode batches >= %d)",
        mlp._front_down_fp8_min_tokens,
    )


def k3_use_front_down_fp8(mlp: nn.Module, num_tokens: int) -> bool:
    return (
        getattr(mlp, "_front_down_fp8_w", None) is not None
        and num_tokens >= mlp._front_down_fp8_min_tokens
    )


def k3_run_front_down_fp8(
    mlp: nn.Module, hidden_states: torch.Tensor
) -> Optional[torch.Tensor]:
    """``hidden_states @ latent_down.T`` in PTPC FP8, or None if uncovered."""
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if not ptpc_fp8_aiter_hip.covered(hidden_states, mlp._front_down_fp8_w):
        return None
    return ptpc_fp8_aiter_hip.run(
        hidden_states,
        mlp._front_down_fp8_w,
        mlp._front_down_fp8_s,
        mlp._front_down_fp8_n,
    )


def k3_prepare_shared_expert_ptpc(mlp: nn.Module) -> None:
    """Pack shared gate_up and down as preshuffle PTPC copies.

    The BF16 parameters stay in place so ``_merge_front_weights`` and
    ``_eligible_for_fused_front`` keep working. The router rows of the merged
    front are not packed.
    """
    want_gate_up = envs.SGLANG_ROCM_K3_ONLINE_FP8_SHARED_EXPERTS.get()
    want_down = want_gate_up or envs.SGLANG_ROCM_K3_PTPC_FP8_SHARED_DOWN.get()
    if not want_gate_up and not want_down:
        return
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if not ptpc_fp8_aiter_hip.available():
        return

    if (
        want_gate_up
        and mlp._front_w is not None
        and mlp._front_sizes is not None
        and len(mlp._front_sizes) == 3
        and not mlp._front_is_ep_pair
    ):
        rows = mlp._front_sizes[0]
        gate_up = mlp._front_w[:rows]
        if (
            isinstance(gate_up, torch.Tensor)
            and gate_up.ndim == 2
            and gate_up.dtype == torch.bfloat16
        ):
            if mlp._front_head is None:
                mlp._front_head = mlp._front_w[: sum(mlp._front_sizes[:-1])]
            (
                mlp._front_gate_up_fp8_w,
                mlp._front_gate_up_fp8_s,
                mlp._front_gate_up_fp8_n,
            ) = ptpc_fp8_aiter_hip.pack(gate_up.contiguous())
            ptpc_fp8_aiter_hip.warmup(
                mlp._front_gate_up_fp8_w,
                mlp._front_gate_up_fp8_s,
                mlp._front_gate_up_fp8_n,
                gate_up.shape[1],
            )
            _k3_log_once(
                "k3_front_gate_up_fp8",
                "K3 ROCm: shared gate_up packed as PTPC FP8 (router stays BF16)",
            )

    shared = getattr(mlp, "shared_experts", None)
    down = getattr(getattr(shared, "down_proj", None), "weight", None)
    if (
        want_down
        and isinstance(down, torch.Tensor)
        and down.ndim == 2
        and down.dtype == torch.bfloat16
    ):
        (
            mlp._shared_down_fp8_w,
            mlp._shared_down_fp8_s,
            mlp._shared_down_fp8_n,
        ) = ptpc_fp8_aiter_hip.pack(down.contiguous())
        mlp._shared_down_fp8_min_tokens = envs.SGLANG_ROCM_K3_PTPC_FP8_MIN_TOKENS.get()
        ptpc_fp8_aiter_hip.warmup(
            mlp._shared_down_fp8_w,
            mlp._shared_down_fp8_s,
            mlp._shared_down_fp8_n,
            down.shape[1],
        )
        _k3_log_once(
            "k3_shared_down_fp8",
            "K3 ROCm: shared down-projection packed as PTPC FP8 "
            "(BF16 weight kept, decode batches >= %d)",
            mlp._shared_down_fp8_min_tokens,
        )


def k3_run_front_gate_up_fp8(
    mlp: nn.Module, hidden_states: torch.Tensor
) -> Optional[torch.Tensor]:
    """``hidden_states @ gate_up.T`` in PTPC FP8, or None if uncovered."""
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    weight = getattr(mlp, "_front_gate_up_fp8_w", None)
    if not ptpc_fp8_aiter_hip.covered(hidden_states, weight):
        return None
    return ptpc_fp8_aiter_hip.run(
        hidden_states,
        weight,
        mlp._front_gate_up_fp8_s,
        mlp._front_gate_up_fp8_n,
    )


def k3_run_shared_down_fp8(
    mlp: nn.Module, act: torch.Tensor, out: torch.Tensor
) -> bool:
    """Write the shared down GEMM into ``out``. False leaves the BF16 path."""
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    weight = getattr(mlp, "_shared_down_fp8_w", None)
    if weight is None or act.shape[0] < mlp._shared_down_fp8_min_tokens:
        return False
    if not act.is_contiguous():
        act = act.contiguous()
    if not ptpc_fp8_aiter_hip.covered(act, weight):
        return False
    # Land in the symmetric-buffer slice. copy_ covers the N-padded case,
    # where torch._scaled_mm cannot take out= of the logical width.
    out.copy_(
        ptpc_fp8_aiter_hip.run(
            act,
            weight,
            mlp._shared_down_fp8_s,
            mlp._shared_down_fp8_n,
        )
    )
    return True
