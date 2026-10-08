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
"""ROCm latent-MoE, preroute, and PTPC paths for Kimi-K3.

Shared ``kimi_k3.py`` only keeps ``_is_hip`` hooks into this module. These
functions take the ``KimiK3MoE`` module as ``self``.
"""

from typing import Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.communication.k3_moe_pair_ar import all_reduce_moe_latent_shared
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

_use_aiter = envs.SGLANG_USE_AITER.get()
_aiter_moe_preroute_fp8 = envs.SGLANG_ROCM_K3_AITER_MOE_PREROUTE_FP8.get()
_aiter_latent_tail_fp8 = envs.SGLANG_ROCM_K3_AITER_LATENT_TAIL_FP8.get()
_aiter_tuned_moe_front = envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT.get()
_aiter_tuned_moe_front_min_tokens = (
    envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT_MIN_TOKENS.get()
)
_aiter_tuned_moe_front_max_tokens = (
    envs.SGLANG_ROCM_K3_AITER_TUNED_MOE_FRONT_MAX_TOKENS.get()
)
_moe_latent_mxfp4 = envs.SGLANG_ROCM_K3_MOE_LATENT_MXFP4.get()
_moe_latent_mxfp4_min_tokens = envs.SGLANG_ROCM_K3_MOE_LATENT_MXFP4_MIN_TOKENS.get()


def prepare_moe_latent_mxfp4(self) -> None:
    """Pack non-EP latent projections for the large-M MXFP4 path."""
    if (
        not _moe_latent_mxfp4
        or not self.use_latent_moe
        or not (self._eligible_for_fused_front or self._partial_fused_front)
        or self._front_sizes is None
        or len(self._front_sizes) not in (2, 3)
    ):
        return
    from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip

    if not latent_mxfp4_aiter_hip.supported():
        return
    head_rows = sum(self._front_sizes[:-1])
    front_head = self._front_w[:head_rows]
    down = self._front_w[head_rows:]
    up = self.routed_expert_up_proj.weight
    if (
        tuple(down.shape) != (3584, 7168)
        or tuple(up.shape) != (7168, 3584)
        or down.dtype != torch.bfloat16
        or up.dtype != torch.bfloat16
    ):
        return
    self._front_head = front_head
    self._front_down_w4, self._front_down_scale4 = latent_mxfp4_aiter_hip.pack(
        down, "latent down_proj"
    )
    self._latent_up_w4, self._latent_up_scale4 = latent_mxfp4_aiter_hip.pack(
        up, "latent up_proj"
    )


def use_moe_latent_mxfp4(self, num_tokens: int) -> bool:
    return (
        self._front_down_w4 is not None
        and self._front_down_scale4 is not None
        and self._latent_up_w4 is not None
        and self._latent_up_scale4 is not None
        and num_tokens >= _moe_latent_mxfp4_min_tokens
    )


def preroute_dense_weight(linear: torch.nn.Module) -> torch.Tensor:
    """Materialize a dense weight only while building decode-side caches."""
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16):
        return weight
    # Quark hangs the dequant on the scheme, other quant configs on the
    # quant method; either way only this cache build wants it dense.
    for owner in (getattr(linear, "scheme", None), linear.quant_method):
        materialize = getattr(owner, "materialize_bf16_weight", None)
        if materialize is not None:
            return materialize(linear)
    return weight


def prepare_preroute_fp8(self) -> None:
    if (
        not _aiter_moe_preroute_fp8
        or not self.use_latent_moe
        or self.shared_experts is None
    ):
        return
    from sglang.kernels.ops.moe import moe_preroute_aiter_hip
    from sglang.kernels.ops.quantization.aiter_fusion import (
        quantize_fp8_rows,
    )

    routed = preroute_dense_weight(self.routed_expert_down_proj)
    shared = preroute_dense_weight(self.shared_experts.gate_up_proj)
    shared_down = preroute_dense_weight(self.shared_experts.down_proj)
    if (
        tuple(routed.shape) != (3584, 7168)
        or tuple(shared.shape) != (1536, 7168)
        or tuple(shared_down.shape) != (7168, 768)
        or tuple(self.gate.weight.shape) != (896, 7168)
    ):
        return
    self._preroute_routed_weight, self._preroute_routed_scale = quantize_fp8_rows(
        routed.contiguous()
    )
    self._preroute_shared_weight, self._preroute_shared_scale = quantize_fp8_rows(
        shared.contiguous()
    )
    if moe_preroute_aiter_hip.cooperative_preactivated_enabled():
        self._preroute_shared_interleaved_weight = (
            self._preroute_shared_weight.view(2, 768, 7168)
            .permute(1, 0, 2)
            .contiguous()
            .view(1536, 7168)
        )
        self._preroute_shared_interleaved_scale = (
            self._preroute_shared_scale.view(2, 768).t().contiguous().view(1536)
        )
    (
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
    ) = quantize_fp8_rows(shared_down.contiguous())
    moe_preroute_aiter_hip.warmup(
        self._preroute_routed_weight,
        self._preroute_routed_scale,
        self._preroute_shared_weight,
        self._preroute_shared_scale,
        self.gate.weight,
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
        self._preroute_shared_interleaved_weight,
        self._preroute_shared_interleaved_scale,
        situ_beta=self._situ_beta,
        situ_linear_beta=self._situ_linear_beta,
    )


def use_latent_up_ptpc_fp8(self, latent: torch.Tensor) -> bool:
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_ptpc_fp8_moe_gemm_ok,
    )

    if self._latent_up_fp8_w is None or not _k3_ptpc_fp8_moe_gemm_ok(latent.shape[0]):
        return False
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    return ptpc_fp8_aiter_hip.covered(latent, self._latent_up_fp8_w)


def prepare_latent_up_ptpc_fp8(self) -> None:
    """Quantize the latent up-projection for the PTPC FP8 decode path."""
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_ptpc_fp8,
    )

    if not _k3_ptpc_fp8 or self.routed_expert_up_proj is None:
        return
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    weight = self.routed_expert_up_proj.weight
    if (
        not ptpc_fp8_aiter_hip.available()
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
    ):
        return
    out_features, in_features = weight.shape
    (
        self._latent_up_fp8_w,
        self._latent_up_fp8_s,
        self._latent_up_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack(weight.contiguous())
    ptpc_fp8_aiter_hip.warmup(
        self._latent_up_fp8_w,
        self._latent_up_fp8_s,
        self._latent_up_fp8_n,
        in_features,
    )


def prepare_shared_down_ptpc_fp8(self) -> None:
    """Quantize the shared-expert down projection for decode."""
    from sglang.srt.models.kimi_k3_rocm_fusion import (
        _k3_ptpc_fp8_shared_down,
    )

    if not _k3_ptpc_fp8_shared_down or self.shared_experts is None:
        return
    # Requantizing Quark's dequantized MXFP4 weight to FP8 stacks two
    # rounding steps; full GSM8K fell to 0.937 (vs 0.949 in BF16).
    if getattr(self.shared_experts.down_proj, "dequantized_bf16", False):
        return
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    weight = preroute_dense_weight(self.shared_experts.down_proj)
    if (
        not ptpc_fp8_aiter_hip.available()
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
    ):
        return
    _, in_features = weight.shape
    (
        self._shared_down_fp8_w,
        self._shared_down_fp8_s,
        self._shared_down_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack(weight.contiguous())
    ptpc_fp8_aiter_hip.warmup(
        self._shared_down_fp8_w,
        self._shared_down_fp8_s,
        self._shared_down_fp8_n,
        in_features,
        token_buckets=(2, 4, 8, 16, 32, 64, 128, 256),
    )


def prepare_latent_tail_fp8(self) -> None:
    if (
        not _aiter_latent_tail_fp8
        or not self.fuse_ar_norm
        or self.routed_expert_up_proj is None
        or self.routed_expert_norm is None
    ):
        return
    from sglang.kernels.ops.moe import latent_tail_aiter_hip

    if tuple(self.routed_expert_up_proj.weight.shape) != (7168, 3584):
        return
    self._latent_tail_weight, self._latent_tail_scale = latent_tail_aiter_hip.pack(
        self.routed_expert_up_proj.weight
    )
    norm_weight, epsilon = self._get_fused_norm_params()
    latent_tail_aiter_hip.warmup(
        norm_weight,
        self._latent_tail_weight,
        self._latent_tail_scale,
        epsilon,
    )


def try_shared_down_ptpc(self, x: torch.Tensor, out: torch.Tensor) -> bool:
    from sglang.srt.models.kimi_k3_rocm_fusion import _k3_ptpc_fp8_moe_gemm_ok

    if self._shared_down_fp8_w is None or not _k3_ptpc_fp8_moe_gemm_ok(x.shape[0]):
        return False
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if not ptpc_fp8_aiter_hip.covered(x, self._shared_down_fp8_w):
        return False
    ptpc_fp8_aiter_hip.run(
        x,
        self._shared_down_fp8_w,
        self._shared_down_fp8_s,
        self._shared_down_fp8_n,
        out=out,
    )
    return True


def try_shared_down_preroute(self, gate_up, shared_output) -> bool:
    if (
        self._preroute_shared_down_weight is None
        or self._preroute_shared_down_scale is None
    ):
        return False
    from sglang.kernels.ops.moe import moe_preroute_aiter_hip

    if not moe_preroute_aiter_hip.shared_down_covered(
        gate_up,
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
    ):
        return False
    moe_preroute_aiter_hip.run_shared_down(
        gate_up,
        self._preroute_shared_down_weight,
        self._preroute_shared_down_scale,
        situ_beta=self._situ_beta,
        situ_linear_beta=self._situ_linear_beta,
        out=shared_output,
    )
    return True


def try_moe_preroute_front(self, hidden_states: torch.Tensor, num_tokens: int):
    if not (
        num_tokens <= 4
        and self._preroute_routed_weight is not None
        and self._preroute_routed_scale is not None
        and self._preroute_shared_weight is not None
        and self._preroute_shared_scale is not None
    ):
        return None
    from sglang.kernels.ops.moe import moe_preroute_aiter_hip

    if (
        self._preroute_shared_interleaved_weight is not None
        and self._preroute_shared_interleaved_scale is not None
        and moe_preroute_aiter_hip.cooperative_preactivated_tri_covered(
            hidden_states,
            self._preroute_routed_weight,
            self._preroute_routed_scale,
            self._preroute_shared_interleaved_weight,
            self._preroute_shared_interleaved_scale,
            self.gate.weight,
        )
    ):
        routed_input, gate_up, router_logits = (
            moe_preroute_aiter_hip.run_tri_cooperative_preactivated(
                hidden_states,
                self._preroute_routed_weight,
                self._preroute_routed_scale,
                self._preroute_shared_interleaved_weight,
                self._preroute_shared_interleaved_scale,
                self.gate.weight,
                situ_beta=self._situ_beta,
                situ_linear_beta=self._situ_linear_beta,
            )
        )
        return routed_input, gate_up, router_logits, True
    if moe_preroute_aiter_hip.tri_covered(
        hidden_states,
        self._preroute_routed_weight,
        self._preroute_routed_scale,
        self._preroute_shared_weight,
        self._preroute_shared_scale,
        self.gate.weight,
    ):
        routed_input, gate_up, router_logits = moe_preroute_aiter_hip.run_tri(
            hidden_states,
            self._preroute_routed_weight,
            self._preroute_routed_scale,
            self._preroute_shared_weight,
            self._preroute_shared_scale,
            self.gate.weight,
        )
        return routed_input, gate_up, router_logits, False
    return None


def run_latent_mxfp4(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip

    return latent_mxfp4_aiter_hip.run(x, weight, scale)


def run_latent_up_ptpc(self, latent: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    return ptpc_fp8_aiter_hip.run(
        latent,
        self._latent_up_fp8_w,
        self._latent_up_fp8_s,
        self._latent_up_fp8_n,
    )


def try_latent_tail(self, latent, shared_output, prefix_sum, fused_norm):
    from sglang.kernels.ops.moe import latent_tail_aiter_hip

    norm_weight, epsilon = self._get_fused_norm_params()
    if not latent_tail_aiter_hip.covered(
        latent,
        shared_output,
        norm_weight,
        self._latent_tail_weight,
        self._latent_tail_scale,
        epsilon,
        prefix_sum,
    ):
        return None
    return latent_tail_aiter_hip.run(
        latent,
        shared_output,
        norm_weight,
        self._latent_tail_weight,
        self._latent_tail_scale,
        epsilon,
        prefix_sum,
        skip_rms=fused_norm,
    )


def init_moe_rocm_state(self, config) -> None:
    """ROCm-only buffers, filled by prepare_moe_rocm after weights load."""
    self._partial_fused_front = False
    self._preroute_routed_weight = None
    self._preroute_routed_scale = None
    self._preroute_shared_weight = None
    self._preroute_shared_scale = None
    self._preroute_shared_interleaved_weight = None
    self._preroute_shared_interleaved_scale = None
    self._preroute_shared_down_weight = None
    self._preroute_shared_down_scale = None
    self._latent_tail_weight = None
    self._latent_tail_scale = None
    self._latent_up_fp8_w = None
    self._latent_up_fp8_s = None
    self._latent_up_fp8_n = 0
    self._shared_down_fp8_w = None
    self._shared_down_fp8_s = None
    self._shared_down_fp8_n = 0
    self._front_head = None
    self._front_down_w4 = None
    self._front_down_scale4 = None
    self._front_down_fp8_w = None
    self._front_down_fp8_s = None
    self._front_down_fp8_n = 0
    self._front_down_fp8_min_tokens = 0
    self._latent_up_w4 = None
    self._latent_up_scale4 = None
    self._situ_beta = float(config.activation_situ_beta)
    self._situ_linear_beta = float(config.activation_situ_linear_beta)


def select_front_modules(self) -> Optional[list]:
    """Modules for the merged MoE front on ROCm, or None to skip the merge."""
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable

    if envs.SGLANG_ROCM_K3_QUARK_SHARED_FULL_FRONT.get():
        from sglang.srt.models.kimi_k3_rocm_quant import (
            _k3_densify_quark_shared_experts,
        )

        _k3_densify_quark_shared_experts(self)
    if self.shared_experts is not None and get_moe_a2a_backend().is_none():
        full_front = [
            self.shared_experts.gate_up_proj,
            self.gate,
            self.routed_expert_down_proj,
        ]
        if _is_unquantized_mergeable([m.weight for m in full_front]):
            return full_front
        if envs.SGLANG_K3_FUSED_FRONT.get():
            # Quark quantizes the shared experts but leaves the router and
            # latent projections dense; keep the gate+latent merge and leave
            # the shared branch on its native quantized kernels.
            return [self.gate, self.routed_expert_down_proj]
        return None
    if envs.SGLANG_K3_FUSED_FRONT.get():
        return [self.gate, self.routed_expert_down_proj]
    return None


def prepare_moe_rocm(self) -> None:
    """Post-load ROCm packing; runs after the shared front merge."""
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.models.kimi_k3_rocm_quant import k3_prepare_front_down_fp8

    # Dense gate+latent front with a separately quantized shared branch.
    self._partial_fused_front = (
        self.use_latent_moe
        and self.shared_experts is not None
        and self._front_w is not None
        and self._front_is_ep_pair
        and get_moe_a2a_backend().is_none()
    )
    prepare_moe_latent_mxfp4(self)
    k3_prepare_front_down_fp8(self)
    prepare_preroute_fp8(self)
    prepare_latent_tail_fp8(self)
    prepare_latent_up_ptpc_fp8(self)
    prepare_shared_down_ptpc_fp8(self)
    if _use_aiter:
        # AITER's router reads the correction bias in the gate-logit dtype.
        bias = self.gate.e_score_correction_bias
        if bias.dtype != self.gate.weight.dtype:
            bias.data = bias.data.to(self.gate.weight.dtype)


def try_aiter_tuned_front_gemm(
    x: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """AITER tuned GEMM for the merged MoE front, or None."""
    if not (
        _use_aiter
        and _aiter_tuned_moe_front
        and _aiter_tuned_moe_front_min_tokens
        <= x.shape[0]
        <= _aiter_tuned_moe_front_max_tokens
        # BF16 K3 uses the 6016-row full front; Quark's MXFP4 shared experts
        # leave a 4480-row router+latent front. Both are tuned for gfx950.
        and tuple(weight.shape) in ((6016, 7168), (4480, 7168))
        and type(weight.data) is torch.Tensor
    ):
        return None
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(x, weight, None, otype=x.dtype)


def _front_needs_dense_bf16(self) -> bool:
    # AITER's MXFP8 activation route indexes rows by input.stride(-2), so it
    # consumes the fused-front split view directly.
    runner = self.experts.runner
    if runner is not None and runner.runner_backend.is_aiter():
        return False
    return self._moe_front_needs_dense_bf16


def _run_shared_down(self, x: torch.Tensor, out: torch.Tensor) -> None:
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

    if try_shared_down_ptpc(self, x, out):
        return
    _k3_bf16_gemm(x, self.shared_experts.down_proj.weight, out=out)


def _forward_shared(self, gate_up, shared_output, *, preactivated: bool) -> None:
    if preactivated:
        _run_shared_down(self, gate_up, shared_output)
        return
    if try_shared_down_preroute(self, gate_up, shared_output):
        return
    _run_shared_down(self, self.shared_experts.act_fn(gate_up), shared_output)


def _mxfp4_apply_into(
    linear: torch.nn.Module, x: torch.Tensor, output: torch.Tensor
) -> bool:
    """Run an MXFP4 linear into ``output`` when the fused quant+GEMM exists.

    Quark keeps ``apply_into`` on ``linear.scheme``, not ``quant_method``;
    missing it falls back to a separate quant plus GEMM per projection.
    """
    scheme = getattr(linear, "scheme", None)
    apply_into = getattr(scheme, "apply_into", None) if scheme is not None else None
    if apply_into is None:
        apply_into = getattr(getattr(linear, "quant_method", None), "apply_into", None)
    if apply_into is None:
        return False
    apply_into(linear, x, output)
    return True


def _forward_quantized_shared(
    self, hidden_states: torch.Tensor, shared_output: torch.Tensor
) -> None:
    """Run a mixed-layout shared MLP into the fused collective buffer."""
    shared = self.shared_experts
    n_out = shared.gate_up_proj.weight.shape[0]
    num_tokens = hidden_states.shape[0]
    # Fused quant+GEMM helps decode (M<=64); at prefill widths the unfused
    # MXFP4 GEMM is faster.
    use_fused = num_tokens <= 64
    gate_up = hidden_states.new_empty(num_tokens, n_out, dtype=hidden_states.dtype)
    if not (
        use_fused and _mxfp4_apply_into(shared.gate_up_proj, hidden_states, gate_up)
    ):
        gate_up, _ = shared.gate_up_proj(hidden_states)
    activated = shared.act_fn(gate_up)
    if not (
        use_fused and _mxfp4_apply_into(shared.down_proj, activated, shared_output)
    ):
        output, _ = shared.down_proj(activated)
        shared_output.copy_(output)


def _run_front(self, hidden_states: torch.Tensor, *, use_mxfp4: bool):
    """Return (gate_up, router_logits, routed_input, shared_is_preactivated)."""
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm
    from sglang.srt.models.kimi_k3_rocm_quant import (
        k3_run_front_down_fp8,
        k3_use_front_down_fp8,
    )

    num_tokens = hidden_states.shape[0]
    preroute = try_moe_preroute_front(self, hidden_states, num_tokens)
    if preroute is not None:
        routed_input, gate_up, router_logits, shared_is_preactivated = preroute
        return gate_up, router_logits, routed_input, shared_is_preactivated

    partial_front = self._partial_fused_front
    # Once the batch outgrows the preroute megakernel, quantize the latent
    # down-projection instead of running the whole front as one BF16 GEMM.
    front_down_fp8 = None
    if not use_mxfp4 and k3_use_front_down_fp8(self, num_tokens):
        front_down_fp8 = k3_run_front_down_fp8(self, hidden_states)
    if use_mxfp4 or front_down_fp8 is not None:
        head = _k3_bf16_gemm(
            hidden_states,
            self._front_head,
            out_dtype=torch.float32 if use_mxfp4 and self._front_fp32 else None,
        )
        if partial_front:
            gate_up, router_logits = None, head
        else:
            gate_up, router_logits = torch.split(head, self._front_sizes[:2], dim=-1)
        if use_mxfp4:
            routed_input = run_latent_mxfp4(
                hidden_states, self._front_down_w4, self._front_down_scale4
            )
        else:
            routed_input = front_down_fp8
        return gate_up, router_logits, routed_input, False
    if partial_front:
        fused = _k3_bf16_gemm(hidden_states, self._front_w)
        router_logits, routed_input = torch.split(fused, self._front_sizes, dim=-1)
        return None, router_logits, routed_input, False
    fused = _k3_bf16_gemm(
        hidden_states,
        self._front_w,
        out_dtype=torch.float32 if self._front_fp32 else None,
    )
    gate_up, router_logits, routed_input = torch.split(fused, self._front_sizes, dim=-1)
    return gate_up, router_logits, routed_input, False


def _all_reduce_pair(
    self,
    buf: torch.Tensor,
    *,
    num_tokens: int,
    hidden_size: int,
    allow_fused_norm: bool,
):
    """Reduce the flat [latent | shared] buffer.

    Returns (latent, shared_output, fused_norm); with fused_norm the latent is
    already RMS-normalized.
    """
    from sglang.srt.layers.communication import k3_ar_fusion
    from sglang.srt.layers.communication.hip_fused_ar_rmsnorm import (
        try_fused_ar_rmsnorm,
    )

    latent_numel = num_tokens * self.moe_hidden_size
    if allow_fused_norm and self.fuse_ar_norm:
        weight, eps = self._get_fused_norm_params()
        view = buf.view(-1, k3_ar_fusion.NORM_DIM)
        fused = try_fused_ar_rmsnorm(view, weight, eps, num_norm_rows=num_tokens)
        if fused is not None:
            normed, reduced = fused
            if reduced.data_ptr() != view.data_ptr():
                buf.copy_(reduced.reshape(-1))
            shared_output = buf[latent_numel:].view(num_tokens, hidden_size)
            return normed[:num_tokens], shared_output, True
    # A 16K concat (~336 MiB) misses the 256 MiB QR cap and becomes NCCL
    # Generic; split so each slice fits QR.
    latent, shared_output = all_reduce_moe_latent_shared(
        buf,
        num_tokens=num_tokens,
        moe_hidden_size=self.moe_hidden_size,
        hidden_size=hidden_size,
    )
    return latent, shared_output, False


def forward_fused_rocm(
    self,
    hidden_states: torch.Tensor,
    *,
    prefix_sum: Optional[torch.Tensor],
    forward_batch: Optional[ForwardBatch],
) -> torch.Tensor:
    """ROCm counterpart of KimiK3MoE._forward_fused.

    Same [latent | shared] single-collective layout; adds the FP8 preroute
    front, the MXFP4/FP8 latent projections, the partial (Quark) front, the
    fused AR+RMSNorm, and the fused latent tail.
    """
    from sglang.srt.models.kimi_k3 import _add3, _aiter_k3_opt

    num_tokens, hidden_size = hidden_states.shape
    use_mxfp4 = use_moe_latent_mxfp4(self, num_tokens)
    gate_up, router_logits, routed_input, shared_is_preactivated = _run_front(
        self, hidden_states, use_mxfp4=use_mxfp4
    )
    if num_tokens > 1 and not _aiter_k3_opt:
        router_logits = router_logits.contiguous()
    if _front_needs_dense_bf16(self):
        routed_input = routed_input.to(hidden_states.dtype).contiguous()
    latent_numel = num_tokens * self.moe_hidden_size
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        buf = hidden_states.new_empty(latent_numel + num_tokens * hidden_size)
    latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
    shared_output = buf[latent_numel:].view(num_tokens, hidden_size)

    if gate_up is None:
        _forward_quantized_shared(self, hidden_states, shared_output)
    else:
        _forward_shared(
            self, gate_up, shared_output, preactivated=shared_is_preactivated
        )
    self._forward_routed(hidden_states, router_logits, routed_input, latent)

    use_latent_tail = (
        num_tokens in (1, 2, 4)
        and forward_batch is not None
        and forward_batch.forward_mode.is_decode_or_idle()
        and self._latent_tail_weight is not None
        and self._latent_tail_scale is not None
    )
    # The fused latent tail applies the RMSNorm itself.
    latent, shared_output, fused_norm = _all_reduce_pair(
        self,
        buf,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        allow_fused_norm=not use_latent_tail,
    )
    if use_latent_tail:
        out = try_latent_tail(self, latent, shared_output, prefix_sum, fused_norm)
        if out is not None:
            return out
    if not fused_norm:
        latent = self._latent_norm(latent)
    if use_mxfp4:
        out = run_latent_mxfp4(latent, self._latent_up_w4, self._latent_up_scale4)
    elif use_latent_up_ptpc_fp8(self, latent):
        out = run_latent_up_ptpc(self, latent)
    else:
        out, _ = self.routed_expert_up_proj(latent)
    return _add3(out, shared_output, prefix_sum, prefetch_bc=True)
