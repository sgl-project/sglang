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
"""ROCm fused MoE front for Kimi-K3: the Quark partial front, the FP8 pre-route
front and the fused latent tail.

Shared ``kimi_k3.py`` only keeps ``_is_hip`` hooks into this module. These
functions take the ``KimiK3MoE`` module as ``self``.
"""

from typing import Optional

import torch

from sglang.kernels.ops.moe import latent_tail_aiter_hip, moe_preroute_aiter_hip
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SituAndMul
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFP4
from sglang.srt.runtime_context import get_parallel


def init_moe_rocm_state(self) -> None:
    """ROCm-only switches and buffers, filled by prepare_moe_rocm after load."""
    self._k3_preroute_fp8 = envs.SGLANG_OPT_HIP_K3_MOE_PREROUTE_FP8.get()
    self._k3_preroute_preactivated = (
        envs.SGLANG_OPT_HIP_K3_PREROUTE_PREACTIVATED_SHARED.get()
    )
    self._k3_latent_tail_fp8 = envs.SGLANG_OPT_HIP_K3_LATENT_TAIL_FP8.get()
    self._k3_latent_tail_max_tokens = (
        envs.SGLANG_OPT_HIP_K3_LATENT_TAIL_MAX_TOKENS.get()
    )
    self._partial_fused_front = False
    self._preroute_routed_weight: Optional[torch.Tensor] = None
    self._preroute_routed_scale: Optional[torch.Tensor] = None
    self._preroute_shared_weight: Optional[torch.Tensor] = None
    self._preroute_shared_scale: Optional[torch.Tensor] = None
    self._preroute_shared_interleaved_weight: Optional[torch.Tensor] = None
    self._preroute_shared_interleaved_scale: Optional[torch.Tensor] = None
    self._preroute_shared_down_weight: Optional[torch.Tensor] = None
    self._preroute_shared_down_scale: Optional[torch.Tensor] = None
    self._latent_tail_weight: Optional[torch.Tensor] = None
    self._latent_tail_scale: Optional[torch.Tensor] = None
    self._situ_beta = 0.0
    self._situ_linear_beta = 0.0


def select_front_modules(self) -> Optional[list]:
    """Modules for the merged MoE front on ROCm, or None to skip the merge."""
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable

    if self.shared_experts is not None and get_moe_a2a_backend().is_none():
        full_front = [
            self.shared_experts.gate_up_proj,
            self.gate,
            self.routed_expert_down_proj,
        ]
        if _is_unquantized_mergeable([m.weight for m in full_front]):
            return full_front
    if self._k3_preroute_fp8 and envs.SGLANG_K3_FUSED_FRONT.get():
        # Quark quantizes the shared experts but leaves the router and latent
        # projections dense; only the pre-route front uses this partial merge.
        return [self.gate, self.routed_expert_down_proj]
    return None


def _keep_correction_bias_in_logit_dtype(self) -> None:
    # AITER's router reads the correction bias in the gate-logit dtype and casts
    # it on every call otherwise; both ways round it to the same values.
    from sglang.srt.layers.moe.topk import _use_aiter

    bias = self.gate.e_score_correction_bias
    if _use_aiter and bias is not None and bias.dtype != self.gate.weight.dtype:
        bias.data = bias.data.to(self.gate.weight.dtype)


def prepare_moe_rocm(self) -> None:
    """Post-load ROCm packing; runs after the shared front merge."""
    from sglang.srt.layers.moe import get_moe_a2a_backend

    # Dense gate+latent front with a separately quantized shared branch.
    self._partial_fused_front = (
        self.use_latent_moe
        and self.shared_experts is not None
        and self._front_w is not None
        and self._front_is_ep_pair
        and get_moe_a2a_backend().is_none()
    )
    _prepare_preroute_fp8(self)
    _prepare_latent_tail_fp8(self)
    _keep_correction_bias_in_logit_dtype(self)


def _dense_weight(linear: torch.nn.Module) -> torch.Tensor:
    """A dense weight for building decode-side caches; packed weights stay packed."""
    weight = linear.weight
    if weight.dtype in (torch.bfloat16, torch.float16):
        return weight
    if isinstance(linear.quant_method, QuarkLinearMethod) and isinstance(
        linear.scheme, QuarkW4A4MXFP4
    ):
        return linear.scheme.materialize_bf16_weight(linear)
    return weight


def _prepare_preroute_fp8(self) -> None:
    if not self._k3_preroute_fp8 or not self.use_latent_moe:
        return
    act_fn = self.shared_experts.act_fn if self.shared_experts is not None else None
    # The fused shared-down applies SiTU with both betas.
    if not isinstance(act_fn, SituAndMul) or act_fn.linear_beta is None:
        return
    from sglang.kernels.ops.moe.kimi_k3_flydsl.kimi_k3_moe_preroute_fp8 import (
        quantize_preroute_weight,
    )

    if not moe_preroute_aiter_hip.available():
        return

    routed = _dense_weight(self.routed_expert_down_proj)
    shared = _dense_weight(self.shared_experts.gate_up_proj)
    shared_down = _dense_weight(self.shared_experts.down_proj)
    # The kernels are specialized for Kimi-K3 at TP8.
    if (
        tuple(routed.shape) != (3584, 7168)
        or tuple(shared.shape) != (1536, 7168)
        or tuple(shared_down.shape) != (7168, 768)
        or tuple(self.gate.weight.shape) != (896, 7168)
    ):
        return
    self._situ_beta, self._situ_linear_beta = act_fn.beta, act_fn.linear_beta
    self._preroute_routed_weight, self._preroute_routed_scale = (
        quantize_preroute_weight(routed.contiguous())
    )
    self._preroute_shared_weight, self._preroute_shared_scale = (
        quantize_preroute_weight(shared.contiguous())
    )
    if (
        self._k3_preroute_preactivated
        and moe_preroute_aiter_hip.cooperative_preactivated_available()
    ):
        # The cooperative kernel reads gate and up rows interleaved.
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
    ) = quantize_preroute_weight(shared_down.contiguous())
    moe_preroute_aiter_hip.warmup(
        routed_weight=self._preroute_routed_weight,
        routed_scale=self._preroute_routed_scale,
        shared_weight=self._preroute_shared_weight,
        shared_scale=self._preroute_shared_scale,
        router_weight=self.gate.weight,
        shared_down_weight=self._preroute_shared_down_weight,
        shared_down_scale=self._preroute_shared_down_scale,
        shared_interleaved_weight=self._preroute_shared_interleaved_weight,
        shared_interleaved_scale=self._preroute_shared_interleaved_scale,
        situ_beta=self._situ_beta,
        situ_linear_beta=self._situ_linear_beta,
    )


def _prepare_latent_tail_fp8(self) -> None:
    if (
        not self._k3_latent_tail_fp8
        or not self.fuse_ar_norm
        or self.routed_expert_up_proj is None
        or self.routed_expert_norm is None
    ):
        return
    weight = self.routed_expert_up_proj.weight
    if (
        not latent_tail_aiter_hip.available()
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (7168, 3584)
    ):
        return
    if not 1 <= self._k3_latent_tail_max_tokens <= latent_tail_aiter_hip.max_tokens():
        raise ValueError(
            "SGLANG_OPT_HIP_K3_LATENT_TAIL_MAX_TOKENS must be in "
            f"[1, {latent_tail_aiter_hip.max_tokens()}], "
            f"got {self._k3_latent_tail_max_tokens}"
        )
    self._latent_tail_weight, self._latent_tail_scale = latent_tail_aiter_hip.pack(
        self.routed_expert_up_proj.weight
    )
    norm_weight, epsilon = self._get_fused_norm_params()
    latent_tail_aiter_hip.warmup(
        rms_weight=norm_weight,
        up_weight=self._latent_tail_weight,
        up_scale=self._latent_tail_scale,
        epsilon=epsilon,
        max_tokens=self._k3_latent_tail_max_tokens,
    )


def _try_preroute_front(self, hidden_states: torch.Tensor):
    """(routed_input, gate_up, router_logits, shared_is_preactivated) from the
    FP8 pre-route kernels, or None."""
    if hidden_states.shape[0] > 4 or self._preroute_routed_weight is None:
        return None
    if (
        self._preroute_shared_interleaved_weight is not None
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


def _run_front(self, hidden_states: torch.Tensor):
    """Return (gate_up, router_logits, routed_input, shared_is_preactivated);
    gate_up is None when the shared experts run on their own quantized kernels."""
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

    preroute = _try_preroute_front(self, hidden_states)
    if preroute is not None:
        routed_input, gate_up, router_logits, shared_is_preactivated = preroute
        return gate_up, router_logits, routed_input, shared_is_preactivated
    fused = _k3_bf16_gemm(hidden_states, self._front_w)
    if self._partial_fused_front:
        router_logits, routed_input = torch.split(fused, self._front_sizes, dim=-1)
        return None, router_logits, routed_input, False
    gate_up, router_logits, routed_input = torch.split(fused, self._front_sizes, dim=-1)
    return gate_up, router_logits, routed_input, False


def _try_shared_down_preroute(self, gate_up, shared_output) -> bool:
    if self._preroute_shared_down_weight is None:
        return False
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


def _forward_shared(
    self,
    hidden_states: torch.Tensor,
    gate_up: Optional[torch.Tensor],
    shared_output: torch.Tensor,
    *,
    preactivated: bool,
) -> None:
    """Write the shared-expert output into its slice of the collective buffer."""
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

    shared = self.shared_experts
    if gate_up is None:
        gate_up, _ = shared.gate_up_proj(hidden_states)
    elif not preactivated and _try_shared_down_preroute(self, gate_up, shared_output):
        return
    activated = gate_up if preactivated else shared.act_fn(gate_up)
    if shared.down_proj.weight.dtype in (torch.bfloat16, torch.float16):
        _k3_bf16_gemm(activated, shared.down_proj.weight, out=shared_output)
        return
    # Quark keeps the shared experts packed; their kernels cannot write into a
    # caller buffer.
    output, _ = shared.down_proj(activated)
    shared_output.copy_(output)


def _try_latent_tail(self, latent, shared_output, prefix_sum):
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
        skip_rms=False,
    )


def forward_fused_rocm(
    self,
    hidden_states: torch.Tensor,
    *,
    prefix_sum: Optional[torch.Tensor],
) -> torch.Tensor:
    """ROCm counterpart of KimiK3MoE._forward_fused: the same single-collective
    [latent | shared] layout, plus the Quark partial front, the FP8 pre-route
    front and the fused latent tail."""
    from sglang.srt.models.kimi_k3 import _add3, _aiter_k3_opt

    num_tokens, hidden_size = hidden_states.shape
    gate_up, router_logits, routed_input, shared_is_preactivated = _run_front(
        self, hidden_states
    )
    if num_tokens > 1 and not _aiter_k3_opt:
        router_logits = router_logits.contiguous()
    if self._moe_front_needs_dense_bf16:
        routed_input = routed_input.to(hidden_states.dtype).contiguous()
    latent_numel = num_tokens * self.moe_hidden_size
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        buf = hidden_states.new_empty(latent_numel + num_tokens * hidden_size)
    latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
    shared_output = buf[latent_numel:].view(num_tokens, hidden_size)

    _forward_shared(
        self,
        hidden_states,
        gate_up,
        shared_output,
        preactivated=shared_is_preactivated,
    )
    self._forward_routed(hidden_states, router_logits, routed_input, latent)
    buf = self._reduce_latent_and_shared(buf)
    latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
    shared_output = buf[latent_numel:].view(num_tokens, hidden_size)

    if (
        self._latent_tail_weight is not None
        and num_tokens <= self._k3_latent_tail_max_tokens
    ):
        out = _try_latent_tail(self, latent, shared_output, prefix_sum)
        if out is not None:
            return out
    out, _ = self.routed_expert_up_proj(self._latent_norm(latent))
    return _add3(out, shared_output, prefix_sum, prefetch_bc=True)
