"""Kimi-K3's fused KDA decode and target verify on ROCm.

KimiK3DeltaAttention._prepare_fused_decode calls prepare_fused_kda_hip on ROCm.
It arms the AITER or gfx950 fused decode and, under
``--linear-attn-verify-backend aiter``, installs the ``KDAFusedVerify`` hook the
KDA backend calls for target verify on AITER's fused FlyDSL kernel (gfx950).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention import (
    kda_fused_decode_aiter_hip,
    kda_fused_decode_gfx950,
    kda_fused_verify_aiter_hip,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.mem_cache.kv_cache_configurator import host_only_mamba_checkpoint
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.utils.log_utils import print_warning_once

if TYPE_CHECKING:
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention


class KimiK3HipFusedVerify:
    """Dense chain verify of one Kimi-K3 KDA layer whose f_b the model defers:
    f_b, convolution, gate, recurrence and gated RMSNorm in one kernel."""

    def __init__(
        self,
        *,
        f_b_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        a_log: torch.Tensor,
        num_draft_tokens: int,
    ):
        self.f_b_weight = f_b_weight
        self.norm_weight = norm_weight
        self.norm_eps = norm_eps
        self.a_log = a_log
        # The warmed draft-token count is the only one the kernel may run.
        self.num_draft_tokens = num_draft_tokens

    def forward(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        intermediate_state_cache: Optional[torch.Tensor],
        intermediate_conv_window_cache: torch.Tensor,
        num_draft_tokens: int,
        replay_ring: Optional[tuple[torch.Tensor, ...]],
        unsupported: Optional[str],
    ) -> Optional[torch.Tensor]:
        if unsupported is not None:
            kda_fused_verify_aiter_hip.warn_rejected_once(unsupported)
            return None
        tokens = mixed_qkv.shape[0]
        batch = tokens // num_draft_tokens
        kwargs = dict(
            f_a=a[0],
            mixed_qkv=mixed_qkv,
            raw_beta=b,
            output_gate=layer._k3_onorm_gate[:tokens].view(
                -1, layer.num_v_heads, layer.head_v_dim
            ),
            conv_state=conv_states.transpose(-1, -2),
            state=ssm_states,
            cache_indices=cache_indices[:batch],
            intermediate_indices=intermediate_state_indices[:batch],
            intermediate_conv_window=intermediate_conv_window_cache.transpose(-1, -2),
            intermediate_states=(
                intermediate_state_cache if replay_ring is None else None
            ),
            replay_ring=replay_ring,
            num_draft_tokens=num_draft_tokens,
        )
        if not kda_fused_verify_aiter_hip.covered(
            **kwargs, armed_draft_tokens=self.num_draft_tokens
        ):
            kda_fused_verify_aiter_hip.warn_rejected_once(
                f"tokens={tokens}, draft_tokens={num_draft_tokens} "
                f"(armed {self.num_draft_tokens}), "
                f"conv_state={conv_states.dtype}, "
                f"cache_indices={cache_indices.dtype}, "
                f"inter_indices={intermediate_state_indices.dtype}, "
                f"state={tuple(ssm_states.shape)}/{ssm_states.dtype}"
            )
            return None
        core_attn_out = kda_fused_verify_aiter_hip.run(
            **kwargs,
            f_b_weight=self.f_b_weight,
            conv_weight=layer.conv_weights,
            A_log=self.a_log,
            dt_bias=layer.dt_bias,
            lower_bound=float(layer.lower_bound),
            norm_weight=self.norm_weight,
            norm_eps=self.norm_eps,
        )
        layer._k3_onorm_consumed = True
        return core_attn_out

    def unfused_gate(self, a: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

        gate = kimi_k3_tiny_gemm(a[0], self.f_b_weight.flatten(0, 1))
        return gate.unflatten(-1, self.f_b_weight.shape[:2]).unsqueeze(0)


def build_fused_verify_hip(
    *,
    layer: RadixLinearAttention,
    use_full_rank_gate: bool,
    state_dtype: torch.dtype,
) -> Optional[KimiK3HipFusedVerify]:
    """The AITER fused KDA verify hook over the fused decode's static inputs
    (``layer._k3_hip_fused_decode_args``), warmed for this server's draft-token
    count; None when the kernel cannot cover this layer or server."""
    # Only the full-rank gate projection can hand f_a over unprojected.
    if not use_full_rank_gate:
        return None
    if (get_spec().speculative_eagle_topk or 1) > 1:
        print_warning_once(
            "Kimi-K3 fused KDA verify stays off: it covers chain verify only, "
            "not tree speculative decoding (speculative_eagle_topk > 1)"
        )
        return None
    replay_ring = get_exec().mamba.enable_linear_replayssm_spec
    if replay_ring and host_only_mamba_checkpoint():
        print_warning_once(
            "Kimi-K3 fused KDA verify stays off: host-only mamba checkpoints "
            "use the request-indexed ReplaySSM ring"
        )
        return None
    num_draft_tokens = get_spec().speculative_num_draft_tokens or 0
    f_b_weight, norm_weight, norm_eps, a_log = layer._k3_hip_fused_decode_args
    if not kda_fused_verify_aiter_hip.available(
        device=f_b_weight.device,
        state_dtype=state_dtype,
        num_draft_tokens=num_draft_tokens,
    ):
        print_warning_once(
            "Kimi-K3 fused KDA verify stays off: AITER's kernel does not cover "
            f"this device, a {state_dtype} state or {num_draft_tokens} draft "
            "tokens; verify runs the Triton chain"
        )
        return None
    kda_fused_verify_aiter_hip.warmup(
        f_b_weight=f_b_weight,
        conv_weight=layer.conv_weights,
        A_log=a_log,
        dt_bias=layer.dt_bias,
        lower_bound=float(layer.lower_bound),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        state_dtype=state_dtype,
        num_draft_tokens=num_draft_tokens,
        replay_ring=replay_ring,
    )
    return KimiK3HipFusedVerify(
        f_b_weight=f_b_weight,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        a_log=a_log,
        num_draft_tokens=num_draft_tokens,
    )


def prepare_fused_kda_hip(attention: KimiK3DeltaAttention) -> None:
    """ROCm half of KimiK3DeltaAttention._prepare_fused_decode: arm the AITER or
    gfx950 fused decode, and the AITER fused verify hook (layer.kda_fused_verify)
    under --linear-attn-verify-backend aiter."""
    layer = attention.attn
    # A failed re-arm must not leave the previous f_b serving decode or verify.
    attention._kda_hip_fused_decode_ready = False
    layer._k3_hip_fused_decode_args = None
    layer._k3_hip_fused_decode_backend = ""
    layer._k3_hip_fused_decode_ops = None
    layer.kda_fused_verify = None
    w = layer.conv_weights
    backend = envs.SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND.get().lower()
    verify_selected = (
        get_exec().mamba.linear_attn_verify_backend
        == LinearAttnKernelBackend.AITER.value
    )
    if verify_selected and backend != "aiter":
        print_warning_once(
            "Kimi-K3 fused KDA verify stays off: --linear-attn-verify-backend "
            "aiter needs SGLANG_USE_HIP_K3_KDA_FUSED_BACKEND=aiter; verify "
            "runs the Triton chain"
        )
    fused_ops = {
        "aiter": kda_fused_decode_aiter_hip,
        "gfx950": kda_fused_decode_gfx950,
    }.get(backend)
    if fused_ops is None:
        return
    # Either ReplaySSM flag allocates the ring (kv_cache_configurator.py),
    # whose decode skips the fused kernel and needs f_b already applied.
    mamba = get_exec().mamba
    replayssm_ring = mamba.enable_linear_replayssm or mamba.enable_linear_replayssm_spec
    if replayssm_ring:
        print_warning_once(
            "Kimi-K3 fused KDA decode stays off: ReplaySSM uses the unfused ring decode"
        )
        if backend != "aiter" or not verify_selected:
            return
    state_dtype = attention.config.mamba2_cache_params.dtype.temporal
    backend_available = fused_ops.available(
        device=attention.f_b_proj.weight.device, state_dtype=state_dtype
    )
    f_b_weight = attention._fused_decode_f_b_weight() if backend_available else None
    if (
        backend_available
        and w is not None
        and tuple(w.shape) == (3 * 12 * 128, 4)
        and w.dtype == torch.float32
        and f_b_weight is not None
        and f_b_weight.shape == (12 * 128, 128)
        and f_b_weight.dtype == torch.bfloat16
        and layer.A_log is not None
        and layer.A_log.numel() == 12
        and layer.A_log.dtype == torch.float32
        and layer.dt_bias is not None
        and tuple(layer.dt_bias.shape) == (12 * 128,)
        and layer.dt_bias.dtype == torch.float32
        and layer.lower_bound is not None
    ):
        norm_weight = attention.o_norm.weight.data.to(torch.bfloat16).contiguous()
        f_b_weight = f_b_weight.view(12, 128, 128).contiguous()
        a_log = layer.A_log.detach().reshape(-1).contiguous()
        layer._k3_hip_fused_decode_args = (
            f_b_weight,
            norm_weight,
            float(attention.o_norm.eps),
            a_log,
        )
        if not replayssm_ring:
            fused_ops.warmup(
                f_b_weight=f_b_weight,
                conv_weight=w,
                A_log=a_log,
                dt_bias=layer.dt_bias,
                lower_bound=float(layer.lower_bound),
                norm_weight=norm_weight,
                norm_eps=float(attention.o_norm.eps),
                state_dtype=state_dtype,
            )
            layer._k3_hip_fused_decode_backend = backend
            layer._k3_hip_fused_decode_ops = fused_ops
            attention._kda_hip_fused_decode_ready = True
        if backend == "aiter" and verify_selected:
            layer.kda_fused_verify = build_fused_verify_hip(
                layer=layer,
                use_full_rank_gate=attention.use_full_rank_gate,
                state_dtype=state_dtype,
            )
