"""PLA operator wrappers behind the SAIL KDA fast paths.

``pla`` is the PPU linear-attention kernel library shipped as its own wheel.
Every call site here is gated by ``SGLANG_SAIL_PLA`` (turned on by
``arg_groups.platform_hook.handle_sail_backends`` when ``pla`` imports), so a
PLA-less installation and every non-SAIL backend keep the community
Triton/CUDA implementations in ``sglang.kernels``. Keeping the vendor calls in
this module is what lets the upstream kernel files stay platform-free.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs

# The packed-decode operator hard-codes the community softplus constants.
PLA_SOFT_BETA = 1.0
PLA_SOFT_THRESHOLD = 20.0

# Per-rank head counts of the PLA KDA decode kernel.
SUPPORTED_FUSED_DECODE_HEADS = {12, 24, 48, 96}


def pla_enabled() -> bool:
    """Whether the PLA operator library may be used for this process."""
    return envs.SGLANG_SAIL_PLA.get()


def _load_decode():
    from pla.decode import (
        fused_sigmoid_gating_delta_rule_forward_k_last,
        fused_sigmoid_gating_delta_rule_forward_k_last_packed,
    )

    return (
        fused_sigmoid_gating_delta_rule_forward_k_last,
        fused_sigmoid_gating_delta_rule_forward_k_last_packed,
    )


def fused_decode_covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    onorm_g: torch.Tensor,
) -> bool:
    """Whether the PLA fused decode operator takes this decode batch."""
    if not pla_enabled() or ssm_states.ndim < 4:
        return False
    return ssm_states.shape[-3] in SUPPORTED_FUSED_DECODE_HEADS


def fused_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    onorm_eps: float,
    fused_weight: torch.Tensor,
    actual_batch_size: Optional[int],
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """Fused conv + recurrence + gated RMSNorm decode step on PLA.

    Returns the packed-decode output layout ``[1, B, HV, V]``. Rows past
    ``actual_batch_size`` are padding rows of a graph-shaped batch; their
    output is zeroed and their state slots are left alone.
    """
    from pla.decode.kda import fused_kda_decode_mega_forward

    B = mixed_qkv.shape[0]
    H = ssm_states.shape[-3]
    if actual_batch_size is None:
        actual_batch_size = B
    output_shape = (1, B, H, 128)
    out = (
        mixed_qkv.new_empty(output_shape)
        if actual_batch_size == B
        else mixed_qkv.new_zeros(output_shape)
    )
    fused_kda_decode_mega_forward(
        x=mixed_qkv[:actual_batch_size],
        weight=fused_weight,
        bias=conv_bias,
        conv_state=conv_states.transpose(-1, -2),
        raw_g=a[:actual_batch_size].reshape(1, actual_batch_size, H, 128),
        raw_beta=b[:actual_batch_size].unsqueeze(0),
        a_log=A_log,
        dt_bias=dt_bias,
        state_indices=cache_indices[:actual_batch_size],
        state=ssm_states,
        out=out[:, :actual_batch_size],
        lower_bound=lower_bound,
        output_gate=onorm_g[:actual_batch_size].reshape(actual_batch_size, H, 128),
        norm_weight=onorm_weight,
        norm_eps=onorm_eps,
    )
    return out


def recurrent_update_supported(
    cache_ring: bool,
    initial_state_source: Optional[torch.Tensor],
    intermediate_states_buffer: Optional[torch.Tensor],
) -> bool:
    """Whether PLA takes this recurrent decode/verify batch.

    The PLA operator has no ReplaySSM ring-write contract and only carries the
    fp32 state pool the KDA checkpoints use; anything else stays on Triton.
    """
    if not pla_enabled() or cache_ring:
        return False
    return all(
        t is None or t.dtype == torch.float32
        for t in (initial_state_source, intermediate_states_buffer)
    )


def recurrent_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    lower_bound: Optional[float] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Recurrent delta-rule decode / target-verify step on PLA."""
    fused_forward_k_last, _ = _load_decode()

    def _gate_cast(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # The operator builds the softplus gate from A_log/dt_bias on the
        # state-update path; it wants them in the activation dtype there and
        # keeps the verify (state-read) path's own dtype.
        if t is None:
            return None
        return t if disable_state_update else t.to(q.dtype)

    kda_lower_bound = (
        (lower_bound if lower_bound is not None else 0.0) if is_kda else None
    )
    return fused_forward_k_last(
        _gate_cast(A_log),
        a,
        _gate_cast(dt_bias),
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
        scale,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        is_kda,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        retrieve_parent_token,
        kda_lower_bound,
    )


def packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed single-token decode on PLA; returns ``(out, final_state)``."""
    _, fused_packed = _load_decode()

    B = mixed_qkv.shape[0]
    K = initial_state.shape[-1]
    out = fused_packed(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        PLA_SOFT_BETA,
        PLA_SOFT_THRESHOLD,
        K**-0.5,
        initial_state,
        ssm_state_indices,
        out,
        True,
        torch.arange(B + 1, device=a.device, dtype=torch.int32),
        False,
    )
    return out, initial_state


def prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: Optional[float],
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    """FlashKDA-style chunked prefill on PLA (writes ``out``/``final_state``)."""
    from pla.prefill.flashkdapro import flashkda_fwd

    # q/k carry H heads; v, g and the output carry HV. They agree for the
    # checkpoints at hand, so read each tensor's own extents rather than
    # reusing q's.
    tokens = q.shape[0] * q.shape[1]
    num_q_heads, qk_dim = q.shape[2], q.shape[3]
    num_v_heads, v_dim = v.shape[2], v.shape[3]
    flashkda_fwd(
        q=q.view(tokens, num_q_heads, qk_dim),
        k=k.view(tokens, num_q_heads, qk_dim),
        v=v.view(tokens, num_v_heads, v_dim),
        g=g.view(tokens, num_v_heads, g.shape[3]),
        beta=beta.view(tokens, num_q_heads),
        A_log=A_log,
        dt_bias=dt_bias,
        out=out.view(tokens, num_v_heads, v_dim),
        scale=scale,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )


def decode_mtp(
    *,
    layer,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    intermediate_state_cache: torch.Tensor,
    intermediate_conv_window_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DSpark MTP / fused chain verify on PLA (direct g/beta layout)."""
    from pla.decode import kda_mtp_sglang

    seq_len = mixed_qkv.shape[0]
    h = layer.num_v_heads
    x_q, x_k, x_v = mixed_qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
    x_q = x_q.reshape(1, seq_len, h, layer.head_q_dim)
    x_k = x_k.reshape(1, seq_len, h, layer.head_k_dim)
    x_v = x_v.reshape(1, seq_len, h, layer.head_v_dim)
    w_q, w_k, w_v = layer.conv_weights.split(
        [layer.q_dim, layer.k_dim, layer.v_dim], dim=0
    )
    cs_q, cs_k, cs_v = conv_states.split(
        [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
    )
    cs_q = cs_q.transpose(-1, -2)
    cs_k = cs_k.transpose(-1, -2)
    cs_v = cs_v.transpose(-1, -2)
    intermediate_conv_window_cache = intermediate_conv_window_cache.transpose(-1, -2)
    ic_q, ic_k, ic_v = intermediate_conv_window_cache.split(
        [layer.q_dim, layer.k_dim, layer.v_dim], dim=-2
    )
    # PLA keeps the gated RMSNorm inside the model: its interface takes the
    # raw gate/beta layouts and has no norm parameters.
    return kda_mtp_sglang.fused_kda_decode_mtp_dspark(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=a.view(1, seq_len, h, 128),
        beta=b.view(1, seq_len, h),
        A_log=layer.A_log.reshape(-1),
        dt_bias=layer.dt_bias.reshape(-1),
        recurrent_state=ssm_states,
        intermediate_ssm=intermediate_state_cache,
        intermediate_state_indices=intermediate_state_indices,
        intermediate_conv_q=ic_q,
        intermediate_conv_k=ic_k,
        intermediate_conv_v=ic_v,
        ssm_state_indices=cache_indices.to(torch.int32),
        cu_seqlens=query_start_loc.to(torch.int32),
        lower_bound=float(layer.lower_bound),
        scale=layer.head_q_dim**-0.5,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
        onorm_gate=None,
        onorm_weight=None,
        onorm_eps=None,
    )


def supported_fused_decode_heads(community_heads: set) -> set:
    """Head counts the model-side fused decode preparation may accept.

    PLA covers a wider per-rank head set than the community kernel, and the
    preparation guard runs before any batch reaches the backend, so it has to
    see the union of both.
    """
    if not pla_enabled():
        return community_heads
    return community_heads | SUPPORTED_FUSED_DECODE_HEADS


def bind_fused_decode_weight(layer) -> None:
    """Publish the fused-decode conv weights the PLA operator consumes."""
    if not pla_enabled():
        return
    layer._k3_fused_decode_weight = torch.stack(
        layer._k3_fused_decode_args[:3]
    ).contiguous()


def commit_after_verify(
    *,
    spec_state,  # MambaPool.SpeculativeState (all layers)
    state_batch_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_steps_to_track: Optional[torch.Tensor] = None,
    src_indices_raw: Optional[torch.Tensor] = None,
    null_block_id: int = -1,
    ring_indices: Optional[torch.Tensor] = None,
) -> bool:
    """Run the ReplaySSM commit on PLA when it takes this state pool.

    The PLA operator folds the accepted prefix and rolls back conv from the
    width-major conv pool in one launch, so the caller skips its own fold and
    conv scatter. Every condition the operator needs is checked here; a False
    answer leaves the community Triton path in charge.
    """
    if not pla_enabled() or ring_indices is not None or src_indices_raw is not None:
        return False
    if spec_state.temporal.dtype != torch.float32:
        return False
    if spec_state.temporal.shape[-2:] != (128, 128):
        return False
    if spec_state.replayssm_rawv.dtype != torch.bfloat16:
        return False
    if spec_state.replayssm_rawk.dtype != torch.bfloat16:
        return False
    if not 1 <= spec_state.replayssm_rawv.shape[-2] <= 32:
        return False
    if not all(
        t.is_contiguous()
        for t in (
            spec_state.temporal,
            spec_state.replayssm_rawv,
            spec_state.replayssm_rawk,
            spec_state.replayssm_g,
            spec_state.replayssm_beta,
        )
    ):
        return False
    from pla.decode import kda_mtp_sglang

    kda_mtp_sglang.commit_kda_replayssm_after_verify(
        spec_state=spec_state,
        state_batch_indices=state_batch_indices,
        accept_lens=accept_lens,
        last_correct_step_indices=last_correct_step_indices,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
        conv_state_layout="width_dim",
    )
    return True
