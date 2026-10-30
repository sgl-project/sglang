"""Cake (FlashInfer) backends for linear attention (KDA, GDN) in ``attention``.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels.attention_linear_kda`,
:mod:`sglang.kernels.cake_kernels.attention_linear_kda_prefill` and
:mod:`sglang.kernels.cake_kernels.attention_linear_gdn`, which import
FlashInfer only when a kernel is actually called. Callers gate on the
adapters' ``supports_*`` predicates before using the ``cake_*`` entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_KDA = "sglang.kernels.cake_kernels.attention_linear_kda"
_KDA_PREFILL = "sglang.kernels.cake_kernels.attention_linear_kda_prefill"
_GDN = "sglang.kernels.cake_kernels.attention_linear_gdn"
_SM100_SM103 = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})

for _op, _target, _signature, _description in (
    (
        "attention.kda_recurrent",
        f"{_KDA}:recurrent_kda",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "KDA recurrent decode (T=1..6, D128, BF16 pool [N,HV,128,128] "
                "updated in place) and frozen FlashKDA prefill (BF16 [B,T>1,H,128], "
                "BF16 beta logits, BF16/FP32 indexed pool, BF16 checkpoints)"
            ),
        ),
        "Cake recurrent KDA decode/prefill (flashinfer.kda.recurrent_kda, "
        "backend='cake') distributed by FlashInfer.",
    ),
    (
        "attention.kda_packed_decode",
        f"{_KDA}:packed_kda_decode",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "packed Kimi-K3 T=1 decode: BF16 mixed_qkv [B,3*12*128], raw gate "
                "[B,1536], raw beta [B,12], BF16 pool [N,12,128,128] in place; "
                "H=12, K=V=128, lower_bound=-5"
            ),
        ),
        "Cake packed KDA decode (flashinfer.kda_decode.packed_kda_decode) "
        "distributed by FlashInfer.",
    ),
    (
        "attention.kda_fused_decode",
        f"{_KDA}:fused_kda_decode",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "fused Kimi conv + recurrent KDA + RMSNorm decode: BF16 x "
                "[rows,3*H*128], H in {8,12,24,32,48,96}; conv_state and BF16/FP32 "
                "state pool updated in place"
            ),
        ),
        "Cake fused KDA decode (flashinfer.kda_decode.fused_kda_decode, "
        "backend='cake') distributed by FlashInfer.",
    ),
    (
        "attention.kda_prefill_prepare_bf16",
        f"{_KDA_PREFILL}:prepare_bf16_kda_prefill",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "prepared BF16 KDA prefill export: BF16 q/k/v/out [B,T,H,128], "
                "FP32 external state pool, BF16 (or exported FP32) checkpoints; "
                "returns a launch object (.launch() on the current stream)"
            ),
        ),
        "Cake prepared BF16 KDA prefill (flashinfer.kda_prefill."
        "prepare_bf16_kda_prefill) distributed by FlashInfer.",
    ),
    (
        "attention.kda_prefill_prepare_tf32",
        f"{_KDA_PREFILL}:prepare_tf32_kda_prefill",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "prepared TF32-compute KDA prefill export: BF16 q/k/v/out, FP32 "
                "external state, BF16 checkpoints; bounded gate for active beta"
            ),
        ),
        "Cake prepared TF32 KDA prefill (flashinfer.kda_prefill."
        "prepare_tf32_kda_prefill) distributed by FlashInfer.",
    ),
    (
        "attention.kda_prefill_plan_cache",
        f"{_KDA_PREFILL}:kda_prefill_plan_cache",
        FormatSignature(
            description=(
                "host-side LRU of prepared BF16 KDA launches; hits rebind pointers "
                "with one descriptor upload (CUDA-graph capturable)"
            ),
        ),
        "Cake KDA prefill plan cache (flashinfer.kda_prefill.KDAPrefillPlanCache).",
    ),
    (
        "attention.kda_prefill_supports_fp32_checkpoints",
        f"{_KDA_PREFILL}:kda_prefill_supports_fp32_checkpoints",
        FormatSignature(
            description=(
                "host-only registry query: FP32 checkpoint carrier exported for the "
                "device's arch and gate kind"
            ),
        ),
        "Cake KDA prefill FP32-checkpoint capability query "
        "(flashinfer.kda_prefill.kda_prefill_supports_fp32_checkpoints).",
    ),
    (
        "attention.gdn_chunk_gated_delta_rule",
        f"{_GDN}:chunk_gated_delta_rule",
        FormatSignature(
            supported_dtypes=("bfloat16", "float16"),
            in_place=True,
            description=(
                "GDN chunked prefill: rank-3 [tokens,heads,128] q/k/v/output, FP32 "
                "alpha/beta [tokens,H], int32/int64 cu_seqlens, K-last state pool "
                "[N,H,128,128] indexed in place, checkpoints every 64n tokens; "
                "use_cp=True -> four-stage CP route"
            ),
        ),
        "Cake GDN prefill (flashinfer.gdn_prefill.chunk_gated_delta_rule, "
        "backend='cake_gdn') distributed by FlashInfer.",
    ),
    (
        "attention.gdn_cp_prefill_prepare",
        f"{_GDN}:prepare_gdn_cp_prefill",
        FormatSignature(
            supported_dtypes=("bfloat16", "float16"),
            in_place=True,
            description=(
                "prepared GDN CP prefill (GDNCPPrefill with replay()/"
                "launch_with_bindings()); runs once eagerly and optionally captures "
                "an internal fixed-address graph"
            ),
        ),
        "Cake prepared GDN CP prefill (flashinfer.gdn_kernels.blackwell."
        "cake_gdn_cp_backend.prepare_gdn_cp_prefill) distributed by FlashInfer.",
    ),
    (
        "attention.gdn_decode_pretranspose",
        f"{_GDN}:gated_delta_rule_decode_pretranspose",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "GDN decode / MTP verify on an indexed K-last pool [pool,HV,128,128] "
                "(BF16 or FP32): BF16 [B,T,H,128] q/k, [B,T,HV,128] v, BF16 gates, "
                "FP32 A_log/dt_bias, int32 [B] indices, optional [B,>=T,HV,128,128] "
                "intermediate buffer"
            ),
        ),
        "Cake GDN pretranspose decode (flashinfer.gdn_decode."
        "gated_delta_rule_decode_pretranspose, backend='cake_gdn') distributed "
        "by FlashInfer.",
    ),
    (
        "attention.gdn_decode",
        f"{_GDN}:gated_delta_rule_decode",
        FormatSignature(
            supported_dtypes=("bfloat16",),
            in_place=True,
            description=(
                "GDN T=1 decode on a K-major FP32 state [B,HV,128,128] updated in "
                "place; exact contiguous BF16 inputs"
            ),
        ),
        "Cake GDN K-major decode (flashinfer.gdn_decode.gated_delta_rule_decode, "
        "backend='cake_gdn') distributed by FlashInfer.",
    ),
):
    register_kernel(
        KernelSpec(
            op=_op,
            backend=KernelBackend.FLASHINFER,
            target=_target,
            capabilities=_SM100_SM103,
            format_signature=_signature,
            description=_description,
        )
    )
del _op, _target, _signature, _description


def cake_kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_spec_tokens: Optional[int] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    seq_order: Optional[torch.Tensor] = None,
    prefill_workspace=None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_state_indices: Optional[torch.Tensor] = None,
    *,
    disable_state_update: bool = False,
    correction_cache: Optional[torch.Tensor] = None,
    kg_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Explicit Cake entry point; gate on ``supports_kda_recurrent_{decode,prefill}``."""
    return get_kernel("attention.kda_recurrent", KernelBackend.FLASHINFER)(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=num_accepted_tokens,
        output=output,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        beta_is_logit=beta_is_logit,
        seq_order=seq_order,
        prefill_workspace=prefill_workspace,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        checkpoint_state_indices=checkpoint_state_indices,
        disable_state_update=disable_state_update,
        correction_cache=correction_cache,
        kg_cache=kg_cache,
    )


def cake_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; gate on ``supports_kda_packed_decode``."""
    return get_kernel("attention.kda_packed_decode", KernelBackend.FLASHINFER)(
        mixed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
        output=output,
    )


def cake_kda_fused_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: Optional[float] = -5.0,
    norm_eps: float = 1e-5,
    output: Optional[torch.Tensor] = None,
    *,
    state_indices_mode: Literal[
        "positive_unique", "unique_or_null", "repeated_positive"
    ],
) -> torch.Tensor:
    """Explicit Cake entry point; gate on ``supports_kda_fused_decode``."""
    return get_kernel("attention.kda_fused_decode", KernelBackend.FLASHINFER)(
        x,
        weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        output=output,
        state_indices_mode=state_indices_mode,
    )


def cake_kda_prefill_prepare_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = -5.0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[Sequence[int]] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    beta_is_logit: bool = True,
    plan_cache=None,
):
    """Explicit Cake entry point; gate on ``supports_kda_prepared_prefill``."""
    return get_kernel("attention.kda_prefill_prepare_bf16", KernelBackend.FLASHINFER)(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        scale=scale,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        sequence_lengths=sequence_lengths,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        beta_is_logit=beta_is_logit,
        plan_cache=plan_cache,
    )


def cake_kda_prefill_prepare_tf32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = -5.0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[Sequence[int]] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    beta_is_logit: bool = True,
):
    """Explicit Cake entry point; gate on ``supports_kda_prepared_prefill``."""
    return get_kernel("attention.kda_prefill_prepare_tf32", KernelBackend.FLASHINFER)(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        scale=scale,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        sequence_lengths=sequence_lengths,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        beta_is_logit=beta_is_logit,
    )


def cake_kda_prefill_plan_cache(capacity: int = 64, max_bytes: Optional[int] = None):
    """Host-side plan cache for :func:`cake_kda_prefill_prepare_bf16`."""
    return get_kernel("attention.kda_prefill_plan_cache", KernelBackend.FLASHINFER)(
        capacity=capacity, max_bytes=max_bytes
    )


def cake_kda_prefill_supports_fp32_checkpoints(
    device=None, *, lower_bound: Optional[float] = None
) -> bool:
    """Host-only query: FP32 checkpoint carrier exported for this device/gate kind."""
    return get_kernel(
        "attention.kda_prefill_supports_fp32_checkpoints", KernelBackend.FLASHINFER
    )(device, lower_bound=lower_bound)


def cake_gdn_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Union[Literal["auto"], bool] = "auto",
    state_indices: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    *,
    cp_chunk_len: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Explicit Cake entry point; gate on ``supports_gdn_chunk_gated_delta_rule``."""
    return get_kernel("attention.gdn_chunk_gated_delta_rule", KernelBackend.FLASHINFER)(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=output,
        output_state=output_state,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_cp=use_cp,
        state_indices=state_indices,
        max_seqlen=max_seqlen,
        cp_chunk_len=cp_chunk_len,
    )


def cake_gdn_cp_prefill_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    *,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: Optional[int] = None,
    max_seqlen: Optional[int] = None,
    scale: Optional[float] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    capture_graph: bool = True,
):
    """Explicit Cake entry point; gate on ``supports_gdn_chunk_gated_delta_rule``."""
    return get_kernel("attention.gdn_cp_prefill_prepare", KernelBackend.FLASHINFER)(
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        initial_state,
        output=output,
        output_state=output_state,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        capture_graph=capture_graph,
    )


def cake_gdn_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    *,
    backend: Literal["cake_gdn", "auto"] = "cake_gdn",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point; gate on ``supports_gdn_decode_pretranspose``."""
    return get_kernel("attention.gdn_decode_pretranspose", KernelBackend.FLASHINFER)(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        scale=scale,
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_state_indices=output_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=disable_state_update,
        backend=backend,
    )


def cake_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    *,
    backend: Literal["cake_gdn", "auto"] = "cake_gdn",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point; gate on ``supports_gdn_decode_nontranspose``."""
    return get_kernel("attention.gdn_decode", KernelBackend.FLASHINFER)(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        scale=scale,
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        backend=backend,
    )


__all__ = [
    "cake_kda_recurrent",
    "cake_kda_packed_decode",
    "cake_kda_fused_decode",
    "cake_kda_prefill_prepare_bf16",
    "cake_kda_prefill_prepare_tf32",
    "cake_kda_prefill_plan_cache",
    "cake_kda_prefill_supports_fp32_checkpoints",
    "cake_gdn_chunk_gated_delta_rule",
    "cake_gdn_cp_prefill_prepare",
    "cake_gdn_decode_pretranspose",
    "cake_gdn_decode",
]
