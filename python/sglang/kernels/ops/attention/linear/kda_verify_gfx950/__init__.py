"""gfx950 KDA target verify with the ReplaySSM ring (conv1d + safe-gate delta rule in one
FlyDSL launch). If FlyDSL is unavailable, callers keep the Triton kernel."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    # AITER owns the FlyDSL toolchain bootstrap; import it before the kernel module.
    import aiter as _aiter  # noqa: F401
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from .kernel import create_kda_verify_kernel

    _import_error = None
except ImportError as e:
    _import_error = e

_disabled = False


def kda_verify_gfx950_available() -> bool:
    global _disabled
    if _import_error is not None and not _disabled:
        logger.warning("kda_verify_gfx950 disabled: %s", _import_error)
        _disabled = True
    return not _disabled


def default_block_v(batch_size: int) -> int:
    # MI355X, GLM-5.3-Flash TP4 (HV 16, T 6): BV32 wins up to B4, BV64 from B6 to B64.
    return 32 if batch_size <= 4 else 64


def kda_verify_gfx950_supported(
    *,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    intermediate_conv_window: Optional[torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    T: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: Optional[float],
    use_qk_l2norm_in_kernel: bool,
) -> bool:
    return (
        kda_verify_gfx950_available()
        and T in (6, 8)
        and head_k_dim == head_v_dim == 128
        and num_v_heads % num_q_heads == 0
        and lower_bound is not None
        and conv_bias is None
        and use_qk_l2norm_in_kernel
        and mixed_qkv.dtype == a.dtype == b.dtype == torch.bfloat16
        and conv_weight.dtype == torch.float32
        and ssm_states.dtype in (torch.float32, torch.bfloat16)
        and A_log.dtype == dt_bias.dtype == torch.float32
        and conv_state.dtype == torch.bfloat16
        and mixed_qkv.stride(-1) == 1
        and conv_state.stride(1) == 1
        and conv_weight.stride() == (4, 1)
        and ssm_states.is_contiguous()
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
        and (
            intermediate_conv_window is None
            or (
                intermediate_conv_window.dtype == torch.bfloat16
                and intermediate_conv_window.stride(2) == 1
            )
        )
    )


def kda_verify_gfx950(
    *,
    mixed_qkv,
    conv_weight,
    conv_state,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    a2,
    b2,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    scale,
    T,
    num_q_heads,
    num_v_heads,
    lower_bound,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    ring_indices,
    block_v: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Returns o [seq_len, HV, V], or None if the kernel is unavailable."""
    if not kda_verify_gfx950_available():
        return None
    seq_len = mixed_qkv.shape[0]
    B = seq_len // T
    HV = num_v_heads
    bvc = block_v or default_block_v(B)
    o = mixed_qkv.new_empty(seq_len, HV, 128)
    iw = intermediate_conv_window
    iw_s = iw.stride() if iw is not None else (0, 0, 0, 0)
    launch = create_kda_verify_kernel(
        T,
        bvc,
        num_q_heads,
        HV,
        float(lower_bound),
        float(scale),
        iw is not None,
        ssm_states.dtype == torch.bfloat16,
    )
    ptr = lambda t: 0 if t is None else t.data_ptr()
    _run_compiled(
        launch,
        ptr(mixed_qkv),
        ptr(conv_weight),
        ptr(conv_state),
        ptr(conv_state_indices),
        ptr(iw),
        ptr(intermediate_state_indices if iw is not None else conv_state_indices),
        ptr(a2),
        ptr(b2),
        ptr(A_log),
        ptr(dt_bias),
        ptr(ssm_states),
        ptr(cache_indices),
        ptr(o),
        ptr(replayssm_rawv),
        ptr(replayssm_rawk),
        ptr(replayssm_g),
        ptr(replayssm_beta),
        ptr(ring_indices if ring_indices is not None else cache_indices),
        mixed_qkv.stride(0),
        conv_state.stride(0),
        conv_state.stride(2),
        iw_s[0],
        iw_s[1],
        iw_s[3],
        a2.stride(0),
        b2.stride(0),
        replayssm_rawv.stride(0),
        replayssm_rawk.stride(0),
        replayssm_g.stride(0),
        replayssm_beta.stride(0),
        replayssm_rawv.shape[-2],
        B * HV,
        torch.cuda.current_stream(),
    )
    return o
