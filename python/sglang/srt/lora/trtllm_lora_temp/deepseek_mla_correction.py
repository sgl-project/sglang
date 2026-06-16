"""LoRA correction for absorbed-MLA ``kv_b_proj``.

The absorbed-MLA path in ``DeepseekV2AttentionMLA`` bypasses
``kv_b_proj.forward()`` and folds the K/V contribution into two BMMs against
the pre-computed ``w_kc`` / ``w_vc`` weights, so a standard
``ColumnParallelLinearWithLoRA`` wrapper would never see the activations and
the LoRA delta would silently be dropped. These helpers inject the missing
delta on top of the absorbed intermediates via the SGMM-style Triton kernels
in ``triton_ops/kv_b_lora_absorbed.py``.

Used from ``deepseek_common/attention_forward_methods/forward_mla.py``. Call
sites should gate the call with :func:`is_kv_b_lora_active` so non-LoRA
forwards take a single ``getattr`` and skip the helper entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

# The four step kernels live in triton_ops; importing it pulls the LoRA kernel
# modules (and specialized_expand) into the process. They are only ever reached
# after _get_state returns a non-None state (a kv_b LoRA adapter is wrapped), so
# defer the import to that success path: a no-LoRA forward never imports it here.
step_a_q_fwd = step_a_v_fwd = step_b_q_fwd = step_b_v_fwd = None


def _ensure_step_kernels() -> None:
    global step_a_q_fwd, step_a_v_fwd, step_b_q_fwd, step_b_v_fwd
    if step_a_q_fwd is None:
        from sglang.srt.lora.trtllm_lora_temp.triton_ops import step_a_q_fwd as _aq
        from sglang.srt.lora.trtllm_lora_temp.triton_ops import step_a_v_fwd as _av
        from sglang.srt.lora.trtllm_lora_temp.triton_ops import step_b_q_fwd as _bq
        from sglang.srt.lora.trtllm_lora_temp.triton_ops import step_b_v_fwd as _bv

        step_a_q_fwd, step_a_v_fwd, step_b_q_fwd, step_b_v_fwd = _aq, _av, _bq, _bv


if TYPE_CHECKING:
    from sglang.srt.lora.utils import LoRABatchInfo
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


def is_kv_b_lora_active(attn_module: DeepseekV2AttentionMLA) -> bool:
    """Cheap precondition check used at call sites in the attention forward
    to skip the entire LoRA-correction path when no ``kv_b_proj`` adapter is
    wrapped on this module (the common case)."""
    return getattr(attn_module.kv_b_proj, "set_lora", False)


def _get_state(
    attn_module: DeepseekV2AttentionMLA,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, LoRABatchInfo]]:
    if not is_kv_b_lora_active(attn_module):
        return None
    if not hasattr(attn_module.kv_b_proj, "A_buffer"):
        return None
    lora_backend = attn_module.kv_b_proj.lora_backend
    if not hasattr(lora_backend, "batch_info"):
        return None
    batch_info = lora_backend.batch_info
    if batch_info is None:
        return None

    # Triton backend exposes _sgemm_info() to group decode-shape repeats of
    # the same adapter; csgmv-style backends just expose batch_info directly.
    sgemm_info = getattr(lora_backend, "_sgemm_info", None)
    if callable(sgemm_info):
        batch_info = sgemm_info()
    # Non-None state ⇒ a kv_b adapter is active here; load the step kernels now
    # (cached after the first active forward). No-LoRA forwards return above and
    # never import triton_ops.
    _ensure_step_kernels()
    return attn_module.kv_b_proj.A_buffer, attn_module.kv_b_proj.B_buffer, batch_info


def apply_q_correction(
    attn_module: DeepseekV2AttentionMLA,
    q_nope: torch.Tensor,
    q_nope_out: torch.Tensor,
) -> torch.Tensor:
    """LoRA correction for the absorbed ``q_nope @ w_kc`` path.

    Computes ``q_nope_out += q_nope @ B_kc @ A * scaling`` per token, per
    active LoRA slot via two SGMM-style Triton kernels. Factored along the
    LoRA-A/B boundary so we never materialise ``B @ A`` (~268M FMAs per layer
    per slot in the naive implementation)::

      step A_q : ``(S,H,qk_nope) @ B_kc[slot, h] (qk_nope, rank) -> (S,H,rank)``
      step B_q : ``(S,H,rank)    @ A[slot] (rank, kv_lora_rank)  -> += q_nope_out``
    """
    state = _get_state(attn_module)
    if state is None:
        return q_nope_out
    A_buf, B_buf, batch_info = state

    full_K_per_head = attn_module.qk_nope_head_dim + attn_module.v_head_dim
    q_lora_a = step_a_q_fwd(q_nope, B_buf, batch_info, full_K_per_head)
    return step_b_q_fwd(q_lora_a, A_buf, batch_info, q_nope_out)


def apply_v_correction(
    attn_module: DeepseekV2AttentionMLA,
    attn_output: torch.Tensor,
    attn_bmm_flat: torch.Tensor,
) -> torch.Tensor:
    """LoRA correction for the absorbed ``attn_output @ w_vc`` path.

    Computes ``attn_bmm_flat += attn_output @ A.T @ B_vc.T * scaling`` per
    token, per active LoRA slot. ``attn_bmm_flat`` is the flat
    ``(S, H*v_head_dim)`` view of the absorbed BMM result; we pass strides
    matching the implicit ``(S, H, v_head_dim)`` layout to step B_v.
    """
    state = _get_state(attn_module)
    if state is None:
        return attn_bmm_flat
    A_buf, B_buf, batch_info = state

    attn_lora_a = step_a_v_fwd(attn_output, A_buf, batch_info)
    base_view = attn_bmm_flat.view(
        -1, attn_module.num_local_heads, attn_module.v_head_dim
    )
    step_b_v_fwd(
        attn_lora_a,
        B_buf,
        batch_info,
        base_view,
        attn_module.qk_nope_head_dim,
        attn_module.v_head_dim,
    )
    return attn_bmm_flat


# ---------------------------------------------------------------------------
# Two-stream overlap (O12) for the absorbed kv_b correction.
#
# Each correction factors into an input-only A-step (reads q_nope / attn_output,
# independent of the absorbed bmm output) and a B-step that adds into that bmm
# output. ``*_prepare`` forks the A-step onto the shared LoRA side stream so it
# overlaps the main-stream ``q_nope @ w_kc`` / ``attn_output @ w_vc`` bmm;
# ``*_apply`` rejoins and runs the B-step.
#
# Gated by ``SGLANG_LORA_TWO_STREAM`` (decode batches only) via
# ``is_two_stream_active``. When inactive, ``*_prepare`` returns None and
# ``*_apply`` falls back to the serial ``apply_*_correction`` (or a no-op when no
# kv_b adapter is wrapped), so the deepseek call sites stay byte-identical with
# two-stream off. Same fork/join (``wait_stream``) idiom as the O7/O8 attention
# overrides — cuda-graph-capture safe.
# ---------------------------------------------------------------------------


def _kv_b_two_stream_state(attn_module, x):
    from sglang.srt.lora.trtllm_lora_temp import (
        get_lora_side_stream,
        is_two_stream_active,
    )

    if not is_two_stream_active(x):
        return None
    state = _get_state(attn_module)
    if state is None:
        return None
    A_buf, B_buf, batch_info = state
    return A_buf, B_buf, batch_info, get_lora_side_stream()


def kv_b_lora_q_prepare(attn_module, q_nope):
    """Fork the q-correction A-step onto the side stream (``step_a_q`` reads only
    ``q_nope``) so it overlaps the main-stream ``q_nope @ w_kc`` bmm. Returns a
    handle for :func:`kv_b_lora_q_apply`, or None when two-stream is inactive."""
    st = _kv_b_two_stream_state(attn_module, q_nope)
    if st is None:
        return None
    A_buf, B_buf, batch_info, side_stream = st
    full_K_per_head = attn_module.qk_nope_head_dim + attn_module.v_head_dim
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        q_lora_a = step_a_q_fwd(q_nope, B_buf, batch_info, full_K_per_head)
    return q_lora_a, A_buf, batch_info, side_stream


def kv_b_lora_q_apply(attn_module, q_nope, q_nope_out, handle):
    """Finish the q-correction: two-stream (rejoin + B-step) when ``handle`` is
    set, else the serial correction, else a no-op. Single call replacing the
    ``if is_kv_b_lora_active: apply_q_correction`` at the call site."""
    if handle is not None:
        q_lora_a, A_buf, batch_info, side_stream = handle
        torch.cuda.current_stream().wait_stream(side_stream)
        return step_b_q_fwd(q_lora_a, A_buf, batch_info, q_nope_out)
    if is_kv_b_lora_active(attn_module):
        return apply_q_correction(attn_module, q_nope, q_nope_out)
    return q_nope_out


def kv_b_lora_v_prepare(attn_module, attn_output):
    """Fork the v-correction A-step onto the side stream (``step_a_v`` reads only
    ``attn_output``) so it overlaps the main-stream ``attn_output @ w_vc`` bmm.
    Returns a handle for :func:`kv_b_lora_v_apply`, or None when inactive."""
    st = _kv_b_two_stream_state(attn_module, attn_output)
    if st is None:
        return None
    A_buf, B_buf, batch_info, side_stream = st
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        attn_lora_a = step_a_v_fwd(attn_output, A_buf, batch_info)
    return attn_lora_a, B_buf, batch_info, side_stream


def kv_b_lora_v_apply(attn_module, attn_output, attn_bmm_flat, handle):
    """Finish the v-correction: two-stream (rejoin + B-step) when ``handle`` is
    set, else the serial correction, else a no-op."""
    if handle is not None:
        attn_lora_a, B_buf, batch_info, side_stream = handle
        torch.cuda.current_stream().wait_stream(side_stream)
        base_view = attn_bmm_flat.view(
            -1, attn_module.num_local_heads, attn_module.v_head_dim
        )
        step_b_v_fwd(
            attn_lora_a,
            B_buf,
            batch_info,
            base_view,
            attn_module.qk_nope_head_dim,
            attn_module.v_head_dim,
        )
        return attn_bmm_flat
    if is_kv_b_lora_active(attn_module):
        return apply_v_correction(attn_module, attn_output, attn_bmm_flat)
    return attn_bmm_flat
