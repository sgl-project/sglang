"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Vendored from flashinfer 0.6.18.dev20260807 (SM100 GDN CP prefill closure);
# pending a FlashInfer release that ships it.

import math
import warnings
from typing import Callable, Literal, Optional, Tuple, Union, cast

import torch
from flashinfer.api_logging import flashinfer_api
from flashinfer.gdn_kernels import (
    chunk_gated_delta_rule_sm90,
    chunk_gated_delta_rule_sm100,
    chunk_gated_delta_rule_sm120,
    cp_delta_rule_dsl_sm90,
    cp_delta_rule_dsl_sm120,
)
from flashinfer.trace.templates.gdn import gdn_prefill_trace
from flashinfer.utils import get_compute_capability, get_device_sm_count

from sglang.kernels.ops.attention.gdn_cp_prefill.flashinfer_compat import (
    get_device_name,
)
from sglang.kernels.ops.attention.gdn_cp_prefill.varlen_helper import (
    should_use_cp_host,
)

try:
    from sglang.kernels.ops.attention.gdn_cp_prefill.gdn_cp_prefill import (
        cp_delta_rule_dsl_sm100,
    )
except ImportError:
    cp_delta_rule_dsl_sm100 = None


_SM100_STATE_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


def _format_dtype_list(dtypes: tuple[torch.dtype, ...]) -> str:
    return ", ".join(str(dtype).removeprefix("torch.") for dtype in dtypes)


def _cp_delta_rule_rejection_reason(
    *,
    arch_major: int,
    cuda_major: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    output: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
) -> Optional[str]:
    if arch_major == 9:
        if cp_delta_rule_dsl_sm90 is None:
            return "CP delta rule SM90 DSL kernel is unavailable"
    elif arch_major == 10:
        if cuda_major < 13:
            return "CP delta rule SM100 requires CUDA 13 or newer"
        if cp_delta_rule_dsl_sm100 is None:
            return "CP delta rule SM100 DSL kernel is unavailable"
    elif arch_major == 12:
        if cp_delta_rule_dsl_sm120 is None:
            return "CP delta rule SM120 DSL kernel is unavailable"
    else:
        return "CP delta rule is currently implemented only for SM90, SM100, and SM120"
    if (
        checkpoint_every_n_tokens > 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ):
        return "CP delta rule does not support state checkpointing yet"
    if q.shape[-1] != 128:
        return f"CP delta rule only supports head_size=128, got {q.shape[-1]}"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"CP delta rule only supports fp16/bf16 inputs, got {q.dtype}"
    if k.dtype != q.dtype or v.dtype != q.dtype or output.dtype != q.dtype:
        return "CP delta rule requires q/k/v/output dtypes to match"
    for name, tensor in (("g", g), ("beta", beta)):
        if tensor is not None:
            if tensor.dtype != torch.float32:
                return f"CP delta rule requires {name} to be float32"
            if not tensor.is_contiguous():
                return f"CP delta rule requires {name} to be contiguous"
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("output", output),
    ):
        if tensor is None:
            continue
        if not tensor.is_contiguous():
            return f"CP delta rule requires {name} to be contiguous"
    if initial_state is not None:
        if state_indices is None and not initial_state.is_contiguous():
            return "CP delta rule requires initial_state to be contiguous"
        if state_indices is not None and initial_state.stride()[1:] != (
            initial_state.shape[2] * initial_state.shape[3],
            initial_state.shape[3],
            1,
        ):
            return "CP delta rule requires initial_state to be contiguous in [H, V, K]"
    return None


@flashinfer_api(trace=gdn_prefill_trace)
def chunk_gated_delta_rule(
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
    use_cp: Literal["auto"] | bool = "auto",
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule (GDN) attention for prefill.

    Implements the gated delta rule linear attention mechanism for efficient
    training and inference.  Supports both GQA (grouped query attention)
    and GVA (grouped value attention) configurations.

    Parameters
    ----------
    q : torch.Tensor
        Queries of shape ``[total_seq_len, num_q_heads, head_size]``.  Must
        be contiguous and on CUDA.
    k : torch.Tensor
        Keys of shape ``[total_seq_len, num_k_heads, head_size]``.  Must be
        contiguous and on CUDA.
    v : torch.Tensor
        Values of shape ``[total_seq_len, num_v_heads, head_size]``.  Must
        be contiguous and on CUDA.
    g : torch.Tensor, optional
        Forget gate (alpha) of shape ``[total_seq_len, num_sab_heads]``
        where ``num_sab_heads = max(num_q_heads, num_v_heads)``.  Must be
        float32.  Defaults to all ones when ``None``.
    beta : torch.Tensor, optional
        Update gate (beta) of shape ``[total_seq_len, num_sab_heads]``.
        Must be float32.  Defaults to all ones when ``None``.
    scale : float, optional
        Scale factor for the attention scores.  Defaults to
        ``1 / sqrt(head_size)`` when ``None``.
    initial_state : torch.Tensor, optional
        Initial KV state. Packed, sequence-ordered shape
        ``[num_seqs, num_sab_heads, head_size, head_size]``.  Must be
        float32 on SM90/SM120.  The SM100 path also accepts bfloat16,
        float16, float8_e4m3fn, and float8_e5m2.  Starts from zero state
        when ``None``.  When ``state_indices`` is given (SM100/SM103 only),
        this is instead the state **pool** ``[N_pool, num_sab_heads,
        head_size, head_size]`` and sequence ``i`` reads its initial state
        from row ``state_indices[i]``; the pool may be non-compact (padded
        first-dimension stride, inner ``[H, V, K]`` block contiguous).
    output_final_state : bool
        Whether to output the final state.  Default: ``False``.
    cu_seqlens : torch.Tensor
        Cumulative sequence lengths of shape ``[num_seqs + 1]``, integer
        dtype on the same CUDA device as ``q``.  Required for
        variable-length sequences (varlen mode); must not be ``None``
        (asserted at the top of the function body).  Internally cast to
        ``int32`` for the SM100/Blackwell CuTe-DSL kernel and to ``int64``
        for the SM90/Hopper C++ kernel, so the caller can pass either
        dtype.
    use_qk_l2norm_in_kernel : bool
        Whether to use QK L2 normalization in kernel.  Default: ``False``.
    output : torch.Tensor, optional
        Pre-allocated output tensor of shape
        ``[total_seq_len, num_o_heads, head_size]`` where ``num_o_heads =
        max(num_q_heads, num_v_heads)``.  Allocated automatically when
        ``None``.
    output_state : torch.Tensor, optional
        Pre-allocated output state tensor. Packed, sequence-ordered shape
        ``[num_seqs, num_sab_heads, head_size, head_size]``.  Must be float32
        on SM90/SM120.  The SM100 path also accepts bfloat16, float16,
        float8_e4m3fn, and float8_e5m2.  Required when
        ``output_final_state=True``.  When ``state_indices`` is given it is
        instead the output state **pool** ``[N_pool, ...]`` and sequence
        ``i``'s final state is written to row ``state_indices[i]`` (in place
        when ``output_state is initial_state``); it must be provided by the
        caller (auto-allocation is rejected, since a compact ``[num_seqs, ...]``
        buffer would be indexed out of bounds by the pool slot ids).
    state_checkpoints : torch.Tensor, optional
        Pre-allocated checkpoint tensor of shape ``[total_checkpoints,
        num_sab_heads, head_size, head_size]``.  Must be float32 on
        SM90/SM120.  The SM100 path also accepts bfloat16, float16,
        float8_e4m3fn, and float8_e5m2.  Required when
        ``checkpoint_every_n_tokens > 0``.
    checkpoint_cu_starts : torch.Tensor, optional
        Cumulative checkpoint counts of shape ``[num_seqs + 1]``, int64.
        ``checkpoint_cu_starts[i+1] - checkpoint_cu_starts[i]`` is the
        number of checkpoints for sequence ``i`` (= ``seq_len_i //
        checkpoint_every_n_tokens``).  Required when
        ``checkpoint_every_n_tokens > 0``.
    checkpoint_every_n_tokens : int
        Store intermediate state every N tokens.  Must be a multiple of the
        chunk size (64).  ``0`` disables checkpointing (default).
    use_cp : Literal["auto"] | bool, optional:
        Whether to use the SM90/SM120 context-parallel DSL implementation when
        low-parallelism heuristics match. ``"auto"`` enables conservative
        routing, ``True`` requires CP support, and ``False`` disables CP.
        Default: ``"auto"``.
    state_indices : torch.Tensor, optional
        Int32 tensor of shape ``[num_seqs]`` (SM100/SM103 only). When provided,
        ``initial_state`` and ``output_state`` are treated as a state pool whose
        first dimension is indexed by these slot ids rather than laid out in
        sequence order: sequence ``i`` reads its initial state from row
        ``state_indices[i]`` and writes its final state back to the same row
        (in place when ``output_state is initial_state``). This lets callers
        that keep a paged/indexed state pool avoid gathering the active rows
        into a packed buffer and scattering the result back. The pool may be
        non-compact (padded first-dimension stride). ``None`` (default) keeps
        the packed, sequence-ordered layout.

        The ids **must be unique**: as with any indexed scatter, two sequences
        sharing a slot id would concurrently write the same pool row across
        work tiles, leaving that row's final state nondeterministic. Uniqueness
        is a caller precondition (not checked at launch, to avoid a per-call
        host sync); the caller's slot allocator is expected to guarantee it.


    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        When ``output_final_state=False``, the output tensor of shape
        ``[total_seq_len, num_o_heads, head_size]``.  Otherwise a tuple
        ``(output, final_state)`` where ``final_state`` has shape
        ``[num_seqs, num_sab_heads, head_size, head_size]`` — or, when
        ``state_indices`` is given, the state pool ``[N_pool, ...]`` itself
        (i.e. ``output_state``), whose rows named by ``state_indices`` now
        hold the updated final states.

    Notes
    -----
    - Supports GQA (``num_q_heads > num_k_heads = num_v_heads``) and GVA
      (``num_v_heads > num_q_heads = num_k_heads``).
    - The final state layout is ``[N, H, V, K]``.
    - Requires SM90 (Hopper) or SM100 (Blackwell) architecture.  The SM100
      path requires ``head_size == 128`` and
      ``nvidia-cutlass-dsl[cu13]>=4.4.2`` (``pip install
      flashinfer-python[cu13]``).
    """
    if use_cp not in ("auto", True, False):
        raise ValueError(f'use_cp must be "auto", True, or False, got {use_cp!r}')
    if checkpoint_every_n_tokens < 0:
        raise ValueError(
            f"checkpoint_every_n_tokens must be non-negative, "
            f"got {checkpoint_every_n_tokens}"
        )
    if checkpoint_every_n_tokens > 0:
        if checkpoint_every_n_tokens % 64 != 0:
            raise ValueError(
                f"checkpoint_every_n_tokens must be a multiple of the chunk size (64), "
                f"got {checkpoint_every_n_tokens}"
            )
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError(
                "state_checkpoints and checkpoint_cu_starts must both be provided "
                "when checkpoint_every_n_tokens > 0"
            )
    if checkpoint_every_n_tokens == 0 and (
        state_checkpoints is not None or checkpoint_cu_starts is not None
    ):
        raise ValueError(
            "state_checkpoints and checkpoint_cu_starts must be None "
            "when checkpoint_every_n_tokens == 0"
        )

    assert cu_seqlens is not None, "cu_seqlens is required for varlen mode"

    num_seqs = cu_seqlens.size(0) - 1
    total_seq_len = q.size(0)
    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    if checkpoint_every_n_tokens > 0:
        assert state_checkpoints is not None and checkpoint_cu_starts is not None
        state_checkpoint_dtypes: tuple[torch.dtype, ...] = (torch.float32,)
        if q.is_cuda and get_compute_capability(q.device)[0] == 10:
            state_checkpoint_dtypes = _SM100_STATE_DTYPES
        if state_checkpoints.dtype not in state_checkpoint_dtypes:
            raise ValueError(
                "state_checkpoints must have dtype "
                f"{_format_dtype_list(state_checkpoint_dtypes)}, "
                f"got {state_checkpoints.dtype}"
            )
        if state_checkpoints.ndim != 4:
            raise ValueError(
                f"state_checkpoints must be 4D "
                f"[total_checkpoints, num_sab_heads, head_size, head_size], "
                f"got {state_checkpoints.ndim}D"
            )
        if checkpoint_cu_starts.dtype != torch.int64:
            raise ValueError(
                f"checkpoint_cu_starts must be int64, got {checkpoint_cu_starts.dtype}"
            )
        if checkpoint_cu_starts.ndim != 1:
            raise ValueError(
                f"checkpoint_cu_starts must be 1D [num_seqs + 1], "
                f"got {checkpoint_cu_starts.ndim}D"
            )
        if checkpoint_cu_starts.size(0) != num_seqs + 1:
            raise ValueError(
                f"checkpoint_cu_starts must have {num_seqs + 1} elements, "
                f"got {checkpoint_cu_starts.size(0)}"
            )
        expected_shape = (
            state_checkpoints.size(0),
            num_sab_heads,
            head_size,
            head_size,
        )
        if tuple(state_checkpoints.shape[1:]) != expected_shape[1:]:
            raise ValueError(
                f"state_checkpoints shape mismatch: expected "
                f"[*, {num_sab_heads}, {head_size}, {head_size}], "
                f"got {list(state_checkpoints.shape)}"
            )

    # Allocate output if not provided
    if output is None:
        output = torch.empty(
            (total_seq_len, num_o_heads, head_size),
            dtype=q.dtype,
            device=q.device,
        )

    device = q.device
    _scale = scale if scale is not None and scale != 0.0 else 1.0 / math.sqrt(head_size)

    _sm_count = get_device_sm_count(device)
    _cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    _device_capability = get_compute_capability(device)
    _arch_major = _device_capability[0]
    _device_name = get_device_name(device)
    cp_heuristic_matches = _arch_major in (9, 10, 12) and should_use_cp_host(
        num_seqs * num_sab_heads,
        _sm_count,
        _device_name,
        device_capability=_device_capability,
    )
    will_use_cp = use_cp is True or (use_cp == "auto" and cp_heuristic_matches)
    if state_indices is not None:
        # Indexed state-pool I/O is only implemented in the SM100/SM103 non-CP
        # CuTe-DSL kernel. Reject it on every other dispatch path (SM90, SM120,
        # or CP) rather than silently ignoring it and reading/writing the state
        # in packed, sequence-ordered layout.
        if _arch_major != 10:
            raise NotImplementedError(
                "state_indices is only supported on the SM100/SM103 GDN prefill "
                f"kernel (non-CP); got compute-capability major {_arch_major}, "
                f"use_cp={use_cp!r}."
            )
        # The kernel writes each final state to output_state[state_indices[i]],
        # so a compact [num_seqs, ...] auto-allocation would be indexed out of
        # bounds by arbitrary pool slot ids. Require the caller to pass the pool.
        if output_final_state and output_state is None:
            raise ValueError(
                "state_indices requires an explicit output_state pool sized like "
                "the state pool ([N_pool, H, V, K]); refusing to auto-allocate a "
                "compact [num_seqs, ...] tensor that would be indexed out of bounds."
            )
    if will_use_cp:
        cp_rejection_reason = _cp_delta_rule_rejection_reason(
            arch_major=_arch_major,
            cuda_major=_cuda_major,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output=output,
            initial_state=initial_state,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            state_indices=state_indices,
        )
        if cp_rejection_reason is not None:
            if use_cp is True:
                raise ValueError(cp_rejection_reason)
            warnings.warn(
                f"CP delta rule heuristic matched but CP dispatch is unavailable: {cp_rejection_reason}; "
                "falling back to non-CP delta rule.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if output_final_state and output_state is None:
                output_state = torch.empty(
                    (num_seqs, num_sab_heads, head_size, head_size),
                    dtype=torch.float32,
                    device=device,
                )
            _g = (
                g
                if g is not None
                else torch.ones(
                    total_seq_len, num_sab_heads, dtype=torch.float32, device=device
                )
            )
            _beta = (
                beta
                if beta is not None
                else torch.ones(
                    total_seq_len, num_sab_heads, dtype=torch.float32, device=device
                )
            )
            cp_delta_rule_dsl = cast(
                Callable[..., None],
                {
                    9: cp_delta_rule_dsl_sm90,
                    10: cp_delta_rule_dsl_sm100,
                    12: cp_delta_rule_dsl_sm120,
                }[_arch_major],
            )
            state_indices_kwargs = (
                {"state_indices": state_indices} if state_indices is not None else {}
            )
            cp_delta_rule_dsl(
                output,
                output_state,
                q,
                k,
                v,
                _g,
                _beta,
                cu_seqlens,
                _scale,
                initial_state=initial_state,
                max_seqlen=total_seq_len,
                **state_indices_kwargs,
            )
            if output_final_state:
                return output, output_state
            return output
    if _arch_major == 10:
        if _cuda_major < 13:
            raise NotImplementedError(
                "Blackwell GDN prefill is only supported on CUDA 13+"
            )
        if chunk_gated_delta_rule_sm100 is None:
            raise NotImplementedError("Blackwell GDN prefill kernel is unavailable")

        # Blackwell SM100 and SM103 path (CuTe DSL kernel)
        assert (
            head_size == 128
        ), f"Blackwell GDN prefill requires head_size=128, got {head_size}"

        # Allocate output_state only when needed
        if not output_final_state:
            output_state = None
        elif output_state is None:
            output_state = torch.empty(
                (num_seqs, num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        _g = (
            g
            if g is not None
            else torch.ones(
                total_seq_len, num_sab_heads, dtype=torch.float32, device=device
            )
        )
        _beta = (
            beta
            if beta is not None
            else torch.ones(
                total_seq_len, num_sab_heads, dtype=torch.float32, device=device
            )
        )

        # Convert checkpoint_cu_starts from int64 cu_starts to int32 cu_checkpoints
        _cu_checkpoints = None
        if checkpoint_every_n_tokens > 0 and checkpoint_cu_starts is not None:
            _cu_checkpoints = checkpoint_cu_starts.to(torch.int32)

        chunk_gated_delta_rule_sm100(
            q,
            k,
            v,
            _g,
            _beta,
            output,
            cu_seqlens.to(torch.int32),
            initial_state,
            output_state,
            _scale,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            cu_checkpoints=_cu_checkpoints,
            output_checkpoints=state_checkpoints,
            state_indices=state_indices,
        )
    elif _arch_major == 12:
        # SM120 Blackwell path (CuTe DSL kernel)
        if chunk_gated_delta_rule_sm120 is None:
            raise NotImplementedError("SM120 GDN prefill DSL kernel is unavailable")
        if output_state is None:
            output_state = torch.empty(
                (num_seqs, num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )
        chunk_gated_delta_rule_sm120(
            output,
            output_state,
            q,
            k,
            v,
            initial_state,
            g,
            beta,
            cu_seqlens.to(torch.int64),
            _scale,
            state_checkpoints,
            (
                checkpoint_cu_starts.to(torch.int64)
                if checkpoint_cu_starts is not None
                else None
            ),
            checkpoint_every_n_tokens,
        )
    elif _arch_major == 9:
        # SM90 Hopper path (CuTe DSL kernel)
        if chunk_gated_delta_rule_sm90 is None:
            raise NotImplementedError("SM90 GDN prefill DSL kernel is unavailable")

        if output_state is None:
            output_state = torch.empty(
                (num_seqs, num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        chunk_gated_delta_rule_sm90(
            output,
            output_state,
            q,
            k,
            v,
            initial_state,
            g,
            beta,
            cu_seqlens.to(torch.int64),
            _scale,
            state_checkpoints,
            (
                checkpoint_cu_starts.to(torch.int64)
                if checkpoint_cu_starts is not None
                else None
            ),
            checkpoint_every_n_tokens,
        )
    else:
        raise NotImplementedError("GDN prefill DSL kernel is unavailable")

    if output_final_state:
        return output, output_state
    else:
        return output
