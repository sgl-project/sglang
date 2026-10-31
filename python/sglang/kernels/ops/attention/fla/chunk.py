# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
from einops import rearrange

from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.utils import (
    SUPPRESS_LEVEL,
    autocast_custom_fwd,
    input_guard,
    is_intel,
    tensor_cache,
)
from sglang.srt.environ import envs

if is_intel:
    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_intra,
    )

CHUNK_SIZE = 64
# Tokens per part of a pipelined scan (see chunk_gated_delta_rule_fwd_pipelined),
# at most 4 parts; a sequence shorter than 2 parts runs in one pass.
PIPELINE_PART_TOKENS = 4096
PIPELINE_MAX_PARTS = 4

_side_streams = {}


@tensor_cache
def _cu_seqlens_host(cu_seqlens: torch.Tensor) -> list:
    return cu_seqlens.tolist()


def _pipeline_plan(q, cu_seqlens, initial_state, initial_state_indices, inplace_update):
    """(parts, sequence length) for a pipelined scan, or (0, 0) to run one pass.

    Only a single sequence whose state goes through the pool in place qualifies:
    the parts hand the state over through the pool slot. Not while a CUDA graph
    is captured (the plan needs the length on the host).
    """
    if (
        not envs.SGLANG_GDN_CHUNK_PIPELINE.get()
        or not q.is_cuda
        or cu_seqlens is None
        or cu_seqlens.numel() != 2
        or initial_state is None
        or initial_state_indices is None
        or not inplace_update
        or torch.cuda.is_current_stream_capturing()
    ):
        return 0, 0
    seq_len = _cu_seqlens_host(cu_seqlens)[1]
    parts = min(PIPELINE_MAX_PARTS, seq_len // PIPELINE_PART_TOKENS)
    return (parts, seq_len) if parts >= 2 else (0, 0)


def chunk_gated_delta_rule_fwd_pipelined(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    seq_len: int,
    parts: int,
    output: Optional[torch.Tensor] = None,
):
    """The chunk scan of one sequence (rows [0, seq_len) of the [1, T, ...]
    inputs) as `parts` runs of whole chunks over two streams.

    chunk_h is a serial recurrence over the chunks with a few programs per head
    and leaves most CUs idle. The calling stream runs it part after part; the
    state passes from one part to the next through the pool slot in fp32, as it
    stays in fp32 registers within one launch. A side stream runs the per-chunk
    kernels beside it: kkt + solve + w/u of part i + 1 and the output of part
    i - 1 while chunk_h does part i. The kernels and their per-chunk inputs are
    the same as in one pass, so the output, the per-chunk states and the final
    state are bit-identical.
    """
    T = q.shape[1]
    H, K, V = v.shape[-2], k.shape[-1], v.shape[-1]
    if (
        output is not None
        and output.shape == v.shape
        and output.dtype == v.dtype
        and output.is_contiguous()
    ):
        o = output
    else:
        o = torch.empty_like(v)
    if seq_len < T:
        o[:, seq_len:].zero_()
    nt = triton.cdiv(seq_len, CHUNK_SIZE)
    cuts = [nt * i // parts * CHUNK_SIZE for i in range(parts)] + [seq_len]
    g = chunk_local_cumsum(g[:, :seq_len], chunk_size=CHUNK_SIZE)
    h = k.new_empty(1, nt, H, V, K)

    def rows(x, i):
        return x[:, cuts[i] : cuts[i + 1]]

    def chunks(i):
        return h[:, cuts[i] // CHUNK_SIZE : triton.cdiv(cuts[i + 1], CHUNK_SIZE)]

    def intra(i):
        w, u, _ = chunk_gated_delta_rule_fwd_intra(
            k=rows(k, i), v=rows(v, i), g=rows(g, i), beta=rows(beta, i)
        )
        return w, u

    def out(i, v_new):
        chunk_fwd_o(
            q=rows(q, i),
            k=rows(k, i),
            v=v_new,
            h=chunks(i),
            g=rows(g, i),
            scale=scale,
            output=rows(o, i),
        )

    main = torch.cuda.current_stream()
    side = _side_streams.get(q.device)
    if side is None:
        side = _side_streams[q.device] = torch.cuda.Stream(device=q.device)
    side.wait_stream(main)
    wu = [intra(0)] + [None] * (parts - 1)
    wu_done = [None] * parts
    v_new = [None] * parts
    h_done = [None] * parts
    for i in range(parts):
        with torch.cuda.stream(side):
            if i + 1 < parts:
                wu[i + 1] = intra(i + 1)
                wu_done[i + 1] = side.record_event()
            if i > 0:
                side.wait_event(h_done[i - 1])
                out(i - 1, v_new[i - 1])
        if i > 0:
            main.wait_event(wu_done[i])
        w, u = wu[i]
        _, v_new[i] = chunk_gated_delta_rule_fwd_h(
            k=rows(k, i),
            w=w,
            u=u,
            g=rows(g, i),
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            inplace_update=True,
            h=chunks(i),
        )
        h_done[i] = main.record_event()
    out(parts - 1, v_new[parts - 1])
    main.wait_stream(side)
    return g, o, None, None, h, None


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: torch.LongTensor | None = None,
    inplace_update: bool = True,
    output: Optional[torch.Tensor] = None,
):
    parts, seq_len = _pipeline_plan(
        q, cu_seqlens, initial_state, initial_state_indices, inplace_update
    )
    if parts:
        return chunk_gated_delta_rule_fwd_pipelined(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            initial_state_indices,
            seq_len,
            parts,
            output=output,
        )
    g = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        inplace_update=inplace_update,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output=output,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, None, h, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        inplace_update: bool = True,
        output: Optional[torch.Tensor] = None,
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
            if cu_seqlens is not None
            else None
        )
        g, o, A, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            inplace_update=inplace_update,
            output=output,
        )
        return o.to(q.dtype), h


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    inplace_update: bool = True,
    output: Optional[torch.Tensor] = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.
        inplace_update (Optional[bool]):
            Whether to write final states back to `initial_state`. Default: `True`.
        output (Optional[torch.Tensor]):
            Contiguous buffer of `v`'s shape and dtype to write the output into (`head_first=False`); `o` is then
            this tensor. Default: `None` (a new tensor).

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, (
        "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
    )

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if (
            initial_state_indices is not None
            and initial_state_indices.shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, h = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        inplace_update,
        output,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, None, h
