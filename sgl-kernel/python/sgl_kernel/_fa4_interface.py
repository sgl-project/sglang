# Adapted from https://github.com/Dao-AILab/flash-attention/blob/54d8aa6751fc9d5f0357854079261913d5df1f9d/flash_attn/cute/interface.py

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-10-14] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.2.1.


import copy
import gc
import logging
import math
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute import utils
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Callable | None = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    buffers: Optional[list[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert (
            page_table.stride(-1) == 1
        ), "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (
        batch_size,
    ), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (
        batch_size,
    ), "seqused_k must have shape (batch_size,)"
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert (
                t.dtype == torch.int32
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert (
                t.stride(0) == 1
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"
    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=device,
        )
    else:
        expected_out_shape = (*q_batch_seqlen_shape, num_head, head_dim_v)
        assert (
            out.shape == expected_out_shape
        ), f"out tensor shape {out.shape} does not match expected shape {expected_out_shape}"
        assert (
            out.dtype == out_torch_dtype
        ), f"out tensor dtype {out.dtype} does not match expected dtype {out_torch_dtype}"
        assert (
            out.device == device
        ), f"out tensor device {out.device} does not match input device {device}"
        assert out.is_cuda, "out tensor must be on CUDA device"

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        assert (
            lse.shape == lse_shape
        ), f"lse tensor shape {lse.shape} does not match expected shape {lse_shape}"
        assert (
            lse.dtype == torch.float32
        ), f"lse tensor dtype {lse.dtype} does not match expected dtype torch.float32"
        assert (
            lse.device == device
        ), f"lse tensor device {lse.device} does not match input device {device}"
        assert lse.is_cuda, "lse tensor must be on CUDA device"

    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        for t in (q, k, v, out)
    ]
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 1
        )
        if lse is not None
        else None
    )
    (
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
    ) = [
        (
            from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
            if t is not None
            else None
        )
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    page_table_tensor = (
        from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=1
        )
        if page_table is not None
        else None
    )
    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
        else:
            causal, local = False, True
    compute_capability = (
        torch.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    assert compute_capability in [
        9,
        10,
    ], "Unsupported compute capability. Supported: 9.x, 10.x"
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim
        # Perf heuristic from upstream: hdim=128, noncausal, non-local benefits from larger n_block
        if head_dim == head_dim_v == 128 and not causal and not local:
            n_block_size = 192
    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    if score_mod is not None:
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
        if is_varlen:
            raise NotImplementedError(
                "score_mod with buffers is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    cute_buffers = None
    if buffers is not None:
        cute_buffers = [from_dlpack(buf) for buf in buffers]

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        utils.hash_callable(score_mod) if score_mod is not None else None,
        buffers is not None,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        pack_gqa,
        compute_capability,
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        if compute_capability == 9:
            assert page_table is None, "paged KV not supported on SM 9.0"
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                has_buffers=buffers is not None,
            )
        elif compute_capability == 10:
            assert page_size in [
                None,
                128,
            ], "Only page_size=128 is supported for paged KV on SM 10.0"
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                is_persistent=not causal
                and not local
                and cu_seqlens_q is None
                and seqused_q is None,
                score_mod=score_mod,
                has_buffers=buffers is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        # TODO caching for buffers; cute_buffers
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            cute_buffers,
        )
    _flash_attn_fwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        page_table_tensor,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
        cute_buffers,
    )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def warmup_flash_attn(f):
    """
    Decorator for flash_attn_varlen_func:
    - On first call, run several warmup passes with different flag combinations:
        * return_softmax_lse in {False, True}
        * global noncausal (window_size=(None,None))
        * causal (window_size=(None,0))
        * local sliding window (window_size=(64,64))
        * optionally pack_gqa=True if qheads > kvheads and allowed
    - No score_mod / softcap (not supported for varlen yet)
    - Executes sequentially to minimize peak GPU mem
    - Does not modify user tensors (clones)
    """
    done = False

    def _clone_args(args, kwargs):
        """Clone tensor arguments to avoid sharing storage; deepcopy for others."""

        def maybe_clone(x):
            if isinstance(x, torch.Tensor):
                return x.detach().clone()  # detach to avoid autograd edges
            return copy.deepcopy(x)

        return tuple(maybe_clone(a) for a in args), {
            k: maybe_clone(v) for k, v in kwargs.items()
        }

    def _infer_heads(args, kwargs):
        """Infer q and kv head counts from arguments."""
        # Expect signature: (q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
        q = args[0] if len(args) > 0 else kwargs.get("q")
        k = args[1] if len(args) > 1 else kwargs.get("k")
        try:
            qh = int(q.shape[-2])
            kvh = int(k.shape[-2])
            return qh, kvh
        except Exception:
            return None, None

    def _run_warmups(args, kwargs):
        """Run warmup calls sequentially and release memory after each."""
        base_args, base_kwargs = _clone_args(args, kwargs)

        qh, kvh = _infer_heads(base_args, base_kwargs)
        can_pack_gqa = (
            qh is not None and kvh is not None and qh % kvh == 0 and qh // kvh > 1
        )
        has_page_table = (
            "page_table" in base_kwargs and base_kwargs["page_table"] is not None
        )

        # Window presets covering global, causal, and local
        window_presets = [
            (None, None),  # global noncausal
            (None, 0),  # causal
            (64, 64),  # local sliding window
        ]

        lse_flags = [False, True]

        # Base combo list
        combos = []
        for ws in window_presets:
            for return_lse_flag in lse_flags:
                combos.append(dict(window_size=ws, return_softmax_lse=return_lse_flag))

        # Optionally add a pack_gqa=True variant (FA4 may disable it internally for some varlen shapes/SMs)
        if can_pack_gqa:
            for ws in window_presets:
                combos.append(
                    dict(window_size=ws, return_softmax_lse=False, pack_gqa=True)
                )

        # If page_table is present, warm one combo with it (page_table in compile key for SM100)
        if has_page_table:
            combos.append(dict(window_size=(None, None), return_softmax_lse=False))

        # Run sequentially
        for combo in combos:
            wa, wk = _clone_args(base_args, base_kwargs)
            # Keep user-provided softcap/score_mod OUT (varlen+score_mod unsupported)
            wk.pop("score_mod", None)
            if "softcap" in wk and wk["softcap"]:
                wk["softcap"] = 0.0
            # Apply combo
            wk.update(combo)
            with torch.cuda.stream(torch.cuda.current_stream()):
                try:
                    f(*wa, **wk)
                except Exception as e:
                    # Some combos can be invalid for specific head dims / arch. Ignore and continue.
                    logger.debug("Warmup combo skipped: %s", e)
            del wa, wk
            torch.cuda.empty_cache()
            gc.collect()

    def wrapper(*args, **kwargs):
        nonlocal done
        if not done:
            logger.info(
                "Running FA4 warmup (global/causal/local, LSE on/off, optional GQA pack)..."
            )
            _run_warmups(args, kwargs)
            done = True
        return f(*args, **kwargs)

    return wrapper


@warmup_flash_attn
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    return_softmax_lse: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        softcap=softcap,
        pack_gqa=pack_gqa,
        return_lse=return_softmax_lse,
    )

    return (out, lse) if return_softmax_lse else out
