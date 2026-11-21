"""ROCm glue of triton_backend, bound there under _is_hip as _hip: TritonAttnBackend's
state for the gfx950 M3.1 DSpark draft (its grouped-head verify kernel reads a sliding
window, its target verify reads only the window KV indices, and its KV store skips the
defensive k/v clone when the pool stores them as given), and Inkling2's relative-bias
attention kernels, the DSpark draft's block-anchored windows and the one-launch verify
indptrs."""

from __future__ import annotations

import functools

import torch
import triton

from sglang.kernels.ops.attention.metadata import verify_indptrs
from sglang.srt.configs.model_config import is_dspark_draft
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MHATokenToKVPool
from sglang.srt.runtime_context import get_model
from sglang.srt.utils import is_gfx95_supported


def _windows_every_layer(hf_config) -> bool:
    from sglang.srt.speculative.dflash_utils import get_dflash_layer_types

    layer_types = get_dflash_layer_types(hf_config)
    return bool(layer_types) and all(t == "sliding_attention" for t in layer_types)


def is_minimax_m31_dspark_draft(hf_config) -> bool:
    """The DSpark draft of an M3.1 target (NVFP4 or Quark MXFP4); the MXFP8 M3 base
    target shares its arch."""
    # is_dspark_draft first, so other models never call get_model() here.
    return (
        is_dspark_draft(hf_config)
        and hf_config.architectures[0] == "DSparkMiniMaxDraftModel"
        and get_model().quantization in ("minimax_m3_nvfp4", "quark")
    )


def init_m31_draft(backend, hf_config) -> None:
    """The ROCm state of TritonAttnBackend.__init__; reads backend.sliding_window_size."""
    from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd

    m31_draft = is_minimax_m31_dspark_draft(hf_config)
    # The sliding-window path is validated only on the M3.1 draft (window 4095).
    backend.verify_shared_kv_fwd = torch.compiler.disable(
        functools.partial(verify_shared_kv_fwd, allow_sliding_window=m31_draft)
    )
    backend.is_gfx950_m31_draft = is_gfx95_supported() and m31_draft
    # The M3.1 draft windows every layer and reads only window_kv_indices; its
    # target-verify steps skip packing the full-context index stream.
    backend.verify_reads_window_kv_only = (
        backend.is_gfx950_m31_draft
        and backend.sliding_window_size is not None
        and backend.sliding_window_size > 0
        and _windows_every_layer(hf_config)
    )


def stores_kv_as_given(backend, k: torch.Tensor) -> bool:
    """Whether set_kv_buffer stores k and v as given: a plain MHA pool of their
    dtype ignores the layer's scales."""
    pool = backend.token_to_kv_pool
    return (
        type(pool) is MHATokenToKVPool
        and pool.dtype == k.dtype
        and not pool.is_quantized_kv_cache
    )


def init_inkling2_state(backend, model_runner, max_bs) -> None:
    """The ROCm state of TritonAttnBackend.__init__ for Inkling2 and DSpark drafts."""
    backend.anchor_max_bs = max_bs
    # (indptr, indices) buffers of block-anchored windows, sized on first use, and
    # (forward_batch, forward_metadata, width, (indptr, indices)) of the last fill.
    backend.anchor_window_buffers = None
    backend._anchor_window_fill = None
    backend._zero_extent_biases = {}
    # (forward_metadata, kv_indptr, (window, capturing), splits, bound) of the last fill
    backend._window_splits_fill = None
    backend.window_num_kv_splits_buf = None
    # ROCm DSpark drafts run their short blocks on the split-KV verify kernel
    backend.is_hip_dspark_draft = (
        model_runner.is_draft_worker and model_runner.spec_algorithm.is_dspark()
    )


def _window_rail(backend):
    """(name, window) of the backend's one window rail, or (None, None) without one;
    the ROCm paths here build a single rail."""
    if not backend.window_lengths:
        return None, None
    if len(backend.window_lengths) > 1:
        raise NotImplementedError(
            "The ROCm Triton verify and full-graph prefill paths build one window rail"
        )
    ((name, window),) = backend.window_lengths.items()
    return name, window


def rel_verify_max_extend():
    from sglang.kernels.ops.attention.inkling2_attn_hip import REL_VERIFY_MAX_EXTEND

    return REL_VERIFY_MAX_EXTEND


def update_target_verify_buffers(
    backend, plan, bs, seq_lens, spec_info, req_pool_indices
):
    """_update_target_verify_buffers on ROCm: the indptrs (and the window's lengths
    and starts) in one launch instead of the base's ten, the same integers."""
    from sglang.srt.layers.attention.triton_backend import pack_sliding_window_indices

    num_tokens_per_req = backend._target_verify_num_tokens_per_req(spec_info)
    name, window = _window_rail(backend)
    qo_indptr = backend.qo_indptr[: bs + 1]
    kv_indptr = backend.kv_indptr[: bs + 1]
    mask_indptr = backend.mask_indptr[: bs + 1]
    window_kv_lens, window_kv_start_idx = verify_indptrs(
        seq_lens[:bs],
        num_tokens_per_req,
        qo_indptr,
        kv_indptr,
        mask_indptr,
        window,
        backend.window_kv_indptr[name] if name is not None else None,
        backend.cuda_graph_window_kv_offsets[name] if name is not None else None,
    )
    if not backend.verify_reads_window_kv_only:
        backend.runner_ids.pack_read_stream(
            plan,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens[:bs],
            indptr=kv_indptr,
            out=backend.cuda_graph_kv_indices,
        )
    if name is not None:
        pack_sliding_window_indices(
            backend.window_kv_indptr[name][: bs + 1],
            backend.runner_ids,
            plan,
            req_pool_indices,
            window_kv_lens,
            window_kv_start_idx,
            token_to_kv_pool=backend.token_to_kv_pool,
            window_kv_indices=backend.cuda_graph_window_kv_indices[name],
            kind=name,
        )
    custom_mask = (
        backend._verify_mask.buffer if backend._verify_mask is not None else None
    )
    if spec_info is not None and spec_info.custom_mask is not None:
        custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
    else:
        custom_mask = None
    return qo_indptr, kv_indptr, custom_mask, mask_indptr


def init_full_cg_extend_metadata(backend, forward_batch, in_capture):
    """Full-graph prefill (EXTEND / MIXED): the eager extend branch's indptrs and
    prefix / window indices, written into buffers the captured kernels read on
    replay. Prefill graphs are captured before the decode graphs and with a fixed
    request-slot count, so they get their own buffers, sized at the first capture.
    max_extend_len is baked at capture as the bucket's token count, an upper bound:
    the kernels read real extents from qo_indptr."""
    from sglang.srt.layers.attention.triton_backend import (
        ForwardMetadata,
        WindowRail,
        update_sliding_window_buffer,
    )

    if backend.dcp_size > 1 or backend.enable_deterministic:
        raise NotImplementedError(
            "Full-graph prefill on the Triton backend has no DCP or deterministic mode"
        )
    bs = forward_batch.batch_size
    name, window = _window_rail(backend)
    bufs = getattr(backend, "full_cg_extend_buffers", None)
    if bufs is None:
        assert in_capture
        bufs = backend.full_cg_extend_buffers = (
            # every prefix is shorter than the context
            torch.empty(
                bs * backend.max_context_len, dtype=torch.int64, device=backend.device
            ),
            torch.empty(
                bs * window if name is not None else 0,
                dtype=torch.int64,
                device=backend.device,
            ),
            torch.zeros(bs, dtype=torch.int32, device=backend.device),
        )
    kv_indices, window_kv_indices, window_kv_offsets = bufs
    assert bs <= window_kv_offsets.shape[0], (bs, window_kv_offsets.shape[0])
    plan = forward_batch.iteration_ids
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    prefix_lens = forward_batch.extend_prefix_lens[:bs]
    kv_indptr = backend._fill_kv_indptr_and_indices(
        plan, bs, prefix_lens, req_pool_indices, kv_indices
    )
    qo_indptr = backend.qo_indptr[: bs + 1]
    qo_indptr[1:] = torch.cumsum(forward_batch.extend_seq_lens[:bs], dim=0)
    windows = {}
    if name is not None:
        window_kv_indptr, _, _, window_kv_start_idx = update_sliding_window_buffer(
            backend.window_kv_indptr[name],
            backend.runner_ids,
            plan,
            req_pool_indices,
            window,
            prefix_lens,
            bs,
            token_to_kv_pool=backend.token_to_kv_pool,
            window_kv_indices=window_kv_indices,
            kind=name,
        )
        window_kv_offsets = window_kv_offsets[:bs]
        window_kv_offsets.copy_(window_kv_start_idx)
        windows[name] = WindowRail(
            kv_indptr=window_kv_indptr,
            kv_indices=window_kv_indices,
            kv_offsets=window_kv_offsets,
            out_cache_loc=_fill_full_cg_extend_window_out_cache_loc(
                backend, forward_batch, name, in_capture
            ),
        )
    if not in_capture:
        return
    backend.forward_metadata = ForwardMetadata(
        attn_logits=None,
        attn_lse=None,
        max_extend_len=forward_batch.input_ids.shape[0],
        num_kv_splits=None,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        qo_indptr=qo_indptr,
        custom_mask=None,
        mask_indptr=None,
        windows=windows,
    )


def decode_split_bound(backend, num_token, window=None):
    """Most KV splits a length-based decode launch takes: one wave of programs for
    num_token rows, and with window no more than the window's keys need."""
    group = backend.num_head // backend.num_kv_head
    head_tiles = (
        backend.num_head
        if group == 1
        else triton.cdiv(backend.num_head, min(16, group))
    )
    bound = backend.device_core_count // max(1, num_token * head_tiles)
    bound = max(1, min(backend.max_kv_splits, bound))
    if window is not None:
        bound = min(bound, triton.cdiv(window + 1, backend.decode_split_min_tokens))
    return bound


def window_num_kv_splits(backend, layer, kv_indptr, num_token):
    """(num_kv_splits, bound) of a sliding-window decode layer: cdiv(window length,
    decode_split_min_tokens) splits per request. The first window layer of a forward
    (or of a captured graph) computes them from the rail's indptr; the rest reuse."""
    from sglang.kernels.ops.attention.metadata import (
        get_num_kv_splits_by_length_triton,
    )

    window = layer.sliding_window_size
    # A capture's eager warmup shares its forward_metadata: key on capturing too.
    capturing = torch.cuda.is_current_stream_capturing()
    fill = backend._window_splits_fill
    if (
        fill is not None
        and fill[0] is backend.forward_metadata
        and fill[1] is kv_indptr
        and fill[2] == (window, capturing)
    ):
        return fill[3], fill[4]
    num_seq = kv_indptr.shape[0] - 1
    bound = decode_split_bound(backend, num_token, window)
    if capturing:
        if backend.window_num_kv_splits_buf is None:
            backend.window_num_kv_splits_buf = torch.empty_like(
                backend.cuda_graph_num_kv_splits
            )
        splits = backend.window_num_kv_splits_buf[:num_token]
    else:
        splits = torch.empty(num_token, dtype=torch.int32, device=backend.device)
    get_num_kv_splits_by_length_triton[(1,)](
        splits,
        kv_indptr,
        num_seq,
        num_token // max(1, num_seq),
        bound,
        MIN_TOKENS=backend.decode_split_min_tokens,
        MAX_NUM_SEQ=max(256, triton.next_power_of_2(num_seq)),
        INDPTR=True,
    )
    backend._window_splits_fill = (
        backend.forward_metadata,
        kv_indptr,
        (window, capturing),
        splits,
        bound,
    )
    return splits, bound


def check_extend(backend, rel_bias, sinks, layer, gate_sink):
    """gate_sink is applied only by the split-KV relative-bias verify kernel, and
    full-graph prefill is built for the relative-bias extend kernel only: the generic
    extend kernels size their grids from host-side lengths, which a graph would
    freeze at capture."""
    if gate_sink is not None and not (
        takes_rel_extend(backend, rel_bias, sinks, layer)
        and backend.forward_metadata.max_extend_len <= rel_verify_max_extend()
    ):
        raise NotImplementedError(
            "gate_sink is applied only by the split-KV relative-bias verify kernel"
        )
    bufs = getattr(backend, "full_cg_extend_buffers", None)
    if (
        bufs is not None
        and backend.forward_metadata.kv_indices is bufs[0]
        and not takes_rel_extend(backend, rel_bias, sinks, layer)
    ):
        raise NotImplementedError(
            "Full-graph prefill on the ROCm Triton backend takes only the relative-"
            "bias extend attention (Inkling2); use --disable-prefill-cuda-graph"
        )


def _fill_full_cg_extend_window_out_cache_loc(backend, forward_batch, name, in_capture):
    """The window rail's write ids of a full-graph prefill in a buffer sized at the
    first (largest) capture; the bucket's padded tail writes the window's zero dummy
    slot."""
    out_cache_loc = forward_batch.out_cache_loc
    if name not in backend.window_write_sub_pools or out_cache_loc is None:
        return None
    n = out_cache_loc.shape[0]
    buf = getattr(backend, "full_cg_extend_window_out_cache_loc", None)
    if buf is None:
        assert in_capture
        buf = backend.full_cg_extend_window_out_cache_loc = torch.zeros(
            (n,), dtype=torch.int64, device=backend.device
        )
    assert n <= buf.shape[0], (n, buf.shape[0])
    if in_capture:
        buf[:n].zero_()
    else:
        buf[:n].copy_(backend.runner_ids.write_ids(forward_batch, name))
    buf[n:].zero_()
    return buf[:n]


def takes_rel_extend(backend, rel_bias, sinks, layer):
    """Whether forward_extend takes forward_rel_extend for this layer."""
    return (
        rel_bias is not None
        and backend.forward_metadata.custom_mask is None
        and sinks is None
        and backend.dllm_attention is None
        and layer.k_scale is None
        and not layer.window_anchor_at_start
    )


def forward_rel_extend(
    backend,
    q,
    k,
    v,
    o,
    lse,
    rel_bias,
    layer,
    kv_indptr,
    kv_indices,
    window,
    causal,
    gate_sink=None,
):
    """Inkling2's relative-bias extend: skips the bias on key tiles beyond its
    extent, which the generic score_mod kernel evaluates everywhere. Short
    extends (MTP verify, and the bidirectional MTP draft blocks) split the
    prefix keys across programs instead. Returns o, or with gate_sink (short
    extends) the gated output of the split-KV reduction."""
    from sglang.kernels.ops.attention.inkling2_attn_hip import (
        REL_VERIFY_MAX_EXTEND,
        rel_extend_attention,
        rel_verify_attention,
    )

    if lse is None:
        lse = q.new_empty((q.shape[0], layer.tp_q_head_num), dtype=torch.float32)
    gated = None
    if backend.forward_metadata.max_extend_len <= REL_VERIFY_MAX_EXTEND:
        epilogue = None
        if gate_sink is not None:
            gated = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
            epilogue = (gate_sink[0], gate_sink[1], gated)
        attention = functools.partial(
            rel_verify_attention, causal=causal, gate_sink=epilogue
        )
    else:
        assert causal, "rel_extend_attention is causal"
        attention = rel_extend_attention
    attention(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
        backend.kv_layers.read(layer.layer_id).key(),
        backend.kv_layers.read(layer.layer_id).value(),
        rel_bias,
        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        lse,
        backend.forward_metadata.qo_indptr,
        kv_indptr,
        kv_indices,
        backend.forward_metadata.max_extend_len,
        layer.scaling,
        window=window,
        page_size=backend.page_size,
    )
    return o if gated is None else gated


def zero_extent_bias(backend, q):
    """An empty (1, 1, 0) relative bias in q's dtype, made once (it is never read)."""
    bias = backend._zero_extent_biases.get(q.dtype)
    if bias is None:
        bias = q.new_zeros((1, 1, 1))[:, :, :0]
        backend._zero_extent_biases[q.dtype] = bias
    return bias


def anchored_window_kv(backend, layer, forward_batch):
    """FA4's window_anchor_at_start: every query of a bidirectional block sees the
    sliding_window_size + 1 cached keys before the block, plus the block itself."""
    from sglang.srt.layers.attention.triton_backend import update_sliding_window_buffer

    if (
        layer.attn_type != AttentionType.ENCODER_ONLY
        or backend.forward_metadata.custom_mask is not None
        or backend.dcp_size > 1
    ):
        raise NotImplementedError(
            "Block-anchored windows require bidirectional attention without "
            "a custom mask"
        )
    bs = forward_batch.batch_size
    width = layer.sliding_window_size + 1
    # Every anchored layer of one forward reads the same window: fill it once
    # (the DSpark draft's 4 local layers otherwise rebuild it each, 5 launches).
    # The init_forward_metadata* entries reset the fill for each new forward.
    fill = backend._anchor_window_fill
    if (
        fill is not None
        and fill[0] is forward_batch
        and fill[1] is backend.forward_metadata
        and fill[2] == width
    ):
        return fill[3]
    if backend.anchor_window_buffers is None:
        backend.anchor_window_buffers = (
            torch.zeros(
                backend.anchor_max_bs + 1, dtype=torch.int32, device=backend.device
            ),
            torch.empty(
                backend.anchor_max_bs * width, dtype=torch.int64, device=backend.device
            ),
        )
    indptr, indices = backend.anchor_window_buffers
    prefix_lens = (
        forward_batch.seq_lens
        if forward_batch.forward_mode.is_target_verify()
        else forward_batch.extend_prefix_lens
    )
    indptr, indices, _, _ = update_sliding_window_buffer(
        indptr,
        backend.runner_ids,
        forward_batch.iteration_ids,
        forward_batch.req_pool_indices,
        width,
        prefix_lens[:bs],
        bs,
        backend.device,
        backend.token_to_kv_pool,
        window_kv_indices=indices[: bs * width],
        kind=backend.kv_layers.route(layer.layer_id).sub_pool,
    )
    backend._anchor_window_fill = (
        forward_batch,
        backend.forward_metadata,
        width,
        (indptr, indices),
    )
    return indptr, indices


def fused_kv_write_buffers(backend, layer, forward_batch):
    """(k_buffer, v_buffer, loc) where save_kv_cache would store this layer's K/V
    rows; None for MLA, DCP, a translating pool or a cache that transforms them."""
    if (
        backend.use_mla
        or backend.dcp_size > 1
        or backend.runner_ids.is_translating
        or layer.k_scale is not None
    ):
        return None
    write_loc = KVWriteLoc.for_layer(
        forward_batch, layer, window_locs=backend._window_write_locs()
    )
    sub_pool = backend.kv_layers.route(layer.layer_id).sub_pool
    locations = (
        (write_loc.window_locs or {}).get(sub_pool)
        if sub_pool in backend.window_write_sub_pools
        else write_loc.loc
    )
    if locations is None:
        return None
    buffers = backend.kv_layers.read(layer.layer_id).kv()
    shapes = (
        (layer.tp_k_head_num, layer.qk_head_dim),
        (layer.tp_v_head_num, layer.v_head_dim),
    )
    for buffer, shape in zip(buffers, shapes):
        if (
            buffer.dtype != backend.token_to_kv_pool.dtype
            or buffer.shape[1:] != shape
            or buffer.stride()[-2:] != (shape[-1], 1)
        ):
            return None
    return (*buffers, locations)


def takes_draft_block(
    backend, layer, rel_bias, score_mod, sinks, logits_soft_cap, sliding_window_size
):
    """Whether forward_extend runs this DSpark draft block (at most
    REL_VERIFY_MAX_EXTEND tokens, full or block-anchored windows) on the split-KV
    verify kernel instead of the extend kernel's serial walk over the prefix."""
    return (
        backend.is_hip_dspark_draft
        and rel_bias is None
        and score_mod is None
        and sinks is None
        and not logits_soft_cap
        and backend.forward_metadata.custom_mask is None
        and backend.dllm_attention is None
        and layer.k_scale is None
        and sliding_window_size < 0
        and not (layer.xai_temperature_len and layer.xai_temperature_len > 0)
        and layer.qk_head_dim == layer.v_head_dim
        and backend.forward_metadata.max_extend_len <= rel_verify_max_extend()
    )


def forward_draft_block(backend, q, k, v, o, lse, layer, kv_indptr, kv_indices, causal):
    """A DSpark draft block on the split-KV verify kernel, with a zero-extent bias."""
    forward_rel_extend(
        backend,
        q,
        k,
        v,
        o,
        lse,
        zero_extent_bias(backend, q),
        layer,
        kv_indptr,
        kv_indices,
        -1,
        causal,
    )
