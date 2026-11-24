import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit

_CUDA_TYPES = {torch.float32: "float", torch.bfloat16: "__nv_bfloat16"}
# safe_idx, cache_mask, cu, si, track mask, track indices, src2dst
_INT_DTYPES = (
    torch.int64,
    torch.bool,
    torch.int64,
    torch.int32,
    torch.bool,
    torch.int64,
    torch.int32,
)
# Extend batches of at least this many rows use the 2-D tiled kernel.
PREFILL_MIN_ROWS = 2048


@triton.jit
def _sconv_kernel(
    X,
    Weight,
    Dynamic,
    Cache,
    Indices,
    Mask,
    Cu,
    Seq,
    Y,
    TrackMask,
    TrackIndices,
    Src2Dst,
    Partial,
    Attention,
    Out,
    SX: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SC0: tl.constexpr,
    SC1: tl.constexpr,
    SC2: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DYNAMIC: tl.constexpr,
    DECODE: tl.constexpr,
    TRACK: tl.constexpr,
    RESIDUAL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr = 0,
    EPILOGUE: tl.constexpr = 0,
    Window=None,
    SWB: tl.constexpr = 0,
    SWT: tl.constexpr = 0,
    SWW: tl.constexpr = 0,
    WINDOW: tl.constexpr = False,
    ROUND_BF16: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < D
    seq = row if DECODE else tl.load(Seq + row)
    bos = row if DECODE else tl.load(Cu + seq)
    slot = tl.load(Indices + seq).to(tl.int64)
    # Every CUDA-graph padding row maps to the reserved slot 0; sharing its state
    # across padding rows corrupts real rows, so padding rows neither read nor write it.
    has_cache = (tl.load(Mask + seq) != 0) & (slot > 0)
    if TOP_K:
        gathered = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(TOP_K):
            dst = tl.load(Src2Dst + row * TOP_K + k).to(tl.int64)
            gathered += tl.load(X + dst * SX + col, valid) * 1.0
        current = gathered.to(X.dtype.element_ty).to(tl.float32)
        if ROUND_BF16:
            current = current.to(tl.bfloat16).to(tl.float32)
    else:
        current = tl.load(X + row * SX + col, valid, other=0).to(tl.float32)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(W):
        lag = tap if DYNAMIC else W - 1 - tap
        src = row - lag
        from_x = src >= bos
        if TOP_K and (DECODE or lag == 0):
            value = current
        elif TOP_K:
            gathered = tl.zeros([BLOCK], dtype=tl.float32)
            for k in tl.static_range(TOP_K):
                dst = tl.load(Src2Dst + src * TOP_K + k, from_x, other=0).to(tl.int64)
                gathered += tl.load(X + dst * SX + col, valid & from_x, other=0) * 1.0
            value = gathered.to(X.dtype.element_ty).to(tl.float32)
            if ROUND_BF16:
                value = value.to(tl.bfloat16).to(tl.float32)
        else:
            value = tl.load(X + src * SX + col, valid & from_x, other=0).to(tl.float32)
        prefix = tl.load(
            Cache + slot * SC0 + (src - bos + W - 1) * SC1 + col * SC2,
            valid & ~from_x & has_cache,
            other=0,
        ).to(tl.float32)
        value = tl.where(from_x, value, prefix)
        if WINDOW and lag < W - 1:
            # save_intermediate_conv_windows: raw inputs, cache read for every
            # pooled row (padding rows included).
            in_pool = tl.load(Mask + seq) != 0
            raw_prefix = tl.load(
                Cache + slot * SC0 + (src - bos + W - 1) * SC1 + col * SC2,
                valid & ~from_x & in_pool,
                other=0,
            ).to(tl.float32)
            tl.store(
                Window + seq * SWB + (row - bos) * SWT + (W - 2 - lag) * SWW + col,
                tl.where(from_x, value, raw_prefix),
                valid & in_pool,
            )
        if DYNAMIC:
            dyn = tl.load(
                Dynamic + row * (D // HEAD_DIM) * W + (col // HEAD_DIM) * W + lag,
                valid,
                other=0,
            ).to(tl.float32)
            weight = tl.load(Weight + lag * SW0 + col * SW1, valid, other=0).to(
                tl.float32
            )
            weight = weight + dyn
        else:
            weight = (
                tl.load(Weight + col * SW0 + tap * SW1, valid, other=0)
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )
        if DYNAMIC and tap == 0:
            acc = value * weight
        else:
            acc = tl.fma(value, weight, acc)
    if RESIDUAL:
        acc = acc + current
    acc = acc * SCALE
    if EPILOGUE == 0:
        tl.store(Y + row * D + col, acc, valid)
    else:
        y = acc.to(X.dtype.element_ty).to(tl.float32)
        attention = tl.load(Attention + row * D + col, valid, other=0).to(tl.float32)
        if EPILOGUE == 1:
            total = tl.zeros([BLOCK], dtype=tl.float32)
            total += tl.load(Partial + row * D + col, valid, other=0).to(tl.float32)
            total += attention
            total += y
        else:
            total = attention + y
        tl.store(Out + row * D + col, total, valid)
    if DECODE:
        for tap in tl.static_range(W - 1):
            if tap == W - 2:
                value = current
            else:
                value = tl.load(
                    Cache + slot * SC0 + (tap + 1) * SC1 + col * SC2,
                    valid & has_cache,
                    other=0,
                )
            tl.store(
                Cache + slot * SC0 + tap * SC1 + col * SC2, value, valid & has_cache
            )
            if TRACK:
                track = tl.load(TrackMask + seq)
                dest = tl.load(TrackIndices + seq).to(tl.int64)
                tl.store(
                    Cache + dest * SC0 + tap * SC1 + col * SC2,
                    value,
                    valid & has_cache & track,
                )


@triton.jit
def _sconv_prefill_kernel(
    X,
    Weight,
    Dynamic,
    Cache,
    Indices,
    Mask,
    Cu,
    Seq,
    Y,
    n_rows,
    SX: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SC0: tl.constexpr,
    SC1: tl.constexpr,
    SC2: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DYNAMIC: tl.constexpr,
    RESIDUAL: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    """_sconv_kernel for extend batches over (BM rows, BN columns) tiles."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    valid = (rows < n_rows)[:, None] & (col < D)[None, :]
    rows = tl.minimum(rows, n_rows - 1)
    seq = tl.load(Seq + rows)
    bos = tl.load(Cu + seq)
    slot = tl.load(Indices + seq).to(tl.int64)
    has_cache = (tl.load(Mask + seq) != 0) & (slot > 0)
    current = tl.load(X + rows[:, None] * SX + col[None, :], valid, other=0).to(
        tl.float32
    )
    acc = tl.full((BM, BN), 0, tl.float32)
    for tap in tl.static_range(W):
        lag = tap if DYNAMIC else W - 1 - tap
        src = rows - lag
        from_x = (src >= bos)[:, None]
        value = tl.load(
            X + src[:, None] * SX + col[None, :], valid & from_x, other=0
        ).to(tl.float32)
        prefix = tl.load(
            Cache
            + slot[:, None] * SC0
            + (src - bos + W - 1)[:, None] * SC1
            + col[None, :] * SC2,
            valid & ~from_x & has_cache[:, None],
            other=0,
        ).to(tl.float32)
        value = tl.where(from_x, value, prefix)
        if DYNAMIC:
            dyn = tl.load(
                Dynamic
                + rows[:, None] * (D // HEAD_DIM) * W
                + (col // HEAD_DIM)[None, :] * W
                + lag,
                valid,
                other=0,
            ).to(tl.float32)
            weight = tl.load(Weight + lag * SW0 + col * SW1, col < D, other=0).to(
                tl.float32
            )
            weight = weight[None, :] + dyn
        else:
            weight = (
                tl.load(Weight + col * SW0 + tap * SW1, col < D, other=0)
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )[None, :]
        if DYNAMIC and tap == 0:
            acc = value * weight
        else:
            acc = tl.fma(value, weight, acc)
    if RESIDUAL:
        acc = acc + current
    acc = acc * SCALE
    tl.store(Y + rows[:, None] * D + col[None, :], acc, valid)


@cache_once
def _sconv_module(
    config, x_dtype, weight_dtype, cache_dtype, attention_dtype, out_dtype
):
    dtypes = (x_dtype, weight_dtype, cache_dtype, attention_dtype, out_dtype)
    args = ", ".join(
        [*(str(v).lower() for v in config), *(_CUDA_TYPES[t] for t in dtypes)]
    )
    return load_jit(
        "inkling2_sconv",
        *(str(v) for v in config),
        *(str(t).split(".")[-1] for t in dtypes),
        cuda_files=["inkling2/sconv.cuh"],
        cuda_wrappers=[("run", f"inkling2_sconv::run<{args}>")],
    )


def _cuda_sconv(
    x,
    weight,
    dynamic_weights,
    cache,
    metadata,
    output,
    *,
    width,
    head_dim,
    decode,
    residual,
    scale,
    track_mask=None,
    track_indices=None,
    src2dst=None,
    top_k=0,
    partial=None,
    attention=None,
    out=None,
    window=None,
    round_bf16=False,
):
    """_sconv_kernel in CUDA; False when an operand layout or dtype is unsupported.
    FP32 x only in the routed epilogue, and with top-k rows only under round_bf16
    (the CUDA gather rounds the sum to bf16)."""
    rows, d = (attention if attention is not None else x).shape
    ints = (
        metadata["safe_idx"],
        metadata["cache_mask"],
        metadata["cu"],
        metadata["si"],
        track_mask,
        track_indices,
        src2dst,
    )
    tensors = (x, cache, dynamic_weights, attention, partial, out, output, window)
    if not (
        (
            x.dtype == torch.bfloat16
            or (
                x.dtype == torch.float32
                and output is None
                and (top_k == 0 or round_bf16)
            )
        )
        and (dynamic_weights is None or dynamic_weights.dtype == torch.bfloat16)
        and weight.dtype in _CUDA_TYPES
        and cache.dtype in _CUDA_TYPES
        and (window is None or window.dtype == cache.dtype)
        and (output is None or output.dtype == torch.bfloat16)
        and (attention is None or attention.dtype in _CUDA_TYPES)
        and (out is None or out.dtype in _CUDA_TYPES)
        and (partial is None or partial.dtype == out.dtype)
        and all(t is None or t.data_ptr() % 16 == 0 for t in tensors)
        and all(
            t is None or t.is_contiguous()
            for t in (dynamic_weights, attention, partial, out, output)
        )
        and d % 8 == 0
        and x.stride(0) % 8 == 0
        and cache.stride(2) == 1
        and cache.stride(0) % 8 == cache.stride(1) % 8 == 0
        and (window is None or window.stride(0) % 8 == window.stride(1) % 8 == 0)
        and (
            dynamic_weights is None
            or (
                weight.stride(1) == 1
                and weight.stride(0) % 8 == 0
                and weight.data_ptr() % 16 == 0
            )
        )
        and all(
            t is None or t.dtype == dtype
            for t, dtype in zip(ints, _INT_DTYPES, strict=True)
        )
    ):
        return False
    config = (
        width,
        top_k,
        dynamic_weights is not None,
        decode,
        window is not None,
        residual,
        0 if out is None else 2 if partial is None else 1,
    )
    dtypes = (
        x.dtype,
        weight.dtype,
        cache.dtype,
        attention.dtype if attention is not None else torch.float32,
        out.dtype if out is not None else torch.float32,
    )
    ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
    _sconv_module(config, *dtypes).run(
        x,
        rows,
        x.stride(0),
        weight.data_ptr(),
        *weight.stride(),
        ptr(dynamic_weights),
        cache.data_ptr(),
        *cache.stride()[:2],
        *(t.data_ptr() for t in ints[:4]),
        ptr(output),
        ptr(track_mask),
        ptr(track_indices),
        ptr(src2dst),
        ptr(partial),
        ptr(attention),
        ptr(out),
        ptr(window),
        *(window.stride()[:3] if window is not None else (0, 0, 0)),
        d,
        head_dim,
        scale,
    )
    return True


def _sconv_cpu_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    dynamic_weights: torch.Tensor | None,
    cache: torch.Tensor,
    metadata: dict,
    *,
    width: int,
    head_dim: int,
    decode: bool,
    use_residual: bool,
    scale: float,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
    src2dst: torch.Tensor | None = None,
    top_k: int = 0,
    partial: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
    window: torch.Tensor | None = None,
    round_bf16: bool = False,
) -> torch.Tensor:
    is_epilogue = attention is not None
    rows, d = (attention if is_epilogue else x).shape
    out_dtype = (
        (attention.dtype if partial is None else partial.dtype)
        if is_epilogue
        else x.dtype
    )
    if rows == 0:
        return torch.empty((0, d), dtype=out_dtype, device=cache.device)

    dynamic = dynamic_weights is not None
    if top_k > 0:
        gathered = x[src2dst.long().view(rows, top_k)].float().sum(dim=1)
        all_inputs = gathered.to(x.dtype).float()
        if round_bf16:
            all_inputs = all_inputs.to(torch.bfloat16).float()
    else:
        all_inputs = x.float()

    safe_idx = metadata["safe_idx"].long()
    cache_mask = metadata["cache_mask"].reshape(-1).bool()
    cu = metadata["cu"].long()
    si = metadata["si"].long()

    row_idx = torch.arange(rows, dtype=torch.long)
    seq = row_idx if decode else si[:rows]
    bos = row_idx if decode else cu[seq]
    slot = safe_idx[seq]
    in_pool = cache_mask[seq]
    has_cache = in_pool & (slot > 0)

    acc = torch.zeros((rows, d), dtype=torch.float32)
    for tap in range(width):
        lag = tap if dynamic else (width - 1 - tap)
        src_pos = row_idx - lag
        from_x = src_pos >= bos
        x_val = all_inputs[src_pos.clamp(min=0)]
        c_pos = (src_pos - bos + (width - 1)).clamp(min=0, max=max(width - 2, 0))
        c_val = cache[slot, c_pos].float()
        val = torch.where(
            from_x[:, None], x_val, torch.where(has_cache[:, None], c_val, 0.0)
        )
        if window is not None and lag < width - 1:
            raw_val = torch.where(
                from_x[:, None], x_val, torch.where(in_pool[:, None], c_val, 0.0)
            )
            if in_pool.any():
                window[seq[in_pool], (row_idx - bos)[in_pool], width - 2 - lag] = (
                    raw_val[in_pool].to(window.dtype)
                )
        if dynamic:
            w = (
                weight[lag].float()[None, :]
                + dynamic_weights[:, :, lag].repeat_interleave(head_dim, dim=1).float()
            )
        else:
            w = weight[:, tap].to(x.dtype).float()[None, :]
        acc = acc + val * w

    if use_residual:
        acc = acc + all_inputs
    acc = acc * scale

    if is_epilogue:
        y_val = acc.to(x.dtype).float()
        total = attention.float() + y_val
        if partial is not None:
            total = total + partial.float()
        out = total.to(out_dtype)
    else:
        out = acc.to(out_dtype)

    if decode and has_cache.any():
        active_slots = slot[has_cache]
        if width > 2:
            new_win = torch.cat(
                (
                    cache[active_slots, 1:],
                    all_inputs[has_cache].to(cache.dtype).unsqueeze(1),
                ),
                dim=1,
            )
        else:
            new_win = all_inputs[has_cache].to(cache.dtype).unsqueeze(1)
        cache[active_slots] = new_win
        if track_mask is not None and track_indices is not None:
            tm = track_mask.reshape(-1)[seq].bool() & has_cache
            if tm.any():
                tm_slots = slot[tm]
                if width > 2:
                    t_win = torch.cat(
                        (
                            cache[tm_slots, 1:],
                            all_inputs[tm].to(cache.dtype).unsqueeze(1),
                        ),
                        dim=1,
                    )
                else:
                    t_win = all_inputs[tm].to(cache.dtype).unsqueeze(1)
                # Note: cache[active_slots] already updated tm_slots, so slice[:-1] was shifted!
                # Reuse new_win indexed by tm[has_cache]
                cache[track_indices.reshape(-1)[seq[tm]].long()] = new_win[
                    tm[has_cache]
                ]
    return out


def short_convolution(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    metadata: dict,
    *,
    dynamic_weights: torch.Tensor | None = None,
    decode: bool = False,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """With window (MTP verify, [batch, tokens, W - 1, D]), also saves each row's
    conv window like save_intermediate_conv_windows."""
    dynamic = dynamic_weights is not None
    assert window is None or (not decode and window.stride(-1) == 1)
    width = weight.shape[0] if dynamic else weight.shape[1]
    head_dim = x.shape[1] // dynamic_weights.shape[1] if dynamic else x.shape[1]
    scale = (width * (2 if dynamic else 1) + int(use_residual)) ** -0.5
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    if not x.is_cuda:
        if x.shape[0] == 0:
            return output
        assert x.stride(-1) == 1
        assert cache.ndim == 3 and cache.shape[1:] == (width - 1, x.shape[1])
        assert (track_mask is None) == (track_indices is None)
        if hasattr(torch.ops.sgl_kernel, "inkling2_short_convolution_cpu"):
            return torch.ops.sgl_kernel.inkling2_short_convolution_cpu(
                x,
                weight,
                cache,
                metadata["safe_idx"],
                metadata["cache_mask"],
                metadata["cu"],
                metadata["si"],
                dynamic_weights=dynamic_weights,
                decode=decode,
                use_residual=use_residual,
                track_mask=track_mask,
                track_indices=track_indices,
                window=window,
            )
        return _sconv_cpu_fallback(
            x,
            weight,
            dynamic_weights,
            cache,
            metadata,
            width=width,
            head_dim=head_dim,
            decode=decode,
            use_residual=use_residual,
            scale=scale,
            track_mask=track_mask,
            track_indices=track_indices,
            window=window,
        )
    if x.shape[0]:
        assert x.is_cuda and x.stride(-1) == 1
        assert cache.ndim == 3 and cache.shape[1:] == (width - 1, x.shape[1])
        assert (track_mask is None) == (track_indices is None)
        if not decode and window is None and x.shape[0] >= PREFILL_MIN_ROWS:
            _sconv_prefill_kernel[
                (triton.cdiv(x.shape[0], 16), triton.cdiv(x.shape[1], 128))
            ](
                x,
                weight,
                dynamic_weights,
                cache,
                metadata["safe_idx"],
                metadata["cache_mask"],
                metadata["cu"],
                metadata["si"],
                output,
                x.shape[0],
                x.stride(0),
                *weight.stride(),
                *cache.stride(),
                x.shape[1],
                width,
                head_dim,
                dynamic,
                use_residual,
                scale,
                16,
                128,
                enable_fp_fusion=False,
                **(
                    {"USE_PDL": True, "launch_pdl": True}
                    if is_arch_support_pdl()
                    else {}
                ),
            )
            return output
        if _cuda_sconv(
            x,
            weight,
            dynamic_weights,
            cache,
            metadata,
            output,
            width=width,
            head_dim=head_dim,
            decode=decode,
            residual=use_residual,
            scale=scale,
            track_mask=track_mask,
            track_indices=track_indices,
            window=window,
        ):
            return output
        _sconv_kernel[(x.shape[0], triton.cdiv(x.shape[1], 256))](
            x,
            weight,
            dynamic_weights,
            cache,
            metadata["safe_idx"],
            metadata["cache_mask"],
            metadata["cu"],
            metadata["si"],
            output,
            track_mask,
            track_indices,
            None,
            None,
            None,
            None,
            x.stride(0),
            *weight.stride(),
            *cache.stride(),
            x.shape[1],
            width,
            head_dim,
            dynamic,
            decode,
            track_mask is not None,
            use_residual,
            scale,
            256,
            Window=window,
            SWB=window.stride(0) if window is not None else 0,
            SWT=window.stride(1) if window is not None else 0,
            SWW=window.stride(2) if window is not None else 0,
            WINDOW=window is not None,
            enable_fp_fusion=False,
            **({"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}),
        )
    return output


def routed_sconv_epilogue(
    down,
    src2dst,
    top_k,
    weight,
    cache,
    metadata,
    attention,
    partial=None,
    *,
    track_mask=None,
    track_indices=None,
    window=None,
    round_bf16=False,
):
    """Fusion of the routed-expert combine (unit weights, rounded like
    post_reorder), the short convolution and its cache update, and the residual
    update: returns partial + attention + conv, or attention + conv without partial.
    Decode, or MTP verify with window (each row's conv window saved instead of
    updating the cache)."""
    rows, width = attention.shape
    out = torch.empty_like(attention if partial is None else partial)
    scale = (weight.shape[1] + 1) ** -0.5
    if not attention.is_cuda:
        if rows == 0:
            return out
        assert down.stride(-1) == 1 and weight.ndim == 2
        assert cache.ndim == 3 and cache.shape[1:] == (weight.shape[1] - 1, width)
        assert (track_mask is None) == (track_indices is None)
        if hasattr(torch.ops.sgl_kernel, "inkling2_routed_sconv_epilogue_cpu"):
            return torch.ops.sgl_kernel.inkling2_routed_sconv_epilogue_cpu(
                down,
                src2dst,
                top_k,
                weight,
                cache,
                metadata["safe_idx"],
                metadata["cache_mask"],
                metadata["cu"],
                metadata["si"],
                attention,
                partial=partial,
                track_mask=track_mask,
                track_indices=track_indices,
                window=window,
                round_bf16=round_bf16,
            )
        return _sconv_cpu_fallback(
            down,
            weight,
            None,
            cache,
            metadata,
            width=weight.shape[1],
            head_dim=width,
            decode=window is None,
            use_residual=True,
            scale=scale,
            track_mask=track_mask,
            track_indices=track_indices,
            src2dst=src2dst,
            top_k=top_k,
            partial=partial,
            attention=attention,
            window=window,
            round_bf16=round_bf16,
        )
    if rows:
        assert down.stride(-1) == 1 and weight.ndim == 2
        assert cache.ndim == 3 and cache.shape[1:] == (weight.shape[1] - 1, width)
        assert (track_mask is None) == (track_indices is None)
        if _cuda_sconv(
            down,
            weight,
            None,
            cache,
            metadata,
            None,
            width=weight.shape[1],
            head_dim=width,
            decode=window is None,
            residual=True,
            scale=scale,
            track_mask=track_mask,
            track_indices=track_indices,
            src2dst=src2dst,
            top_k=top_k,
            partial=partial,
            attention=attention,
            out=out,
            window=window,
            round_bf16=round_bf16,
        ):
            return out
        _sconv_kernel[(rows, triton.cdiv(width, 256))](
            down,
            weight,
            None,
            cache,
            metadata["safe_idx"],
            metadata["cache_mask"],
            metadata["cu"],
            metadata["si"],
            None,
            track_mask,
            track_indices,
            src2dst,
            partial,
            attention,
            out,
            down.stride(0),
            *weight.stride(),
            *cache.stride(),
            width,
            weight.shape[1],
            width,
            False,
            window is None,
            track_mask is not None,
            True,
            scale,
            256,
            TOP_K=top_k,
            ROUND_BF16=round_bf16,
            EPILOGUE=2 if partial is None else 1,
            Window=window,
            SWB=window.stride(0) if window is not None else 0,
            SWT=window.stride(1) if window is not None else 0,
            SWW=window.stride(2) if window is not None else 0,
            WINDOW=window is not None,
            enable_fp_fusion=False,
            **({"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}),
        )
    return out
