# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd Conv3D with Triton transforms and batched GEMMs.

A tile has 16 (F2) or 36 (F4) transform coordinates. Each coordinate is
one GEMM over input channels and all three temporal taps.
"""

from math import gcd

import torch
import triton
import triton.language as tl


@triton.jit
def _input_1d(d, TILE: tl.constexpr):
    if TILE == 2:
        return (d[0] - d[2], d[1] + d[2], d[2] - d[1], d[1] - d[3])
    else:
        odd1 = d[3] - 4 * d[1]
        even1 = d[4] - 4 * d[2]
        odd2 = 2 * (d[3] - d[1])
        even2 = d[4] - d[2]
        return (
            4 * d[0] - 5 * d[2] + d[4],
            even1 + odd1,
            even1 - odd1,
            even2 + odd2,
            even2 - odd2,
            4 * d[1] - 5 * d[3] + d[5],
        )


@triton.jit
def _output_1d(d, TILE: tl.constexpr):
    if TILE == 2:
        return (d[0] + d[1] + d[2], d[1] - d[2] - d[3])
    else:
        s12, d12 = d[1] + d[2], d[1] - d[2]
        s34, d34 = d[3] + d[4], d[3] - d[4]
        return (d[0] + s12 + s34, d12 + 2 * d34, s12 + 4 * s34, d12 + 8 * d34 + d[5])


@triton.jit
def _store_input_columns(
    horizontal, packed, dst, plane: tl.constexpr, TILE: tl.constexpr
):
    for j in tl.static_range(TILE + 2):
        column = ()
        for i in tl.static_range(TILE + 2):
            column += (horizontal[i][j],)
        values_out = _input_1d(column, TILE)
        for i in tl.static_range(TILE + 2):
            tl.store(packed + dst + (i * (TILE + 2) + j) * plane, values_out[i])


@triton.jit
def _cat_pad_winograd_pack(
    x,
    history,
    packed,
    next_history,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    HISTORY: tl.constexpr,
    TILE: tl.constexpr,
    REUSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # H and W are the unpadded spatial dimensions. Each CTA handles a channel
    # block of one spatial tile and temporal tap.
    tw: tl.constexpr = tl.cdiv(W, TILE)
    tiles: tl.constexpr = tl.cdiv(H, TILE) * tw
    spatial = tl.program_id(0) // (C // BLOCK)
    channel = tl.program_id(0) % (C // BLOCK) * BLOCK + tl.arange(0, BLOCK)
    z = tl.program_id(1)
    if REUSE:
        frame = z
        dst = (spatial * D + frame) * C + channel
        plane: tl.constexpr = tiles * D * C
        keep = frame >= D - 2
    else:
        output_frame, tap = z // 3, z % 3
        frame = output_frame + tap
        dst = ((output_frame * tiles + spatial) * 3 + tap) * C + channel
        plane: tl.constexpr = (D - 2) * tiles * 3 * C
        # The final output frame owns its final two taps.
        keep = (output_frame == D - 3) & (tap >= 1)

    if HISTORY:
        src = tl.where(
            frame < 2,
            history + (frame - 2 + HISTORY) * H * W * C,
            x + (frame - 2) * H * W * C,
        )
    else:
        src = x + (frame - 2) * H * W * C
    y, col = (spatial // tw) * TILE, (spatial % tw) * TILE
    horizontal = ()
    for i in tl.static_range(TILE + 2):
        values = ()
        for j in tl.static_range(TILE + 2):
            iy, ix = y + i - 1, col + j - 1
            off = (iy * W + ix) * C + channel
            valid = (frame >= 2 - HISTORY) & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
            v = tl.load(src + off, mask=valid, other=0)
            if 1 <= i and i <= TILE and 1 <= j and j <= TILE:
                # Only the interior of each tile owns these cache elements.
                tl.store(
                    next_history + (frame - D + 2) * H * W * C + off,
                    v,
                    mask=keep & (iy < H) & (ix < W),
                )
            values += (v.to(tl.float32),)
        horizontal += (_input_1d(values, TILE),)
    _store_input_columns(horizontal, packed, dst, plane, TILE)


@triton.jit
def _pack(
    x,
    packed,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    TILE: tl.constexpr,
    REUSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    th: tl.constexpr = tl.cdiv(H - 2, TILE)
    tw: tl.constexpr = tl.cdiv(W - 2, TILE)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = idx % C
    if REUSE:
        row = idx // C
        frame = row // (th * tw)
        dst = ((row % (th * tw)) * D + frame) * C + channel
        plane = th * tw * D * C
    else:
        row = idx // (3 * C)
        frame = row // (th * tw) + idx // C % 3
        dst = idx
        plane = (D - 2) * th * tw * 3 * C

    y, x0 = TILE * (row // tw % th), TILE * (row % tw)
    origin = ((frame * H + y) * W + x0) * C + channel
    # Apply the 1-D transform across columns, then across rows of each tile.
    horizontal = ()
    for i in tl.static_range(TILE + 2):
        values = ()
        for j in tl.static_range(TILE + 2):
            ptr = x + origin + (i * W + j) * C
            if (H - 2) % TILE or (W - 2) % TILE:
                v = tl.load(ptr, mask=(y + i < H) & (x0 + j < W), other=0)
            else:
                v = tl.load(ptr)
            values += (v.to(tl.float32),)
        horizontal += (_input_1d(values, TILE),)
    _store_input_columns(horizontal, packed, dst, plane, TILE)


@triton.jit
def _unpack(
    products,
    out,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = idx // C
    th: tl.constexpr = tl.cdiv(H, TILE)
    tw: tl.constexpr = tl.cdiv(W, TILE)
    frame = row // (th * tw)
    y, x = TILE * (row // tw % th), TILE * (row % tw)
    origin = ((frame * H + y) * W + x) * C + idx % C
    horizontal = ()
    for i in tl.static_range(TILE + 2):
        values = ()
        for j in tl.static_range(TILE + 2):
            mode = i * (TILE + 2) + j
            values += (tl.load(products + idx + mode * ROWS * C).to(tl.float32),)
        horizontal += (_output_1d(values, TILE),)
    for j in tl.static_range(TILE):
        column = ()
        for i in tl.static_range(TILE + 2):
            column += (horizontal[i][j],)
        values_out = _output_1d(column, TILE)
        for i in tl.static_range(TILE):
            ptr = out + origin + (i * W + j) * C
            if H % TILE or W % TILE:
                tl.store(ptr, values_out[i], mask=(y + i < H) & (x + j < W))
            else:
                tl.store(ptr, values_out[i])


def _pack_weight(weight, tile):
    if tile == 2:
        coefficients = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
    else:
        coefficients = [
            [1 / 4, 0, 0],
            [-1 / 6, -1 / 6, -1 / 6],
            [-1 / 6, 1 / 6, -1 / 6],
            [1 / 24, 1 / 12, 1 / 6],
            [1 / 24, -1 / 12, 1 / 6],
            [0, 0, 1],
        ]
    g = weight.new_tensor(coefficients, dtype=torch.float32)
    transformed = torch.einsum("ui,octij,vj->uvotc", g, weight.float(), g)
    return (
        transformed.reshape((tile + 2) ** 2, weight.size(0), -1)
        .half()
        .contiguous()
        .transpose(1, 2)
    )


def try_winograd_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weights: dict,
    *,
    fuse_cache: bool = False,
    history: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Bare unit-stride 3x3x3 cross-correlation, or None for unsupported inputs.

    Supports a single channels-last BF16 volume. The caller selects the layers
    and owns the packed filters for the lifetime of the resident weight.
    With fuse_cache, x is the unpadded activation, the spatial pad is one and
    the temporal pad is two. Return the convolution and a compact two-frame
    history. Otherwise x is the existing padded channels-last input.
    """
    if x.ndim != 5 or weight.ndim != 5:
        return None
    _, cin, depth, height, width = x.shape
    if fuse_cache:
        depth, height, width = depth + 2, height + 2, width + 2
    cout = weight.size(0)
    if (
        x.size(0) != 1
        or min(cin, cout) < 1
        or min(depth, height, width) < 3
        or weight.shape[1:] != (cin, 3, 3, 3)
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or torch.version.cuda is None
        or not x.is_cuda
        or weight.device != x.device
        or not x.is_contiguous(memory_format=torch.channels_last_3d)
        or not weight.is_contiguous(memory_format=torch.channels_last_3d)
        # A captured graph would keep using this packed address after a refit.
        or torch.cuda.is_current_stream_capturing()
        or (
            torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") != x.dtype
        )
    ):
        return None
    if (
        fuse_cache
        and history is not None
        and (
            history.ndim != 5
            or history.shape[:2] != x.shape[:2]
            or history.shape[3:] != x.shape[3:]
            or history.size(2) not in (1, 2)
            or history.device != x.device
            or history.dtype != x.dtype
            or not history.is_contiguous(memory_format=torch.channels_last_3d)
        )
    ):
        return None
    out_d, out_h, out_w = depth - 2, height - 2, width - 2
    tile = 4 if out_h % 4 == 0 and out_w % 4 == 0 else 2
    reuse = depth == 6
    th, tw = triton.cdiv(out_h, tile), triton.cdiv(out_w, tile)
    modes = (tile + 2) ** 2
    rows = th * tw if reuse else out_d * th * tw
    packed_channels = (depth if reuse else 3) * cin
    # The transforms use signed 32-bit offsets. Fused packing launches the
    # temporal taps on the second CUDA grid dimension.
    max_elements = max(
        depth * height * width * cin,
        out_d * out_h * out_w * cout,
        modes * rows * max(packed_channels, cout),
    )
    if max_elements > 2**31 - 1 or (fuse_cache and out_d * 3 > 65535):
        return None
    # Both the fused input channel group and the flat output span use full blocks.
    block = gcd(128, cin, rows * cout)
    packed = x.new_empty((modes, rows, packed_channels), dtype=torch.float16)
    if fuse_cache:
        next_history = torch.empty(
            (1, cin, 2, out_h, out_w),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
        _cat_pad_winograd_pack[(th * tw * cin // block, depth if reuse else out_d * 3)](
            x,
            x if history is None else history,
            packed,
            next_history,
            depth,
            out_h,
            out_w,
            cin,
            0 if history is None else history.size(2),
            tile,
            reuse,
            block,
        )
    else:
        _pack[(packed.numel() // (modes * block),)](
            x, packed, depth, height, width, cin, tile, reuse, block
        )
    out = x.new_empty((1, out_d, out_h, out_w, cout))
    with torch.autocast("cuda", enabled=False):
        # Normal tensors expose in-place mutations through their version. The
        # loader invalidates inference tensors explicitly before weight updates.
        try:
            version = None if weight.is_inference() else weight._version
        except RuntimeError:
            # Layerwise reload can rebind an inference Parameter to normal
            # storage without giving the original Parameter a version counter.
            version = None
        signature = (id(weight), weight.data_ptr(), version)
        stream = torch.cuda.current_stream(weight.device)
        entry = packed_weights.get(tile)
        if entry is None or entry[0] != signature:
            filt = _pack_weight(weight, tile)
            ready = stream.record_event()
            packed_weights[tile] = (signature, filt, stream, ready)
        else:
            _, filt, producer, ready = entry
            if stream != producer:
                stream.wait_event(ready)
                filt.record_stream(stream)
        for frame in range(out_d if reuse else 1):
            window = packed[:, :, frame * cin : (frame + 3) * cin] if reuse else packed
            # F(2,3) uses an FP32 GEMM output. F(4,3) widens FP16 products
            # to FP32 at the inverse transform, as in the validated decoder.
            if tile == 2:
                products = torch.bmm(window, filt, out_dtype=torch.float32)
            else:
                products = torch.bmm(window, filt)
            dest = out[:, frame] if reuse else out
            _unpack[(rows * cout // block,)](
                products, dest, out_h, out_w, cout, rows, tile, block
            )
    out = out.permute(0, 4, 1, 2, 3)
    return (out, next_history) if fuse_cache else out
