import torch
import triton
import triton.language as tl


def can_fuse_dfly_inputs(
    device: torch.device | str, *, query_width: int, window: int
) -> bool:
    return (
        torch.device(device).type == "cuda"
        and torch.version.hip is None
        and 0 < query_width <= 128
        and 0 <= window <= 1024
    )


def _check_tensor(tensor, *, shape, dtype, device, name, contiguous=True):
    if (
        tuple(tensor.shape) != tuple(shape)
        or tensor.dtype not in dtype
        or tensor.device != device
        or (contiguous and not tensor.is_contiguous())
    ):
        raise ValueError(
            f"DFly {name}: expected shape={shape}, dtype={dtype}, device={device}, "
            f"contiguous={contiguous}; got shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}, stride={tensor.stride()}"
        )


@triton.jit
def _plan_dfly_inputs_kernel(
    Req,
    Anchors,
    Bonus,
    Live,
    Slots,
    Hidden,
    Delta,
    ReqToToken,
    InputOut,
    PositionOut,
    VerifyOut,
    SlotOut,
    AnchorOut,
    ActiveOut,
    HiddenOut,
    MropeOut,
    SortedWindow,
    REQ_STRIDE: tl.constexpr,
    ANCHOR_STRIDE: tl.constexpr,
    BONUS_STRIDE: tl.constexpr,
    LIVE_STRIDE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    CONTEXT_STRIDE: tl.constexpr,
    MROPE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    MASK_TOKEN: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
    DISCARD_SLOT: tl.constexpr,
    PHYSICAL: tl.constexpr,
    MROPE: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(Req + row * REQ_STRIDE).to(tl.int64)
    anchor = tl.load(Anchors + row * ANCHOR_STRIDE).to(tl.int64)
    active = tl.load(Live + row * LIVE_STRIDE) & (anchor + WIDTH <= MAX_CONTEXT)
    bonus = tl.load(Bonus + row * BONUS_STRIDE, mask=active, other=0)
    tl.store(AnchorOut + row, anchor)
    tl.store(ActiveOut + row, active)
    q = tl.arange(0, QUERY_BLOCK)
    offset = row * WIDTH + q
    positions = tl.where(active, anchor + q, q)
    slots = tl.load(Slots + offset, mask=(q < WIDTH) & active, other=DISCARD_SLOT)
    tl.store(InputOut + offset, tl.where(q == 0, bonus, MASK_TOKEN), q < WIDTH)
    tl.store(PositionOut + offset, positions, q < WIDTH)
    tl.store(VerifyOut + offset, tl.minimum(anchor + q, MAX_CONTEXT - 1), q < WIDTH)
    tl.store(SlotOut + offset, slots, q < WIDTH)
    if MROPE:
        delta = tl.load(Delta + req)
        rope = positions + tl.where(active, delta, 0)
        for axis in tl.static_range(3):
            tl.store(MropeOut + axis * MROPE_STRIDE + offset, rope, q < WIDTH)
    for start in range(tl.cdiv(HIDDEN_SIZE, HIDDEN_BLOCK)):
        h = start * HIDDEN_BLOCK + tl.arange(0, HIDDEN_BLOCK)
        hidden = tl.load(Hidden + req * HIDDEN_SIZE + h, h < HIDDEN_SIZE, other=0)
        tl.store(HiddenOut + row * HIDDEN_SIZE + h, hidden, h < HIDDEN_SIZE)
    if PHYSICAL and WINDOW > 0:
        w = tl.arange(0, WINDOW_BLOCK)
        pos = anchor - WINDOW + w
        ids = tl.load(
            ReqToToken + req * CONTEXT_STRIDE + pos,
            (w < WINDOW) & (pos >= 0) & active,
            other=-1,
        ).to(tl.int64)
        # Sort once per request; physical slots need not follow logical order.
        ids = tl.sort(tl.where(w < WINDOW, ids, 0x7FFFFFFFFFFFFFFF), descending=False)
        tl.store(SortedWindow + row * WINDOW + w, ids, w < WINDOW)


def plan_dfly_inputs(
    *,
    req_pool_indices,
    anchors,
    bonus_tokens,
    live,
    verify_slots,
    hidden,
    mrope_delta,
    req_to_token,
    input_ids,
    positions,
    verify_positions,
    write_slots,
    out_anchors,
    active,
    target_hidden,
    mrope_positions,
    sorted_window,
    mask_token_id,
    max_context_len,
    discard_slot,
    physical,
):
    """NVIDIA/Triton fixed-output plan; int32/int64 inputs, contiguous state/slots.

    Outputs are caller-owned maximum-capacity buffers sliced to this batch.
    mRoPE may retain the maximum-row plane stride. No device storage is allocated.
    """
    bs, width = verify_slots.shape
    device = anchors.device
    ints = (torch.int32, torch.int64)
    for name, tensor in (
        ("req_pool_indices", req_pool_indices),
        ("anchors", anchors),
        ("bonus_tokens", bonus_tokens),
    ):
        _check_tensor(
            tensor, shape=(bs,), dtype=ints, device=device, name=name, contiguous=False
        )
    _check_tensor(
        live,
        shape=(bs,),
        dtype=(torch.bool,),
        device=device,
        name="live",
        contiguous=False,
    )
    _check_tensor(
        verify_slots, shape=(bs, width), dtype=ints, device=device, name="verify_slots"
    )
    for name, tensor, shape, dtype in (
        ("input_ids", input_ids, (bs * width,), (torch.int64,)),
        ("positions", positions, (bs, width), (torch.int64,)),
        ("verify_positions", verify_positions, (bs, width), (torch.int64,)),
        ("write_slots", write_slots, (bs * width,), (torch.int64,)),
        ("out_anchors", out_anchors, (bs,), (torch.int64,)),
        ("active", active, (bs,), (torch.bool,)),
        (
            "hidden",
            hidden,
            (req_to_token.shape[0], hidden.shape[1]),
            (torch.bfloat16, torch.float16, torch.float32),
        ),
        ("target_hidden", target_hidden, (bs, hidden.shape[1]), (hidden.dtype,)),
        ("mrope_delta", mrope_delta, (hidden.shape[0],), (torch.int64,)),
        ("req_to_token", req_to_token, (hidden.shape[0], req_to_token.shape[1]), ints),
        ("sorted_window", sorted_window, (bs, sorted_window.shape[1]), (torch.int64,)),
    ):
        _check_tensor(tensor, shape=shape, dtype=dtype, device=device, name=name)
    if mrope_positions is not None:
        _check_tensor(
            mrope_positions,
            shape=(3, bs * width),
            dtype=(torch.int64,),
            device=device,
            name="mrope_positions",
            contiguous=False,
        )
        if mrope_positions.stride(1) != 1 or mrope_positions.stride(0) < bs * width:
            raise ValueError("DFly mRoPE must have contiguous, nonoverlapping rows")
    window = sorted_window.shape[1]
    if not 0 < width <= 128 or not 0 <= window <= 1024:
        raise ValueError(
            "DFly fused plan supports query widths 1..128 and windows 0..1024"
        )
    if bs == 0:
        return
    _plan_dfly_inputs_kernel[(bs,)](
        req_pool_indices,
        anchors,
        bonus_tokens,
        live,
        verify_slots,
        hidden,
        mrope_delta,
        req_to_token,
        input_ids,
        positions,
        verify_positions,
        write_slots,
        out_anchors,
        active,
        target_hidden,
        mrope_positions,
        sorted_window,
        req_pool_indices.stride(0),
        anchors.stride(0),
        bonus_tokens.stride(0),
        live.stride(0),
        hidden.shape[1],
        req_to_token.stride(0),
        mrope_positions.stride(0) if mrope_positions is not None else 0,
        width,
        window,
        mask_token_id,
        max_context_len,
        discard_slot,
        physical,
        mrope_positions is not None,
        triton.next_power_of_2(width),
        triton.next_power_of_2(max(window, 1)),
        256,
    )
