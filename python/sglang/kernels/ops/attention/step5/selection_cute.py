# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM10x exact stable Step5 selection bounded by each row's live history."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

_THREADS = 512
_WARPS = _THREADS // 32
_TOPK = 512
_COMPILED = {}


@dsl_user_op
def _bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _atomic_add(ptr: cute.Pointer, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.ADD,
            ptr=ptr.llvm_ptr,
            a=value.ir_value(loc=loc, ip=ip),
        )
    )


@dsl_user_op
def _lane_mask(*, loc=None, ip=None) -> Uint32:
    return Uint32(nvvm.read_ptx_sreg_lanemask_lt(T.i32(), loc=loc, ip=ip))


@cute.jit
def _ordered_key(value: Float32) -> Uint32:
    # Stable descending torch/FlashInfer semantics treat signed zero as a tie
    # and rank NaNs before finite values. Original region order breaks ties.
    if value == Float32(0.0):
        value = Float32(0.0)
    bits = _bits(value)
    key = bits ^ Uint32(0x80000000)
    if value < Float32(0.0):
        key = ~bits
    if value != value:
        key = Uint32(0xFFFFFFFF)
    return key


@cute.jit
def _warp_scan(value: Int32, lane: Int32) -> Int32:
    for index in cutlass.range_constexpr(5):
        offset = 1 << index
        other = cute.arch.shuffle_sync_up(value, offset=offset, mask_and_clamp=0)
        if lane >= Int32(offset):
            value += other
    return value


@cute.jit
def _store_region(output, mapping, row, request, slot, region, pack: cutlass.Constexpr):
    if cutlass.const_expr(pack):
        logical = Int64(region) * Int64(8)
        physical = Int64(mapping[Int64(request), logical])
        output[row, slot] = ((physical // Int64(8)) << Int64(32)) | logical
    else:
        output[row, slot] = region


@cute.kernel
def _stable_live_topk(
    scores: cute.Tensor,
    positions: cute.Tensor,
    requests: cute.Tensor,
    region_ids: cute.Tensor,
    counts: cute.Tensor,
    mapping: cute.Tensor,
    lengths: cute.Tensor,
    pack: cutlass.Constexpr,
    early_exit: cutlass.Constexpr,
    capacity: Int32,
):
    row = cute.arch.block_idx()[0]
    thread = cute.arch.thread_idx()[0]
    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()
    request = Int64(requests[row])
    live = request > Int64(0)
    current = cutlass.max(Int32(positions[row]), Int32(0)) // Int32(8)
    length = cutlass.min(current, capacity)
    if not live:
        length = Int32(0)

    @cute.struct
    class Storage:
        hist: cute.struct.MemRange[Int32, 256]
        warp_hist: cute.struct.MemRange[Int32, _WARPS * 257]
        state: cute.struct.MemRange[Int32, 8]

    storage = cutlass.utils.SmemAllocator().allocate(Storage)
    hist = storage.hist.get_tensor(cute.make_layout((256,), stride=(1,)))
    warp_hist = storage.warp_hist.get_tensor(
        cute.make_layout((_WARPS, 257), stride=(257, 1))
    )
    state = storage.state.get_tensor(cute.make_layout((8,), stride=(1,)))

    region_ids[row, thread] = Int32(-1)
    if thread == Int32(0):
        region_ids[row, Int32(_TOPK)] = Int32(-1)
        counts[row] = Int32(0)
        if cutlass.const_expr(pack):
            lengths[row] = Int32(0)
            if live:
                lengths[row] = Int32(positions[row]) + Int32(1)
    cute.arch.sync_threads()

    if live:
        if length <= Int32(_TOPK):
            if thread < length:
                _store_region(region_ids, mapping, row, request, thread, thread, pack)
        else:
            if thread == Int32(0):
                state[0] = Int32(0)
                state[1] = Int32(_TOPK - 1)
                if cutlass.const_expr(early_exit):
                    state[4] = Int32(0)
            cute.arch.sync_threads()
            tiles = (length + Int32(_THREADS - 1)) // Int32(_THREADS)

            # Refine the threshold prefix. A wholly selected boundary bucket
            # needs no lower bits: its prefix is an exact selection lower bound.
            for round_index in cutlass.range_constexpr(4):
                if cutlass.const_expr(not early_exit) or state[4] == Int32(0):
                    prefix = Uint32(state[0])
                    remaining = state[1]
                    for part in cutlass.range_constexpr(9):
                        digit = lane + Int32(part * 32)
                        if digit < Int32(257):
                            warp_hist[warp, digit] = Int32(0)
                    cute.arch.sync_threads()
                    for tile in cutlass.range(tiles, unroll=1):
                        index = tile * Int32(_THREADS) + thread
                        if index < length:
                            key = _ordered_key(Float32(scores[row, index]))
                            matches = True
                            if cutlass.const_expr(round_index > 0):
                                matches = (key >> Uint32(32 - round_index * 8)) == (
                                    prefix >> Uint32(32 - round_index * 8)
                                )
                            if matches:
                                digit = Int32(
                                    (key >> Uint32(24 - round_index * 8)) & Uint32(255)
                                )
                                pointer = warp_hist.iterator + cute.crd2idx(
                                    (warp, digit), warp_hist.layout
                                )
                                _atomic_add(pointer, Int32(1))
                    cute.arch.sync_threads()

                    count = Int32(0)
                    inclusive = Int32(0)
                    digit = Int32(255) - thread
                    if thread < Int32(256):
                        for warp_index in cutlass.range_constexpr(_WARPS):
                            count += warp_hist[Int32(warp_index), digit]
                        hist[digit] = count
                    cute.arch.sync_threads()
                    if thread < Int32(256):
                        inclusive = _warp_scan(count, lane)
                        if lane == Int32(31):
                            warp_hist[warp, 0] = inclusive
                    cute.arch.sync_threads()
                    if thread == Int32(0):
                        total = Int32(0)
                        for warp_index in cutlass.range_constexpr(8):
                            value = warp_hist[Int32(warp_index), 0]
                            warp_hist[Int32(warp_index), 1] = total
                            total += value
                    cute.arch.sync_threads()
                    if thread < Int32(256):
                        greater = warp_hist[warp, 1] + inclusive - count
                        if greater <= remaining and remaining < greater + count:
                            state[0] = Int32(
                                prefix | (Uint32(digit) << Uint32(24 - round_index * 8))
                            )
                            state[1] = remaining - greater
                            if cutlass.const_expr(early_exit):
                                if remaining - greater == count - Int32(1):
                                    state[4] = Int32(1)
                    cute.arch.sync_threads()

            threshold = Uint32(state[0])
            need_equal = state[1] + Int32(1)
            if thread == Int32(0):
                state[2] = Int32(0)
                state[3] = Int32(0)
            cute.arch.sync_threads()
            for tile in cutlass.range(tiles, unroll=1):
                index = tile * Int32(_THREADS) + thread
                is_equal = False
                is_greater = False
                if index < length:
                    key = _ordered_key(Float32(scores[row, index]))
                    is_equal = key == threshold
                    is_greater = key > threshold
                ballot = cute.arch.vote_ballot_sync(is_equal)
                equal_rank = Int32(cute.arch.popc(Uint32(ballot) & _lane_mask()))
                if lane == Int32(0):
                    warp_hist[warp, 0] = Int32(cute.arch.popc(ballot))
                cute.arch.sync_threads()
                if thread == Int32(0):
                    total = state[2]
                    for warp_index in cutlass.range_constexpr(_WARPS):
                        value = warp_hist[Int32(warp_index), 0]
                        warp_hist[Int32(warp_index), 1] = total
                        total += value
                    state[2] = total
                cute.arch.sync_threads()
                selected = is_greater | (
                    is_equal & (warp_hist[warp, 1] + equal_rank < need_equal)
                )
                ballot = cute.arch.vote_ballot_sync(selected)
                selected_rank = Int32(cute.arch.popc(Uint32(ballot) & _lane_mask()))
                if lane == Int32(0):
                    warp_hist[warp, 0] = Int32(cute.arch.popc(ballot))
                cute.arch.sync_threads()
                if thread == Int32(0):
                    total = state[3]
                    for warp_index in cutlass.range_constexpr(_WARPS):
                        value = warp_hist[Int32(warp_index), 0]
                        warp_hist[Int32(warp_index), 1] = total
                        total += value
                    state[3] = total
                cute.arch.sync_threads()
                if selected:
                    slot = warp_hist[warp, 1] + selected_rank
                    if slot < Int32(_TOPK):
                        _store_region(
                            region_ids, mapping, row, request, slot, index, pack
                        )
                cute.arch.sync_threads()

        if thread == Int32(0):
            count = cutlass.min(length, Int32(_TOPK)) + Int32(1)
            _store_region(
                region_ids, mapping, row, request, count - Int32(1), current, pack
            )
            counts[row] = count


@cute.jit
def _launch(
    scores: cute.Tensor,
    positions: cute.Tensor,
    requests: cute.Tensor,
    region_ids: cute.Tensor,
    counts: cute.Tensor,
    mapping: cute.Tensor,
    lengths: cute.Tensor,
    pack: cutlass.Constexpr,
    early_exit: cutlass.Constexpr,
    capacity: Int32,
    stream,
):
    _stable_live_topk(
        scores,
        positions,
        requests,
        region_ids,
        counts,
        mapping,
        lengths,
        pack,
        early_exit,
        capacity,
    ).launch(
        grid=[scores.shape[0], 1, 1],
        block=[_THREADS, 1, 1],
        stream=stream,
    )


def step5_select_regions_cute(
    scores: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    region_ids: torch.Tensor,
    counts: torch.Tensor,
    *,
    req_to_token: torch.Tensor | None = None,
    causal_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact stable selection, optionally writing the final sparse-attention ABI.

    Packing changes only the output representation: logical order, FP32 radix
    keys and stable ties are unchanged. Mapping uses each selected region's
    actual FULL location; physical contiguity between regions is not assumed.
    """
    pack = req_to_token is not None
    if pack:
        if (
            region_ids.dtype != torch.int64
            or causal_lengths is None
            or causal_lengths.shape != counts.shape
        ):
            raise ValueError(
                "Packed selection requires int64 output and causal lengths"
            )
    elif causal_lengths is not None or region_ids.dtype != torch.int32:
        raise ValueError("Logical selection requires int32 output")
    # Unused arguments are eliminated in the logical-only specialization.
    tensors = (
        scores,
        positions,
        requests,
        region_ids,
        counts,
        req_to_token if pack else scores,
        causal_lengths if pack else counts,
    )
    layouts = tuple(
        tuple(stride if stride in (0, 1) else -1 for stride in tensor.stride())
        for tensor in tensors
    )
    early_exit = torch.cuda.get_device_capability(scores.device) == (10, 3)
    key = scores.device.index, tuple(t.dtype for t in tensors), layouts, early_exit
    compiled = _COMPILED.get(key)
    if compiled is None:
        dtypes = {
            torch.float32: cutlass.Float32,
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
        }
        fake = [
            make_fake_tensor(
                dtypes[tensor.dtype],
                tuple(cute.sym_int() for _ in tensor.shape),
                tuple(
                    stride if stride in (0, 1) else cute.sym_int()
                    for stride in tensor.stride()
                ),
                assumed_align=1,
            )
            for tensor in tensors
        ]
        compiled = cute.compile(
            _launch,
            *fake,
            pack,
            early_exit,
            Int32(scores.shape[1]),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    compiled(*tensors, scores.shape[1])
    return region_ids, counts
