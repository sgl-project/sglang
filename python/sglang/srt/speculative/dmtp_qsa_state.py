"""Owned request-slot storage for prefetched T1 outputs, not KV storage."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _transfer(
    SOURCES,
    DESTINATIONS,
    SLOTS,
    WIDTHS: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row)
    source_row = row if SCATTER else slot
    destination_row = slot if SCATTER else row
    for field in tl.static_range(len(WIDTHS)):
        width = WIDTHS[field]
        value = tl.load(
            SOURCES[field] + source_row * width + offset, offset < width, other=0
        )
        tl.store(
            DESTINATIONS[field] + destination_row * width + offset,
            value,
            offset < width,
        )


class PrefetchedT1State:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffers = None

    def save(self, slots, values):
        if self.buffers is None:
            self.buffers = tuple(
                value.new_empty((self.capacity, *value.shape[1:])) for value in values
            )
        self._copy(slots, values, scatter=True)

    def load(self, slots):
        if self.buffers is None:
            raise RuntimeError("T1 request state has not been initialized")
        values = tuple(
            value.new_empty((slots.numel(), *value.shape[1:])) for value in self.buffers
        )
        self._copy(slots, values, scatter=False)
        return values

    def load_checked(
        self,
        slots,
        expected_slots,
        lengths,
        block_ids,
        temperatures,
        greedy,
        valid_slots=None,
    ):
        values, valid = self._checked_values(
            slots, expected_slots, lengths, block_ids, temperatures, greedy, valid_slots
        )
        torch._assert_async(
            valid.all(), "Stale QSA T1 slot/length/anchor/sampling state"
        )
        return values

    def _checked_values(
        self,
        slots,
        expected_slots,
        lengths,
        block_ids,
        temperatures,
        greedy,
        valid_slots=None,
    ):
        if (
            self.buffers is None
            or len(self.buffers) != 8
            or self.buffers[0].shape != (self.capacity, 1)
            or any(b.shape != (self.capacity,) for b in self.buffers[4:])
            or self.buffers[4].dtype != torch.float32
            or self.buffers[5].dtype != torch.bool
        ):
            raise ValueError("Checked T1 reads require GPU request state fields")
        bs = slots.numel()
        if valid_slots is not None and (
            valid_slots.shape != (self.capacity,)
            or valid_slots.dtype != torch.bool
            or valid_slots.device != slots.device
            or not valid_slots.is_contiguous()
        ):
            raise ValueError("Invalid T1 request validity storage")
        if (
            slots.ndim != 1
            or slots.dtype != torch.int64
            or not bs
            or block_ids.shape != (bs, 8)
            or block_ids.dtype != torch.int64
            or temperatures.dtype != torch.float32
            or greedy.dtype != torch.bool
            or expected_slots.dtype != torch.int64
            or lengths.dtype not in (torch.int32, torch.int64)
            or any(
                t.shape != (bs,)
                for t in (expected_slots, lengths, temperatures, greedy)
            )
            or any(
                t.device != slots.device
                for t in (
                    *self.buffers,
                    expected_slots,
                    lengths,
                    block_ids,
                    temperatures,
                    greedy,
                )
            )
        ):
            raise ValueError("Invalid checked T1 request layout")
        values = tuple(
            value.new_empty((bs, *value.shape[1:])) for value in self.buffers[:4]
        )
        if slots.is_cuda:
            valid = torch.empty(bs, device=slots.device, dtype=torch.bool)
            widths = tuple(math.prod(value.shape[1:]) for value in values)
            _load_checked[(bs, triton.cdiv(max(widths), 256))](
                self.buffers,
                values,
                slots,
                expected_slots,
                lengths,
                block_ids,
                temperatures,
                greedy,
                valid,
                valid_slots if valid_slots is not None else slots,
                valid_slots is not None,
                widths,
                self.capacity,
                slots.stride(0),
                expected_slots.stride(0),
                lengths.stride(0),
                *block_ids.stride(),
                temperatures.stride(0),
                greedy.stride(0),
                256,
            )
        else:
            bounded = (slots > 0) & (slots < self.capacity)
            safe = slots.clamp(0, self.capacity - 1)
            valid = (
                bounded
                & (slots == expected_slots)
                & (self.buffers[6][safe] == lengths)
                & (self.buffers[7][safe] == block_ids[:, 0])
                & (self.buffers[4][safe] == temperatures)
                & (self.buffers[5][safe] == greedy)
            )
            if valid_slots is not None:
                valid &= valid_slots[safe]
            for value, buffer in zip(values, self.buffers):
                torch.index_select(buffer, 0, safe, out=value)
            block_ids[:, 1] = torch.where(valid, values[0][:, 0], block_ids[:, 1])
        return values, valid

    def _copy(self, slots, values, *, scatter):
        if slots.ndim != 1 or slots.dtype != torch.int64:
            raise ValueError("T1 slots must be a vector of int64 request indices")
        if len(values) != len(self.buffers):
            raise ValueError("T1 state field count changed")
        for value, buffer in zip(values, self.buffers):
            if (
                value.shape != (slots.numel(), *buffer.shape[1:])
                or value.dtype != buffer.dtype
                or value.device != buffer.device
                or slots.device != buffer.device
            ):
                raise ValueError("T1 state layout changed")
        # Slots are validated by the controller's request-identity checks and
        # originate in ReqToTokenPool, including its reserved dummy row.
        if slots.is_cuda:
            contiguous = tuple(value.contiguous() for value in values)
            widths = tuple(math.prod(value.shape[1:]) for value in values)
            sources, destinations = (
                (contiguous, self.buffers) if scatter else (self.buffers, values)
            )
            _transfer[(slots.numel(), triton.cdiv(max(widths), 256))](
                sources, destinations, slots, widths, scatter, 256
            )
        elif scatter:
            for buffer, value in zip(self.buffers, values):
                buffer.index_copy_(0, slots, value)
        else:
            for buffer, value in zip(self.buffers, values):
                torch.index_select(buffer, 0, slots, out=value)


@triton.jit
def _load_checked(
    SOURCES,
    OUTPUTS,
    SLOTS,
    EXPECTED,
    LENGTHS,
    IDS,
    TEMP,
    GREEDY,
    VALID,
    VALID_SLOTS,
    HAS_VALID: tl.constexpr,
    WIDTHS: tl.constexpr,
    CAPACITY: tl.constexpr,
    S0: tl.constexpr,
    E0: tl.constexpr,
    L0: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    T0: tl.constexpr,
    G0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + row * S0)
    bounded = (slot > 0) & (slot < CAPACITY)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    for field in tl.static_range(4):
        width = WIDTHS[field]
        value = tl.load(
            SOURCES[field] + slot * width + offset, bounded & (offset < width), other=0
        )
        tl.store(OUTPUTS[field] + row * width + offset, value, offset < width)
    if tile == 0:
        ok = bounded & (slot == tl.load(EXPECTED + row * E0))
        if HAS_VALID:
            ok &= tl.load(VALID_SLOTS + slot, bounded, other=0)
        ok &= tl.load(SOURCES[6] + slot, bounded, other=0) == tl.load(
            LENGTHS + row * L0
        )
        ok &= tl.load(SOURCES[7] + slot, bounded, other=0) == tl.load(IDS + row * I0)
        ok &= tl.load(SOURCES[4] + slot, bounded, other=0) == tl.load(TEMP + row * T0)
        ok &= tl.load(SOURCES[5] + slot, bounded, other=0) == tl.load(GREEDY + row * G0)
        tl.store(VALID + row, ok)
        token = tl.load(SOURCES[0] + slot, bounded, other=0)
        tl.store(IDS + row * I0 + I1, token, ok)
