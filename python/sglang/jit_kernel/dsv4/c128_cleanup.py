from __future__ import annotations

from collections.abc import Iterable

import torch
import triton
import triton.language as tl

_BLOCK_D = 256
_MAX_GRID_Z = 65535


@triton.jit
def _clear_c128_draft_state(
    state,
    req_pool_indices,
    seq_lens,
    accept_lens,
    bid,
    draft_offset,
    block_id,
    ring_size: tl.constexpr,
    half: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    accept_len = tl.load(accept_lens + bid)
    if draft_offset >= accept_len:
        req_pool_idx = tl.load(req_pool_indices + bid).to(tl.int64)
        seq_len = tl.load(seq_lens + bid).to(tl.int64)
        slot = (seq_len + draft_offset) % ring_size
        row = req_pool_idx * ring_size + slot

        offsets = block_id * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offsets < half
        row_base = row * (half * 2)
        tl.store(state + row_base + offsets, 0.0, mask=mask)
        tl.store(state + row_base + half + offsets, float("-inf"), mask=mask)


@triton.jit
def _clear_unaccepted_c128_draft_states_kernel(
    state,
    req_pool_indices,
    seq_lens,
    accept_lens,
    ring_size: tl.constexpr,
    half: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    _clear_c128_draft_state(
        state,
        req_pool_indices,
        seq_lens,
        accept_lens,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        ring_size,
        half,
        BLOCK_D,
    )


@triton.jit
def _fused_clear_c128_draft_states_kernel(
    state_ptrs,
    req_pool_indices,
    seq_lens,
    accept_lens,
    ring_size: tl.constexpr,
    half: tl.constexpr,
    STATE_DTYPE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bid = tl.program_id(0)
    draft_offset = tl.program_id(1)
    layer_block_id = tl.program_id(2)

    num_dim_blocks = tl.cdiv(half, BLOCK_D)
    layer_id = layer_block_id // num_dim_blocks
    block_id = layer_block_id % num_dim_blocks

    state_address = tl.load(state_ptrs + layer_id)
    state = state_address.to(tl.pointer_type(STATE_DTYPE))
    _clear_c128_draft_state(
        state,
        req_pool_indices,
        seq_lens,
        accept_lens,
        bid,
        draft_offset,
        block_id,
        ring_size,
        half,
        BLOCK_D,
    )


class C128DraftCleanup:
    """Validated fused cleanup for a pool-stable list of C128 states.

    The object strongly owns every state tensor and keeps its raw pointer table
    private so their lifetime, dtype, count, and addresses cannot diverge through
    the public API. Rebuild the object if any state tensor is reallocated.
    """

    __slots__ = ("_states", "_state_ptrs", "_ring_size", "_half")

    def __init__(self, states: Iterable[torch.Tensor], *, ring_size: int) -> None:
        states = tuple(states)
        if not states:
            raise ValueError("C128 draft cleanup requires at least one state tensor")
        if ring_size <= 0:
            raise ValueError(f"ring_size must be positive, got {ring_size}")
        if torch.version.cuda is None:
            raise ValueError("fused C128 draft cleanup requires NVIDIA CUDA")

        first_state = states[0]
        if not first_state.is_cuda:
            raise ValueError("C128 draft cleanup states must be CUDA tensors")
        if first_state.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "C128 draft cleanup states must use float32 or bfloat16, "
                f"got {first_state.dtype}"
            )
        if first_state.ndim != 2:
            raise ValueError(
                "C128 draft cleanup states must be 2-D, "
                f"got shape {tuple(first_state.shape)}"
            )
        if not first_state.is_contiguous():
            raise ValueError("C128 draft cleanup states must be contiguous")
        if first_state.shape[1] <= 0 or first_state.shape[1] % 2 != 0:
            raise ValueError(
                "C128 draft cleanup state width must be positive and even, "
                f"got {first_state.shape[1]}"
            )

        expected_shape = first_state.shape
        expected_stride = first_state.stride()
        expected_dtype = first_state.dtype
        expected_device = first_state.device
        for layer_id, state in enumerate(states):
            if not state.is_cuda or state.device != expected_device:
                raise ValueError(
                    "C128 draft cleanup states must share one CUDA device, "
                    f"layer {layer_id} is on {state.device}, "
                    f"expected {expected_device}"
                )
            if state.dtype != expected_dtype:
                raise ValueError(
                    "C128 draft cleanup states must share one dtype, "
                    f"layer {layer_id} uses {state.dtype}, expected {expected_dtype}"
                )
            if state.shape != expected_shape or state.stride() != expected_stride:
                raise ValueError(
                    "C128 draft cleanup states must share shape and stride, "
                    f"layer {layer_id} has shape={tuple(state.shape)}, "
                    f"stride={state.stride()}, "
                    f"expected shape={tuple(expected_shape)}, "
                    f"stride={expected_stride}"
                )
            if not state.is_contiguous():
                raise ValueError(
                    f"C128 draft cleanup state at layer {layer_id} is not contiguous"
                )

        half = first_state.shape[1] // 2
        num_dim_blocks = triton.cdiv(half, _BLOCK_D)
        grid_z = len(states) * num_dim_blocks
        if grid_z > _MAX_GRID_Z:
            raise ValueError(
                "C128 draft cleanup grid exceeds CUDA grid.z limit: "
                f"{len(states)} states * {num_dim_blocks} blocks = {grid_z}"
            )

        self._states = states
        self._state_ptrs = torch.tensor(
            [state.data_ptr() for state in states],
            dtype=torch.int64,
            device=expected_device,
        )
        self._ring_size = ring_size
        self._half = half

    def clear(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        *,
        num_draft_tokens: int,
    ) -> None:
        """Clear rejected C128 draft rows across every state in one launch."""
        if num_draft_tokens <= 1 or req_pool_indices.numel() == 0:
            return

        num_dim_blocks = triton.cdiv(self._half, _BLOCK_D)
        grid = (
            req_pool_indices.numel(),
            num_draft_tokens,
            len(self._states) * num_dim_blocks,
        )
        _fused_clear_c128_draft_states_kernel[grid](
            self._state_ptrs,
            req_pool_indices,
            seq_lens,
            accept_lens,
            ring_size=self._ring_size,
            half=self._half,
            STATE_DTYPE=(
                tl.float32 if self._states[0].dtype == torch.float32 else tl.bfloat16
            ),
            BLOCK_D=_BLOCK_D,
            num_warps=4,
        )


def clear_unaccepted_c128_draft_states(
    state: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    ring_size: int,
    num_draft_tokens: int,
) -> None:
    half = state.shape[-1] // 2
    _clear_unaccepted_c128_draft_states_kernel[
        (req_pool_indices.numel(), num_draft_tokens, triton.cdiv(half, _BLOCK_D))
    ](
        state,
        req_pool_indices,
        seq_lens,
        accept_lens,
        ring_size,
        half,
        BLOCK_D=_BLOCK_D,
    )
