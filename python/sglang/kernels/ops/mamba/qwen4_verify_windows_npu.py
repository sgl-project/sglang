"""Copy each speculative step’s state window directly to its scratch row.

The native BF16 conv_input and native convolution are unchanged. Reading that
already-rounded input preserves state -> input -> scratch dtype conversions.
No formal state is updated here; the existing acceptance commit owns that.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _vector_cores(device):
    return triton.runtime.driver.active.utils.get_device_properties(device)["num_vectorcore"]


@triton.jit
def _store_ple_verify_windows(
    source, valid, destinations, cache,
    CHANNELS: tl.constexpr, STATE: tl.constexpr, WIDTH: tl.constexpr,
    DST_ROW: tl.constexpr, DST_STEP: tl.constexpr, INDIRECT: tl.constexpr,
    BLOCK: tl.constexpr, LOAD: tl.constexpr, OUTPUT: tl.constexpr,
    TASKS: tl.constexpr, GRID: tl.constexpr,
):
    # Reuse the NPU packed-convolution channel tile: DMA a contiguous strip
    # once, then select each overlapping window inside the vector core.
    time: tl.constexpr = STATE + WIDTH
    ix = tl.arange(0, LOAD)
    out = tl.arange(0, OUTPUT)
    live = out < BLOCK * STATE
    for task in range(tl.program_id(0), TASKS, GRID):
        row = task // (CHANNELS // BLOCK)
        channel = task % (CHANNELS // BLOCK) * BLOCK
        x = tl.load(source + (row * CHANNELS + channel) * time + ix,
                    ix < BLOCK * time, other=0)
        dest = tl.load(destinations + row).to(tl.int64) if INDIRECT else row.to(tl.int64)
        for step in tl.static_range(WIDTH):
            positions = (out // STATE) * time + (out % STATE) + step + 1
            values = tl.gather(x, tl.where(live, positions, 0), 0)
            keep = tl.load(valid + row * WIDTH + step) != 0
            tl.store(cache + dest * DST_ROW + step * DST_STEP + channel * STATE + out,
                     tl.where(keep, values, 0), live)


def store_verify_windows(source, valid, cache, width, destinations):
    rows, channels, time = source.shape
    state = time - width
    block = 128
    assert source.is_contiguous() and channels % block == 0
    assert cache.stride(-1) == 1 and cache.stride(-2) == state
    if rows:
        tasks = rows * (channels // block)
        grid = min(tasks, _vector_cores(source.device.index))
        _store_ple_verify_windows[(grid,)](
            source, valid, valid if destinations is None else destinations, cache,
            channels, state, width, cache.stride(0), cache.stride(1), destinations is not None,
            block, triton.next_power_of_2(block * time), triton.next_power_of_2(block * state),
            tasks, grid,
        )
