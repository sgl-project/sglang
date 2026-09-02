"""Guards for the planned-copy replay (`copy_cache_planned_mla`).

Replay is the prefetch feature's only writer into the device buffer, and it is
driven entirely by device-side plan tensors: a fixed grid of one-warp workers
walks every plan row and round-robins that row's items across all workers, so
correctness rests on three things no other registered test exercises -- the
rotation stays a bijection onto the worker set (drop it and items vanish
silently), `num_real_reqs` bounds the walk on the device (ignore it and a reused
plan buffer replays the previous batch's rows), and `skip_io` walks the plan
without writing (the timing probe must not corrupt the buffer).

Item lengths are part of the contract too: on sm120 the linear path copies
through `transfer_item_warp_planned`, which rounds the read base down to 64B and
predicates the leading stores, so a length that is not a multiple of 64 reads
bytes belonging to the neighbouring item. Those bytes must never reach the
destination -- 656B and 1152B below are exactly the lengths that exercise it.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
from sglang.srt.utils import is_cuda, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_npu() or is_xpu() or not is_cuda(),
    reason="The planned-copy kernel needs a CUDA device.",
)

DEVICE = "cuda"
NUM_REQS = 6
PLAN_STRIDE = 8  # top_k
HOST_ROWS = 64
DEV_SLOTS = NUM_REQS * PLAN_STRIDE
NUM_BLOCKS = 2
BLOCK_SIZE = 256


def _buffers(item_bytes: int):
    """Distinct byte pattern per host row, so a mis-addressed copy shows up."""
    host = torch.empty(HOST_ROWS * item_bytes, dtype=torch.uint8, pin_memory=True)
    host.copy_(
        (torch.arange(HOST_ROWS * item_bytes, dtype=torch.int64) * 37 % 251).to(
            torch.uint8
        )
    )
    return host.view(HOST_ROWS, item_bytes), torch.zeros(
        DEV_SLOTS, item_bytes, dtype=torch.uint8, device=DEVICE
    )


def _plan(counts: list[int], num_real: int):
    """Plan rows with unique src rows and unique dst slots per entry."""
    g = torch.arange(NUM_REQS * PLAN_STRIDE, dtype=torch.int64)
    src = (g % HOST_ROWS).view(NUM_REQS, PLAN_STRIDE).to(DEVICE)
    dst = g.view(NUM_REQS, PLAN_STRIDE).to(torch.int32).to(DEVICE)
    cnt = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    real = torch.tensor([num_real], dtype=torch.int32, device=DEVICE)
    return src, dst, cnt, real


def _expected(host, src, dst, counts: list[int], num_real: int, item_bytes: int):
    out = torch.zeros(DEV_SLOTS, item_bytes, dtype=torch.uint8)
    for r in range(num_real):
        for m in range(counts[r]):
            out[int(dst[r, m])] = host[int(src[r, m])]
    return out


def _replay(host, dev, src, dst, cnt, real, item_bytes, skip_io=False):
    copy_cache_planned_mla(
        miss_src=src,
        miss_dst=dst,
        miss_count=cnt,
        num_real_reqs=real,
        host_cache=host,
        device_buffer=dev,
        item_size_bytes=item_bytes,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        skip_io=skip_io,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("item_bytes", [8, 512, 656, 1152])
def test_replay_is_byte_exact_for_ragged_rows(item_bytes: int) -> None:
    """Every live plan entry lands byte-exactly, and nothing else is touched.

    656B is the length whose 64B-aligned read reaches into the neighbouring
    item (GLM-5.2 NVFP4 with fp8_e4m3 KV); 1152B is a multiple of 64 that needs
    more than one unrolled trip; 8B is a single tail chunk with no 16B body.
    """
    counts = [3, 0, PLAN_STRIDE, 1, 5, 2]
    host, dev = _buffers(item_bytes)
    src, dst, cnt, real = _plan(counts, NUM_REQS)

    _replay(host, dev, src, dst, cnt, real, item_bytes)

    assert torch.equal(
        dev.cpu(), _expected(host, src.cpu(), dst.cpu(), counts, NUM_REQS, item_bytes)
    )


def test_partial_batch_ignores_rows_beyond_num_real_reqs() -> None:
    """`num_real_reqs` is a device-side bound; padded rows keep stale counts.

    The coordinator reuses one plan buffer across batch sizes and only refills
    the live prefix, so a replay that walked all rows would copy last batch's
    items into this batch's slots.
    """
    item_bytes = 656
    counts = [4, 2, 7, 3, 1, 5]
    num_real = 3
    host, dev = _buffers(item_bytes)
    src, dst, cnt, real = _plan(counts, num_real)

    _replay(host, dev, src, dst, cnt, real, item_bytes)

    assert torch.equal(
        dev.cpu(), _expected(host, src.cpu(), dst.cpu(), counts, num_real, item_bytes)
    )
    # The dst slots owned by the padded rows must still be zero.
    assert not dev[num_real * PLAN_STRIDE :].any()


def test_skip_io_moves_no_bytes() -> None:
    """The timing probe walks the whole plan but must not write the buffer."""
    item_bytes = 656
    counts = [PLAN_STRIDE] * NUM_REQS
    host, dev = _buffers(item_bytes)
    src, dst, cnt, real = _plan(counts, NUM_REQS)

    _replay(host, dev, src, dst, cnt, real, item_bytes, skip_io=True)

    assert not dev.any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
