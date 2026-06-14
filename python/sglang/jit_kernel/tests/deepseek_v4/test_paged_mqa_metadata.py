"""Unit tests for paged_mqa_metadata JIT kernel.

Verifies byte-equal correctness against a pure-PyTorch reference oracle
across the shape envelope. Output is int32 ``[num_sm + 1, 2]`` — a
deterministic partition table — so equality is strict (``torch.equal``,
no atol/rtol).

Test groups:

1. ``test_matches_pytorch_ref`` — random-input envelope sweep over
   ``bs x max_ctx`` (powers-of-2 + off-by-one for ``bs`` to stress the
   ``ret`` branch where ``bs % num_sm != 0``; kSplitKV=256 boundary
   values for ``max_ctx``).

2. ``test_matches_pytorch_ref_at_ksplitkv_boundary`` — hand-crafted
   ``seq_lens`` straddling the internal ``kSplitKV=256`` boundary
   (catches off-by-one in ``ceil(len/256)``).

3. ``test_byte_equal_at_correctness_floor`` — ``bs`` above the smem-path
   ceiling (``bs > 32768``); exercises the multi-block gmem path. Catches
   regressions where a future kernel adds a ``batch_size`` upper bound for
   smem convenience.

4. ``test_matches_pytorch_ref_at_large_num_sm`` — ``num_sm in [1, 1024]``
   contract; guards against per-block thread-guard truncation across the
   three dispatch paths.

5. ``test_matches_deep_gemm`` — byte-equality against the production
   ``deep_gemm`` oracle; auto-skips at ``bs >= 16384`` where deep_gemm
   exceeds sm_90's smem cap.
"""

import itertools

import pytest
import torch

from sglang.jit_kernel.dsv4 import get_paged_mqa_logits_metadata
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


KSPLITKV = 256  # internal kernel constant (note: public API page_size=64 is unrelated)
NUM_SM = (
    132  # H200 reference; CI box may have fewer SMs but the kernel is SM-count agnostic
)
PAGE_SIZE = 64
DEVICE = "cuda"


# Algorithmic spec for the int32 [num_sm+1, 2] schedule. Ground-truth
# correctness is gated by ``test_matches_deep_gemm`` below; this ref is
# used by the wider envelope tests where deep_gemm exceeds the sm_90
# smem cap.
def paged_mqa_metadata_ref(
    seq_lens: torch.Tensor, num_sm: int, page_size: int
) -> torch.Tensor:
    assert page_size == 64, f"page_size must be 64, got {page_size}"
    assert (
        seq_lens.dtype == torch.int32
    ), f"seq_lens dtype must be int32, got {seq_lens.dtype}"
    assert (
        seq_lens.dim() == 1
    ), f"seq_lens must be 1-D, got shape {tuple(seq_lens.shape)}"

    device = seq_lens.device
    batch_size = int(seq_lens.shape[0])

    work_per_batch = (seq_lens.to(torch.int64) + KSPLITKV - 1) // KSPLITKV
    global_sum = int(work_per_batch.sum().item())
    avg = global_sum // num_sm
    ret = global_sum % num_sm

    schedule_metadata = torch.empty((num_sm + 1, 2), dtype=torch.int32, device=device)
    work = work_per_batch.tolist()
    q = 0
    sum_work = work[0] if batch_size > 0 else 0
    for i in range(num_sm + 1):
        target = i * avg + min(i, ret)
        while sum_work <= target:
            q += 1
            if q >= batch_size:
                break
            sum_work += work[q]
        if q >= batch_size:
            schedule_metadata[i, 0] = batch_size
            schedule_metadata[i, 1] = 0
        else:
            schedule_metadata[i, 0] = q
            schedule_metadata[i, 1] = target - (sum_work - work[q])
    return schedule_metadata


# -----------------------------------------------------------------------------
# Shape envelope
#
# bs values:
#   1               single-request decode (smallest realistic input)
#   17, 129, 257    non-power-of-2; Phase 3 q-advance hits `ret` branch
#                   (bs % num_sm != 0 → uneven work split)
#   1025            ditto, large-bs ret-branch stressor
#   32, 128, 512, 1024, 2048  DSv4 decode/prefill realistic batches
#   4096..32768     multi-block path (bs > kSmallMax = 2048)
#
# max_ctx values:
#   1               degenerate (all seq_lens == 1 → work_per_batch == 1)
#   255             just under kSplitKV=256 boundary (ceil(255/256) = 1)
#   256             exactly at kSplitKV boundary (ceil(256/256) = 1)
#   257             just over boundary (ceil(257/256) = 2)
#   2048, 8192      realistic decode/short-prefill contexts
#   32768           long-context upper bound
# -----------------------------------------------------------------------------
BS_FULL = [1, 17, 32, 128, 129, 257, 512, 1024, 1025, 2048, 4096, 8192, 16384, 32768]
MAX_CTX_FULL = [1, 255, 256, 257, 2048, 8192, 32768]

# bs values above the in-smem path's ceiling (kSmallMax = 2048; multi-block
# path takes over). Tested separately to keep the CI parametrize matrix
# small while still guarding the gmem path.
BS_CORRECTNESS_FLOOR = [65536, 131072]

BS_LIST = get_ci_test_range(
    full_range=BS_FULL,
    ci_range=[1, 128, 1025],  # tiny + typical decode + large ret-branch
)
MAX_CTX_LIST = get_ci_test_range(
    full_range=MAX_CTX_FULL,
    ci_range=[256, 8192],  # kSplitKV boundary + typical decode
)


def _make_seq_lens(bs: int, max_ctx: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randint(
        1, max_ctx + 1, (bs,), dtype=torch.int32, device=DEVICE, generator=g
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("bs,max_ctx", list(itertools.product(BS_LIST, MAX_CTX_LIST)))
def test_matches_pytorch_ref(bs: int, max_ctx: int):
    """Kernel output bit-exact vs PyTorch reference across the shape envelope."""
    seq_lens = _make_seq_lens(bs, max_ctx)
    got = get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, NUM_SM)
    ref = paged_mqa_metadata_ref(seq_lens, NUM_SM, PAGE_SIZE)
    assert torch.equal(got, ref), (
        f"kernel != ref for bs={bs} max_ctx={max_ctx}\n"
        f"  kernel first row: {got[0].tolist()}\n"
        f"  ref    first row: {ref[0].tolist()}"
    )


_KSPLITKV_BOUNDARY_LENS = [
    [1],  # minimum
    [256],  # exact kSplitKV multiple
    [255, 256, 257],  # straddle boundary
    [256] * 132,  # all-equal at boundary, bs == num_sm
    [1] * 131 + [32768],  # one giant + rest minimum (skewed)
    [1, 256, 512, 768, 1024],  # exact multiples
    [255, 511, 767, 1023, 1279],  # one below each multiple
    [257, 513, 769, 1025, 1281],  # one above each multiple
]


@pytest.mark.parametrize("seq_lens_data", _KSPLITKV_BOUNDARY_LENS)
def test_matches_pytorch_ref_at_ksplitkv_boundary(seq_lens_data):
    """Bit-exact vs ref on hand-crafted kSplitKV=256 boundary inputs.

    Catches off-by-one in ceil(len/256) (Phase 1) and uneven work-split
    in Phase 3's advance loop.
    """
    seq_lens = torch.tensor(seq_lens_data, dtype=torch.int32, device=DEVICE)
    got = get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, NUM_SM)
    ref = paged_mqa_metadata_ref(seq_lens, NUM_SM, PAGE_SIZE)
    assert torch.equal(got, ref), (
        f"kernel != ref for seq_lens={seq_lens_data}\n"
        f"  kernel: {got.tolist()}\n  ref:    {ref.tolist()}"
    )


@pytest.mark.parametrize("bs", BS_CORRECTNESS_FLOOR)
@pytest.mark.parametrize("max_ctx", [8192])
def test_byte_equal_at_correctness_floor(bs: int, max_ctx: int):
    """bs above smem-path ceiling: multi-block path must remain byte-equal.

    Guards against a future kernel adding a ``batch_size`` upper bound for
    smem convenience and breaking the gmem fallback.
    """
    seq_lens = _make_seq_lens(bs, max_ctx)
    got = get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, NUM_SM)
    ref = paged_mqa_metadata_ref(seq_lens, NUM_SM, PAGE_SIZE)
    assert torch.equal(got, ref), f"kernel != ref at correctness-floor bs={bs}"


# -----------------------------------------------------------------------------
# Defensive: kernel must handle the full num_sm range [1, 1024] guaranteed
# by the public API contract. The internal dispatch uses smaller per-block
# thread counts on some paths, so the per-target write loop must stride
# rather than use a `tx <= num_sm` thread guard.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bs,num_sm",
    [
        # bs spans the tiny / small / multi-block paths.
        # num_sm in {256, 257, 500, 1024} crosses the boundary where a
        # 256-thread block would silently truncate schedule_metadata.
        (64, 256),
        (64, 257),
        (64, 1024),
        (128, 257),
        (128, 1024),
        (8192, 257),
        (8192, 1024),
    ],
)
def test_matches_pytorch_ref_at_large_num_sm(bs: int, num_sm: int):
    """schedule_metadata[0 .. num_sm] must be fully populated for any
    num_sm in [1, 1024], regardless of which dispatch path bs selects."""
    seq_lens = _make_seq_lens(bs, max_ctx=8192)
    got = get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, num_sm)
    ref = paged_mqa_metadata_ref(seq_lens, num_sm, PAGE_SIZE)
    assert torch.equal(got, ref), (
        f"kernel != ref at bs={bs} num_sm={num_sm}; "
        f"last-5 rows: got={got[-5:].tolist()} ref={ref[-5:].tolist()}"
    )


def _to_2d_context_lens(seq_lens: torch.Tensor) -> torch.Tensor:
    return seq_lens.contiguous().view(-1, 1)


def _load_deep_gemm():
    try:
        import deep_gemm  # noqa: PLC0415

        return deep_gemm
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"deep_gemm unavailable: {type(e).__name__}: {e}")


@pytest.mark.parametrize("bs", [64, 128, 1025, 2048, 4096, 8192, 32768, 65536])
@pytest.mark.parametrize("max_ctx", [256, 32768])
def test_matches_deep_gemm(bs: int, max_ctx: int):
    """Byte-equal against the production deep_gemm reference. Auto-skips
    at large bs where deep_gemm exceeds sm_90 smem cap."""
    deep_gemm = _load_deep_gemm()
    seq_lens = _make_seq_lens(bs, max_ctx)
    got = get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, NUM_SM)
    try:
        dg = deep_gemm.get_paged_mqa_logits_metadata(
            _to_2d_context_lens(seq_lens), PAGE_SIZE, NUM_SM
        )
    except RuntimeError as e:
        msg = str(e)
        if "smem" in msg.lower() or "capacity" in msg.lower():
            pytest.skip(
                f"deep_gemm smem cap exceeded at bs={bs}: {msg.splitlines()[0]}"
            )
        raise
    assert torch.equal(got, dg), (
        f"kernel != deep_gemm for bs={bs} max_ctx={max_ctx}\n"
        f"  kernel first row: {got[0].tolist()}\n"
        f"  dg     first row: {dg[0].tolist()}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
