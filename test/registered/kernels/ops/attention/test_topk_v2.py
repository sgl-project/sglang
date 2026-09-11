"""Correctness tests for the DeepSeek-V4 (DSA indexer) JIT top-k transform v2.

The v2 kernel selects the per-row top-k of ``scores`` (ragged ``seq_lens``) and
writes the page-table transform of the selected raw indices into the output. We
validate against ``torch.topk`` with a small tolerance for boundary ties (the
fp16 coarse histogram can swap elements of equal score).

Coverage is organized around the kernel's dispatch so every template and its
boundaries are exercised:

  template      per-row seq            reached when
  --------      ----------             ------------
  trivial       seq <= k
  Register2     k < seq <= 8192        max_seq <= 8192          (level 0)
  Register4     8192 < seq <= 16384    max_seq <= 16384         (level 1)
  Streaming     16384 < seq <= floor   max_seq > 16384, non-cluster (level 2)
  Cluster       seq > floor(=65536)    max_seq > floor and batch <= 128

and two cluster dispatch shapes: the fused small-batch kernel (batch <= 30) and
the persistent-pool + main kernel (30 < batch <= 128). Boundary seq lengths
(8192/8193, 16384/16385, 65535/65536/65537) and batch sizes (30/31, 128/129) are
included explicitly, across k in {512,1024,2048} and identity/perm page tables.

The opt-in cooperative kernel (``SGLANG_OPT_USE_COOP_TOPK``) is a fifth path that
takes batch-1 rows away from Cluster; it has its own section further down.
"""

from __future__ import annotations

import contextlib
import sys

import pytest
import torch

from sglang.kernels.jit.utils import is_hip_runtime
from sglang.kernels.ops.attention.dsv4.topk import (
    _COOP_TOPK_MIN_FLOOR,
    _coop_topk_floor,
    _coop_topk_workspace,
    plan_topk_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=130, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
MAX_PERMIT_ERROR = 5
FLOOR = 65536  # kClusterFloor

# (batch, seq) chosen to land on each template and each dispatch boundary.
FIXED_CONFIGS = [
    # --- trivial (seq <= k) ---
    (8, 256),  # trivial for every k
    (16, 1024),  # trivial for k>=1024
    # --- Register2 (level 0: max_seq <= 8192) ---
    (8, 4096),
    (8, 8192),  # reg2 upper boundary
    (128, 8192),
    (300, 8192),  # batch > 128, still level 0
    # --- Register4 (level 1: 8192 < max_seq <= 16384) ---
    (8, 8193),  # just over reg2
    (64, 16384),  # reg4 upper boundary
    (256, 16384),  # batch > 128
    # --- Streaming (level 2: max_seq > 16384, non-cluster) ---
    (8, 16385),  # just over reg4 (small batch, seq < floor => non-cluster)
    (4, 32768),
    (16, 65535),  # just under floor
    (4, 65536),  # at floor (seq == floor => non-cluster)
    (100, 65536),
    # --- Cluster, fused small-batch kernel (batch <= 30, max_seq > floor) ---
    (1, 65537),  # single row just over floor
    (2, 131072),
    (8, 98304),
    (30, 131072),  # batch == pool boundary
    # --- Cluster, persistent pool + main kernel (30 < batch <= 128) ---
    (31, 131072),  # just over small-batch
    (40, 262144),  # N > pool of 30 => round-robin
    (64, 196608),
    (128, 131072),  # cluster batch upper boundary
    # --- batch > 128 => non-cluster streaming even at long ctx ---
    (129, 131072),
    (200, 262144),
]


def _assert_topk_close(scores_cpu, ref_raw, our_raw, bs, seq_lens, k):
    """Set-compare our top-k raw indices vs torch's, tolerating equal-score ties."""
    bad = 0
    for i in range(bs):
        L = int(seq_lens[i])
        ref, our = set(ref_raw[i]), set(our_raw[i])
        more, less = our - ref, ref - our
        if more or less:
            mv = sorted(scores_cpu[i, list(more)].tolist())
            lv = sorted(scores_cpu[i, list(less)].tolist())
            if mv != lv:  # not merely a tie swap -> genuine error
                bad += len(more)
                print(
                    f"b={i} L={L} k={k}: more={list(more)[:4]} less={list(less)[:4]} mv={mv[:3]} lv={lv[:3]}"
                )
        assert len(our) == min(k, L), (
            f"b={i} L={L} k={k}: {len(our)} valid != {min(k, L)}"
        )
    assert bad <= MAX_PERMIT_ERROR, f"{bad=} > {MAX_PERMIT_ERROR}"


def _make_page_table(batch, num_pages, mode, device, per_row=False):
    if mode == "identity":
        pt = torch.arange(num_pages, dtype=torch.int32, device=device)
        full = pt.unsqueeze(0).expand(batch, -1).contiguous()
        inv = pt.unsqueeze(0).expand(batch, -1).cpu()
        return full, inv
    # permutation (optionally a distinct permutation per row)
    rows = batch if per_row else 1
    full = torch.stack(
        [torch.randperm(num_pages, device=device) for _ in range(rows)]
    ).to(torch.int32)
    inv = torch.empty_like(full)
    ar = torch.arange(num_pages, dtype=torch.int32, device=device)
    for r in range(rows):
        inv[r, full[r].long()] = ar
    if not per_row:
        full = full.expand(batch, -1).contiguous()
        inv = inv.expand(batch, -1)
    return full, inv.cpu()


def _invert(out_row, inv_row):
    """Undo page_to_indices for one row's page indices (drop -1 padding)."""
    return [
        (int(inv_row[v >> PAGE_BITS]) << PAGE_BITS) | (v & PAGE_MASK)
        for v in out_row
        if v != -1
    ]


def _reference(scores, seq_lens, k):
    """torch.topk reference indices per row (trivial rows -> all positions)."""
    ref = []
    for i in range(scores.shape[0]):
        L = int(seq_lens[i])
        if L <= k:
            ref.append(list(range(L)))
        else:
            ref.append(
                torch.topk(scores[i, :L], k, sorted=False).indices.cpu().tolist()
            )
    return ref


def _plan(seq_lens):
    """Plan, then break stream adjacency with the transform launch.

    The transform kernel prefetches the plan metadata BEFORE its PDL wait, which
    is only legal while the plan kernel is not the immediately preceding kernel
    in the stream -- in production the plan is built during per-forward metadata
    prep, a whole model forward earlier. Launching the two back to back would
    read the plan through a programmatic dependency that guarantees no memory
    visibility, so mirror the production separation instead.
    """
    metadata = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()
    return metadata


def _run(scores, seq_lens, page_table, inv_cpu, k):
    batch = scores.shape[0]
    metadata = _plan(seq_lens)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    topk_transform_paged_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()
    out_cpu = out.cpu().tolist()
    return [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]


def _run_raw(scores, seq_lens, k):
    """Run with no page table and return the selected indices per row, dropping
    -1 padding -- the selected positions themselves, NOT a page transform."""
    batch = scores.shape[0]
    metadata = _plan(seq_lens)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    topk_transform_paged_v2(scores, seq_lens, None, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()
    out_cpu = out.cpu().tolist()
    return [[v for v in out_cpu[i] if v != -1] for i in range(batch)]


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2(batch: int, seq: int, k: int, page_mode: str) -> None:
    torch.manual_seed(batch * 100003 + seq * 7 + k)
    device = "cuda"
    # Pad the row stride to a multiple of 4 (16-byte vectorized load) while keeping
    # the exact seq_len -- this also exercises the scalar-tail path for odd seq.
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)

    our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "batch,shape",
    [
        (20, "small_batch"),  # fused small-batch kernel (<= pool of 30)
        (64, "persistent"),  # persistent pool + main kernel
        (128, "persistent"),  # cluster batch boundary
    ],
)
@pytest.mark.parametrize("per_row_pt", [False, True])
@torch.inference_mode()
def test_topk_v2_ragged(batch: int, shape: str, k: int, per_row_pt: bool) -> None:
    """Ragged lengths spanning trivial..cluster in one launch, both dispatch shapes.

    ``per_row_pt`` gives each row a distinct page-table permutation, exercising
    the per-batch page_table indexing (batch_id stride) rather than a shared one.
    """
    torch.manual_seed(7777 + batch + k + int(per_row_pt))
    device = "cuda"
    seq = 262144
    scores = torch.randn(batch, seq, dtype=torch.float32, device=device)
    # span every path; guarantee at least one > floor row so cluster dispatch fires
    buckets = [max(1, k // 2), k, 4096, 12000, 40000, 65536, 98304, 262144]
    g = torch.Generator(device="cpu").manual_seed(batch + k)
    lengths = torch.tensor(
        [
            buckets[int(torch.randint(0, len(buckets), (1,), generator=g))]
            for _ in range(batch)
        ],
        dtype=torch.int32,
        device=device,
    )
    lengths[0] = max(1, k // 2)  # a trivial row
    lengths[1] = 262144  # a long (cluster) row
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(
        batch, num_pages, "perm", device, per_row=per_row_pt
    )

    our_raw = _run(scores, lengths, page_table, inv_cpu, k)
    ref_raw = _reference(scores, lengths, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, lengths.cpu(), k)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2_output_indices(batch: int, seq: int, k: int) -> None:
    """Validate the selected indices DIRECTLY against torch.topk.

    Runs the no-page-table mode, so the output is the selected positions
    themselves. Unlike ``test_topk_v2`` -- which checks the page-transformed
    output and inverts it through the page table -- this isolates the top-k
    selection from the transform, and it is the only coverage of that mode.
    Covers every dispatch template/boundary.
    """
    torch.manual_seed(batch * 100003 + seq * 7 + k + 1)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)

    our_raw = _run_raw(scores, seq_lens, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


# --- cooperative handoff (SGLANG_OPT_USE_COOP_TOPK) --------------------------
# With the handoff on, a grid-wide cooperative kernel runs after the official one
# and takes the batch-1 rows longer than the floor, which the official kernel
# returns early for. Both read the length from device memory, so the host cannot
# tell which side a row lands on and the partition has to hold per row.
# `_assert_topk_close` already fails on either way of getting it wrong: no writer
# leaves -1 padding, two writers duplicate output slots.
COOP_FLOOR = _COOP_TOPK_MIN_FLOOR  # the lowest floor either side accepts

# CoopWorkspace::parity as an int32 word index -- the witness that the cooperative
# kernel ran. The result comparisons below are satisfied by the official kernel too,
# so by themselves they cannot see arming fail; the lone exception is the 4096-NaN
# case, where the official path selects NaN once the tie buffer overflows.
# Layout from topk_coop.cuh:59-62: hist[3][4096] = 12288 words,
# then Counters cnt[2] = 4 words, then parity. Written only at topk_coop.cuh:293,
# after the last grid.sync(); the below-floor guard (:193) returns before the first,
# so a row under the floor must leave it untouched.
_COOP_WS_PARITY_WORD = 3 * 4096 + 2 * 2

# Same layout, total size: 49,172 bytes through parity, padded to 49,176 because
# TieValue is alignas(8) (topk_impl.cuh:170), then ties[2048] -> 65,560.
_COOP_WS_BYTES = 65560


def _coop_parity(ws: torch.Tensor) -> int:
    # sizeof(CoopWorkspace) as the compiled module reports it, so reordering or
    # extending the struct goes red here instead of silently reading another field;
    # a size-preserving reorder is caught instead by callers comparing
    # {before, after} == {0, 1} rather than !=, which no wrong offset satisfies.
    assert ws.numel() * 4 == _COOP_WS_BYTES, (
        f"workspace is {ws.numel() * 4} bytes, expected {_COOP_WS_BYTES}: the "
        f"CoopWorkspace layout changed, so parity is not at word "
        f"{_COOP_WS_PARITY_WORD}"
    )
    return int(ws[_COOP_WS_PARITY_WORD].item())


@contextlib.contextmanager
def _coop_enabled(floor: int = COOP_FLOOR):
    if is_hip_runtime():
        pytest.skip("the cooperative handoff is CUDA-only")
    with (
        envs.SGLANG_OPT_USE_COOP_TOPK.override(True),
        envs.SGLANG_OPT_COOP_TOPK_FLOOR.override(floor),
    ):
        yield


@pytest.mark.parametrize("floor", [0, COOP_FLOOR - 1, 1 << 32])
def test_topk_v2_coop_floor_out_of_range_is_rejected(floor: int) -> None:
    """Floors the two kernels cannot partition the rows with are refused up front.

    Under kClusterFloorSmall a row just past the floor takes the official
    kernel's non-cluster path, which has no handoff guard, so both kernels write
    it; 0 also reaches the kernel as the wire value for "off". Past 0xffffffff
    the kernel's uint32_t truncates the value, and 1 << 32 truncates to exactly
    that same "off", so the two ends of the range fail the same way.
    """
    with _coop_enabled(floor), pytest.raises(ValueError, match="at least"):
        _coop_topk_floor()


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize(
    "seq",
    [
        COOP_FLOOR + 1,  # one past the floor: the cooperative kernel owns it
        131072,  # whole float4 runs, no tail
        131075,  # partial trailing float4, so the scalar tail is read
    ],
)
@torch.inference_mode()
def test_topk_v2_coop(seq: int, k: int) -> None:
    torch.manual_seed(seq * 31 + k)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(1, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((1,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(1, num_pages, "perm", device)

    with _coop_enabled():
        our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, 1, seq_lens.cpu(), k)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("seq", [1024, 20000, COOP_FLOOR])
@torch.inference_mode()
def test_topk_v2_coop_short_row_in_wide_buffer(seq: int, k: int) -> None:
    """A row at or below the floor in a buffer wide enough to arm the handoff.

    The host arms the cooperative launch without reading the length, so both
    kernels run and only the device-side length decides which one writes.
    ``seq == COOP_FLOOR`` is the boundary: the official kernel keeps it.
    """
    torch.manual_seed(seq * 17 + k)
    device = "cuda"
    width = 131072
    scores = torch.randn(1, width, dtype=torch.float32, device=device)
    seq_lens = torch.full((1,), seq, dtype=torch.int32, device=device)
    page_table, inv_cpu = _make_page_table(1, width // PAGE_SIZE, "perm", device)

    with _coop_enabled():
        ws = _coop_topk_workspace(torch.cuda.current_device())
        before = _coop_parity(ws)
        our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
        after = _coop_parity(ws)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, 1, seq_lens.cpu(), k)
    # Every seq is at or below the floor while the buffer is wide enough to arm, so
    # the coop kernel launches and must return at its below-floor guard without
    # reaching the barrier that writes parity. The comparison above passes whichever
    # kernel wrote the row, so nothing else here can see an inverted guard.
    assert after == before and before in (0, 1), (
        f"parity moved from {before} to {after} on a row at or below the floor: the "
        f"cooperative kernel did work it must leave to the official kernel, so the "
        f"two guards are not complementary"
    )


@pytest.mark.parametrize("k", [512, 2048])
@torch.inference_mode()
def test_topk_v2_coop_output_indices(k: int) -> None:
    """No page table: the cooperative kernel must not run the page transform.

    The transform is compiled out per mode rather than skipped at runtime, so a
    mode that keeps it would gather through a null page table.
    """
    torch.manual_seed(9001 + k)
    device = "cuda"
    seq = 131072
    scores = torch.randn(1, seq, dtype=torch.float32, device=device)
    seq_lens = torch.full((1,), seq, dtype=torch.int32, device=device)

    with _coop_enabled():
        our_raw = _run_raw(scores, seq_lens, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, 1, seq_lens.cpu(), k)


@torch.inference_mode()
def test_topk_v2_coop_reuses_workspace() -> None:
    """Consecutive launches share one workspace that no caller ever clears.

    Each launch leaves the histograms zeroed and zeroes the counter slot the next
    launch will read, alternating between two slots. Launches 1 and 2 still read
    slots the allocation zeroed, so launch 3 is the first to read one a launch
    cleared and launch 4 is the first on the other slot: fewer than four trials
    cannot tell the self-clean apart from the memset.
    """
    device = "cuda"
    seq, k = 65537, 2048
    width = (seq + 3) & ~3
    seq_lens = torch.full((1,), seq, dtype=torch.int32, device=device)
    page_table, inv_cpu = _make_page_table(1, seq // PAGE_SIZE + 1, "perm", device)

    with _coop_enabled():
        ws = _coop_topk_workspace(torch.cuda.current_device())
        for trial in range(4):
            torch.manual_seed(1234 + trial)
            scores = torch.randn(1, width, dtype=torch.float32, device=device)[:, :seq]
            before = _coop_parity(ws)
            our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
            ref_raw = _reference(scores, seq_lens, k)
            _assert_topk_close(scores.cpu(), ref_raw, our_raw, 1, seq_lens.cpu(), k)
            # The comparison above is satisfied by either kernel; this says which one
            # produced it, and so is what puts the alternating-slot protocol on test.
            assert {before, _coop_parity(ws)} == {0, 1}, (
                f"trial {trial}: parity did not toggle, so the cooperative kernel "
                f"never reached its final barrier -- the self-clean handoff this test "
                f"exists to check was not exercised, and the results cannot show that"
            )


@pytest.mark.parametrize("head", [0, 300])
@torch.inference_mode()
def test_topk_v2_coop_quantized_scores_force_refine(head: int) -> None:
    """Scores coarse enough that the k-th value's level-0 bin overflows kMaxNumTie.

    Over random floats the threshold bin sits in the sparse top tail and holds
    far fewer than kMaxNumTie elements, so level 0 always breaks immediately and
    the two refine levels, the extra grid barriers they take, and the
    band_shift == 0 exit never run at all. Eight distinct values put ~16k
    elements in one bin instead -- the shape a saturated or quantized indexer
    score row has. ``head`` seeds a sparse band above the threshold so the
    refine is also covered with a nonzero above-count.
    """
    torch.manual_seed(4242 + head)
    device = "cuda"
    seq, k = 131072, 2048
    scores = torch.randint(0, 8, (1, seq), device=device).to(torch.float32)
    if head:
        scores[0, torch.randperm(seq, device=device)[:head]] = 9.0
    seq_lens = torch.full((1,), seq, dtype=torch.int32, device=device)
    page_table, inv_cpu = _make_page_table(1, seq // PAGE_SIZE, "perm", device)

    with _coop_enabled():
        our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, 1, seq_lens.cpu(), k)


@pytest.mark.parametrize(
    "seq,nan_count",
    [
        (131072, 1),  # whole float4 runs
        (131075, 3),  # odd length, so the last NaN is read by the scalar tail
        (131072, 4096),  # more NaN than kMaxNumTie, so the tie buffer overflows too
    ],
)
@torch.inference_mode()
def test_topk_v2_coop_drops_nan_keeps_inf(seq: int, nan_count: int) -> None:
    """NaN must not be selected, and +inf must still be.

    The radix key is the order-preserving integer image of the float bits, under
    which a positive NaN outranks +inf, so a NaN reaching the histogram is picked
    as the largest element -- while the official kernel below the floor drops it,
    which would make the two sides disagree on one row. NaN is also unordered
    against itself, so the tie comparator answers false both ways: ranks collide,
    output slots stay unwritten, and the page transform reads them anyway. The
    +inf positions pin the other direction, that only NaN is dropped rather than
    every non-finite value.
    """
    torch.manual_seed(31337 + seq + nan_count)
    device = "cuda"
    k = 2048
    width = (seq + 3) & ~3
    scores = torch.randn(1, width, dtype=torch.float32, device=device)[:, :seq]
    perm = torch.randperm(seq - 1, device=device).tolist()
    nan_pos = perm[: nan_count - 1] + [seq - 1]
    inf_pos = perm[nan_count - 1 : nan_count + 7]
    scores[0, nan_pos] = float("nan")
    scores[0, inf_pos] = float("inf")
    seq_lens = torch.full((1,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(1, num_pages, "perm", device)

    with _coop_enabled():
        our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)

    selected = set(our_raw[0])
    assert selected.isdisjoint(nan_pos), (
        f"NaN selected: {sorted(selected & set(nan_pos))[:4]}"
    )
    assert set(inf_pos) <= selected, (
        f"+inf dropped: {sorted(set(inf_pos) - selected)[:4]}"
    )
    # Mapping NaN to -inf is what the kernel does, so torch.topk over the mapped
    # row is the reference; it also keeps the tie compare free of NaN.
    clean = torch.where(scores.isnan(), torch.full_like(scores, -float("inf")), scores)
    ref_raw = _reference(clean, seq_lens, k)
    _assert_topk_close(clean.cpu(), ref_raw, our_raw, 1, seq_lens.cpu(), k)


@torch.inference_mode()
def test_topk_v2_coop_graph_capture_and_replay() -> None:
    """The cooperative launch must survive capture, and replays must be correct.

    cudaLaunchAttributeCooperative is set per launch and the grid is only legal
    while it holds, so a capture that dropped it would turn the kernel's grid
    barrier into a hang with no diagnostic. Decode runs this kernel inside a
    captured graph; every other case here launches eagerly.
    """
    device = "cuda"
    seq, k = 65537, 2048
    width = (seq + 3) & ~3
    seq_lens = torch.full((1,), seq, dtype=torch.int32, device=device)
    page_table, inv_cpu = _make_page_table(1, seq // PAGE_SIZE + 1, "perm", device)
    # The graph replays over this exact buffer, so the row is refilled in place.
    scores = torch.randn(1, width, dtype=torch.float32, device=device)[:, :seq]
    out = torch.full((1, k), -1, dtype=torch.int32, device=device)

    with _coop_enabled():
        # Eager first: the workspace allocation refuses to run under capture, and
        # the plan synchronizes, so both have to be done before the graph opens.
        _run(scores, seq_lens, page_table, inv_cpu, k)
        # Already allocated by the eager run above, so this is a cache hit and cannot
        # trip the under-capture raise. Read outside the graph, per replay.
        ws = _coop_topk_workspace(torch.cuda.current_device())
        metadata = _plan(seq_lens)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            topk_transform_paged_v2(
                scores, seq_lens, page_table, out, PAGE_SIZE, metadata
            )
        for trial in range(3):
            torch.manual_seed(555 + trial)
            scores.copy_(torch.randn(1, seq, dtype=torch.float32, device=device))
            out.fill_(-1)  # so a partial write shows up as missing slots
            before = _coop_parity(ws)
            graph.replay()
            torch.cuda.synchronize()
            # A graph that dropped the cooperative attribute, or never contained the
            # launch, still replays and still returns the right answer; without this
            # the assertions below cannot tell that apart from a captured launch.
            assert {before, _coop_parity(ws)} == {0, 1}, (
                f"replay {trial}: parity did not flip, so the captured graph contains "
                f"no completed cooperative launch -- which is exactly what this test "
                f"claims to rule out"
            )
            our_raw = [_invert(out.cpu().tolist()[0], inv_cpu[0])]
            ref_raw = _reference(scores, seq_lens, k)
            _assert_topk_close(scores.cpu(), ref_raw, our_raw, 1, seq_lens.cpu(), k)


# --- ragged entry point ------------------------------------------------------
# Rows select inside `[row_start, row_start + seq_len)` of their score row and
# emit `position + offset`. The window start is an arbitrary token offset, so
# every `row_start % 4` residue must be covered: the kernel reads from a
# 16-byte-aligned base and masks the <=3 columns that pulls in ahead of the
# window. Everything outside the window is filled with OUTSIDE_SCORE, which
# beats every in-window score, so any leak shows up as a wrong selection.
OUTSIDE_SCORE = 1e3

# (name, per-row (row_start, length)) spanning every template and residue.
RAGGED_CONFIGS = [
    # one length per template band, all four residues plus aligned starts
    ("trivial", [(s, 1500) for s in (0, 1, 2, 3, 4, 7, 4096, 4099)]),
    ("register2", [(s, 6000) for s in (0, 1, 2, 3, 4, 7, 4096, 4099)]),
    ("register4", [(s, 12000) for s in (0, 1, 2, 3, 4, 7, 4096, 4099)]),
    ("streaming", [(s, 40000) for s in (0, 1, 2, 3, 4, 7, 4096, 4099)]),
    # mixed bands in one launch, laid out back to back like a real prefill batch
    (
        "mixed",
        [
            (0, 1000),
            (1000, 3000),
            (4000, 9000),
            (13000, 20000),
            (33000, 1),
            (33001, 2047),
        ],
    ),
    # boundaries: seq == k, seq == k + 1, and the register/streaming edges
    (
        "boundaries",
        [(1, 2048), (2049, 2049), (4098, 8192), (12290, 8193), (20483, 16385)],
    ),
    ("long_ctx", [(0, 131072), (131072, 65537), (196609, 100000)]),
]


def _make_ragged(rows, offset_shift, device):
    width = ((max(s + n for s, n in rows)) + 3) & ~3
    scores = torch.full(
        (len(rows), width), OUTSIDE_SCORE, dtype=torch.float32, device=device
    )
    for i, (start, length) in enumerate(rows):
        scores[i, start : start + length] = torch.randn(length, device=device)
    starts = torch.tensor([s for s, _ in rows], dtype=torch.int32, device=device)
    lengths = torch.tensor([n for _, n in rows], dtype=torch.int32, device=device)
    return scores, starts, lengths, starts + offset_shift


def _run_ragged(scores, lengths, starts, offsets, k):
    """Selected positions per row, rebased back to window-relative."""
    out = torch.empty((scores.shape[0], k), dtype=torch.int32, device=scores.device)
    topk_transform_ragged_v2(
        scores, lengths, out_offsets=offsets, out_indices=out, row_starts=starts
    )
    torch.cuda.synchronize()
    off = offsets.cpu().tolist()
    return [
        [v - off[i] for v in row if v != -1] for i, row in enumerate(out.cpu().tolist())
    ]


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("offset_shift", [0, 4321])
@pytest.mark.parametrize("name,rows", RAGGED_CONFIGS)
@torch.inference_mode()
def test_topk_v2_ragged_window(name: str, rows, k: int, offset_shift: int) -> None:
    torch.manual_seed(len(rows) * 7919 + k + offset_shift)
    device = "cuda"
    scores, starts, lengths, offsets = _make_ragged(rows, offset_shift, device)
    before = scores.clone()

    our_raw = _run_ragged(scores, lengths, starts, offsets, k)

    # reference on the window slice, padded to a common width for the helper
    max_len = max(n for _, n in rows)
    windows = torch.zeros(len(rows), max_len, dtype=torch.float32)
    for i, (start, length) in enumerate(rows):
        windows[i, :length] = before[i, start : start + length].cpu()
    ref_raw = _reference(windows, lengths.cpu(), k)
    _assert_topk_close(windows, ref_raw, our_raw, len(rows), lengths.cpu(), k)

    # the only legal in-place write is the <=3 masked columns ahead of a window
    # that the kernel actually reads (trivial rows read nothing)
    changed = (scores != before).cpu()
    for i, (start, length) in enumerate(rows):
        allowed = torch.zeros(scores.shape[1], dtype=torch.bool)
        if length > k:
            allowed[start - start % 4 : start] = True
        stray = (changed[i] & ~allowed).nonzero().flatten().tolist()
        assert not stray, f"row {i} ({name}) wrote outside its masked head: {stray[:8]}"


@pytest.mark.parametrize("k", [512, 2048])
@torch.inference_mode()
def test_topk_v2_ragged_no_row_starts(k: int) -> None:
    """`row_starts=None` means every window starts at column 0."""
    torch.manual_seed(4242 + k)
    device = "cuda"
    rows = [(0, 900), (0, 5000), (0, 20000), (0, 70000)]
    scores, starts, lengths, offsets = _make_ragged(rows, 0, device)
    explicit = _run_ragged(scores.clone(), lengths, starts, offsets, k)
    implicit = _run_ragged(scores.clone(), lengths, None, offsets, k)
    for i in range(len(rows)):
        assert sorted(explicit[i]) == sorted(implicit[i]), f"row {i} differs"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
