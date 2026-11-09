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
  Streaming     seq > 16384            max_seq > 16384, below the cluster floor (level 2)
  Cluster       seq above the floor    the arch has clusters and batch <= 512

and two cluster dispatch shapes: the fused small-batch kernel (batch up to the
probed persistent-pool size) and the persistent pool + main kernel above it. The
cluster floor and pool size are per-arch (see topk_v2.cuh), so the (batch, seq)
grid below brackets the fixed boundaries (8192/8193, 16384/16385) exactly and
spans the arch-dependent ones, across k in {512,1024,2048} and identity/perm
page tables.

``test_topk_v2_packed_rows`` covers the DSA extend layout on top of that: all
requests packed into one score buffer, sharing a table row per request.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_candidates_v2,
    topk_transform_packed_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
MAX_PERMIT_ERROR = 5
# The default top-k mode is exact, so CUDA pins the packed kernel to exact
# (ROCm keeps the default tolerance).
PACKED_PERMIT_ERROR = MAX_PERMIT_ERROR if is_hip() else 0

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
    (8, 16385),  # just over reg4
    (4, 32768),
    (16, 65535),
    (4, 65536),
    (100, 65536),
    # --- long rows, small batch: fused cluster kernel where the arch has clusters ---
    (1, 65537),
    (2, 131072),
    (8, 98304),
    (30, 131072),
    # --- long rows, mid batch: persistent cluster pool + main kernel ---
    (31, 131072),
    (40, 262144),  # more items than the pool => round-robin
    (64, 196608),
    (128, 131072),
    # --- long rows, large batch ---
    (129, 131072),
    (200, 262144),
]


def _assert_topk_close(
    scores_cpu, ref_raw, our_raw, bs, seq_lens, k, max_permit_error=MAX_PERMIT_ERROR
):
    """Set-compare our top-k raw indices vs torch's, tolerating equal-score ties.

    ``max_permit_error`` covers the fp16 coarse histogram swapping near-equal
    scores; tests that pin exactness pass 0.
    """
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
    assert bad <= max_permit_error, f"{bad=} > {max_permit_error}"


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


def _run_dual(scores, seq_lens, page_table, inv_cpu, k):
    batch = scores.shape[0]
    metadata = _plan(seq_lens)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    raw = torch.full_like(out, -1)
    topk_transform_paged_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()
    out_cpu = out.cpu().tolist()
    raw_cpu = raw.cpu().tolist()
    transformed_raw = [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]
    direct_raw = [[v for v in raw_cpu[i] if v != -1] for i in range(batch)]
    return transformed_raw, direct_raw


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
    # span every path, including rows long enough for the cluster dispatch
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


@pytest.mark.parametrize(
    "batch,seq", [(8, 256), (8, 8192), (4, 32768), (2, 131072), (31, 131072)]
)
@torch.inference_mode()
def test_topk_v2_dual_output(batch: int, seq: int) -> None:
    """The dual mode returns the same selection before and after page transform."""
    k = 512
    torch.manual_seed(batch * 100003 + seq * 7 + k + 2)
    device = "cuda"
    scores = torch.randn(batch, seq, dtype=torch.float32, device=device)
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, "perm", device)

    transformed_raw, direct_raw = _run_dual(scores, seq_lens, page_table, inv_cpu, k)
    for row in range(batch):
        assert sorted(transformed_raw[row]) == sorted(direct_raw[row])
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, direct_raw, batch, seq_lens.cpu(), k)


@pytest.mark.parametrize("batch", [1, 16])
@torch.inference_mode()
def test_topk_v2_cluster_refine_page_outputs(batch: int) -> None:
    """A cluster row whose threshold bin overflows twice (all zeros but a few ones) keeps a
    duplicate-free page-table and dual selection that holds every one."""
    k, seq, ones = 2048, 65536, 1024
    device = "cuda"
    scores = torch.zeros(batch, seq, dtype=torch.float32, device=device)
    scores[:, seq - ones :] = 1.0
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    page_table, inv_cpu = _make_page_table(batch, seq // PAGE_SIZE, "perm", device)

    paged_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
    transformed_raw, direct_raw = _run_dual(scores, seq_lens, page_table, inv_cpu, k)
    for row in range(batch):
        for selected in (paged_raw[row], transformed_raw[row], direct_raw[row]):
            assert len(selected) == k and len(set(selected)) == k
            assert set(range(seq - ones, seq)) <= set(selected)


# --- exactness when the coarse histogram cannot discriminate -----------------
# The tests above draw from ``torch.randn``, which spreads scores over many
# coarse bins and leaves the threshold bin far below the tie-staging capacity.
# These distributions force the threshold bin to overflow the staging buffer.
NARROW_DISTRIBUTIONS = [
    "narrow",
    "narrow_bin",
    "tiny",
    "two_values",
    "signed_zero",
    "all_equal",
    "all_zero",
]


def _single_bin_scores(kind, batch, width, device):
    """Scores whose coarse bin carries no ordering information.

    The bin comes from the top bits of the fp16 cast, so a band narrower than
    one fp16 ulp -- or a magnitude that underflows fp16 -- gives one bin.
    """
    if kind == "narrow":
        # Narrower than one fp32 ulp at 1.0 (1.19e-7): pins the tie count.
        u = torch.rand(batch, width, dtype=torch.float32, device=device) * 2.0 - 1.0
        return 1.0 + u * 1e-6
    if kind == "narrow_bin":
        # One coarse bin, ~1.7e5 distinct fp32 values: pins the ordering too.
        return 1.0 + torch.rand(batch, width, dtype=torch.float32, device=device) * 0.02
    if kind == "tiny":
        # Below the fp16 subnormal floor: every score casts to zero.
        return (
            0.5 + torch.rand(batch, width, dtype=torch.float32, device=device)
        ) * 1e-30
    if kind == "two_values":
        # Two adjacent fp32 values, ~half the row each: the refinement has to
        # consume all 32 bits before the survivors are bit-identical, and more
        # than kMaxNumTie of them still remain at that point.
        hi = torch.tensor(1.0, dtype=torch.float32, device=device).nextafter(
            torch.tensor(2.0, dtype=torch.float32, device=device)
        )
        pick = torch.rand(batch, width, device=device) < 0.5
        return torch.where(pick, hi, torch.ones_like(hi))
    if kind == "signed_zero":
        # The threshold bin ends at +0.0. Its upper boundary must move to the
        # smallest positive subnormal so -0.0 stays in the tie set instead of
        # racing true positive values for output slots.
        out = torch.full((batch, width), -1.0, dtype=torch.float32, device=device)
        out[:, : width // 2] = -0.0
        out[:, -1024:] = 1e-6
        return out
    if kind == "all_zero":
        # Only +0.0: any subset is exact, but the deterministic mode must still
        # take indices 0..k-1 (the zero-bin shortcut may not apply there).
        return torch.zeros((batch, width), dtype=torch.float32, device=device)
    # Control: the bin overflows too, but the candidates are bit-identical, so
    # any subset is correct and this must keep passing.
    return torch.full((batch, width), 0.5, dtype=torch.float32, device=device)


def _narrow_seed(dist, *rest):
    """Reproducible across processes -- ``hash(str)`` is salted per interpreter."""
    seed = NARROW_DISTRIBUTIONS.index(dist)
    for v in rest:
        seed = seed * 100003 + int(v)
    return seed


# Longest row that never reaches the CUDA cluster path (the smallest
# get_static_cluster_floor in topk_v2.cuh).
CUDA_MAX_NON_CLUSTER_SEQ = 24576


# SGLANG_TOPK_V2_MODE values
MODE_FAST, MODE_EXACT, MODE_DETERMINISTIC = 0, 1, 2


def _topk_mode() -> int:
    """The exactness mode this process's top-k v2 module is built with."""
    return envs.SGLANG_TOPK_V2_MODE.get()


def _assert_topk_legal(our_raw, batch, seq_lens, k):
    """FAST mode: an overflowing bin keeps an arbitrary subset, so only legality holds."""
    for i in range(batch):
        L = int(seq_lens[i])
        our = our_raw[i]
        assert len(our) == min(k, L) and len(set(our)) == len(our), f"row {i}"
        assert all(0 <= v < L for v in our), f"row {i}: index out of range"


def _assert_smallest_index_ties(scores_cpu, our_raw, batch, seq_lens, k):
    """DETERMINISTIC mode: elements equal to the k-th value are taken smallest index first."""
    for i in range(batch):
        L = int(seq_lens[i])
        row = scores_cpu[i, :L].tolist()
        expected = sorted(range(L), key=lambda j: (-row[j], j))[:k]
        assert sorted(our_raw[i]) == sorted(expected), (
            f"row {i}: tie choice is not by smallest index"
        )


@pytest.mark.parametrize("dist", NARROW_DISTRIBUTIONS)
@pytest.mark.parametrize("batch,seq", [(6, 8192), (4, 20480), (4, 32768), (4, 65536)])
@torch.inference_mode()
def test_topk_v2_single_coarse_bin_is_exact(batch: int, seq: int, dist: str) -> None:
    """No tolerance here: the threshold bin holds the whole row at k=2048, so a
    dropped candidate is a wrong answer rather than a tie swap. Covers
    register-path and streaming-path shapes, and a ROCm split-path shape (above
    split_floor), and on CUDA the small-batch cluster kernel above the cluster
    floor; tie staging is per-template.
    """
    torch.manual_seed(_narrow_seed(dist, batch, seq))
    device = "cuda"
    k = 2048
    width = (seq + 3) & ~3
    scores = _single_bin_scores(dist, batch, width, device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)

    our_raw = _run_raw(scores, seq_lens, k)
    if _topk_mode() == MODE_FAST:
        _assert_topk_legal(our_raw, batch, seq_lens.cpu(), k)
        return
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(
        scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k, max_permit_error=0
    )


def _mode_check() -> None:
    """Checks of this process's mode, run by test_topk_v2_modes in a subprocess per
    non-default mode (the mode is compiled in once per process). Shapes cover the
    one-block paths and, on CUDA, the C16 / C8 small-batch cluster kernels and the
    persistent cluster pool (34 rows above the plan's smallest threshold); equal-value groups
    larger than the tie buffer included. FAST only promises a legal output."""
    mode = _topk_mode()
    k = 2048
    for dist in NARROW_DISTRIBUTIONS:
        for batch, seq in [
            (6, 8192),
            (4, 16384),
            (4, CUDA_MAX_NON_CLUSTER_SEQ),
            (4, 65536),
            (12, 40960),
            (34, 131076),
            (1, 131072),  # ROCm: the split path
        ]:
            torch.manual_seed(_narrow_seed(dist, batch, seq))
            width = (seq + 3) & ~3
            scores = _single_bin_scores(dist, batch, width, "cuda")[:, :seq]
            seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
            scores_cpu, lens_cpu = scores.cpu(), seq_lens.cpu()
            if mode == MODE_FAST:
                _assert_topk_legal(_run_raw(scores, seq_lens, k), batch, lens_cpu, k)
                continue
            runs = [_run_raw(scores, seq_lens, k) for _ in range(3)]
            ref_raw = _reference(scores, seq_lens, k)
            _assert_topk_close(
                scores_cpu, ref_raw, runs[0], batch, lens_cpu, k, max_permit_error=0
            )
            if mode == MODE_DETERMINISTIC:
                for other in runs[1:]:
                    assert [sorted(r) for r in other] == [sorted(r) for r in runs[0]], (
                        dist
                    )
                _assert_smallest_index_ties(scores_cpu, runs[0], batch, lens_cpu, k)
        print(f"mode {mode} {dist}: ok", flush=True)


_EXACT_MODE_CHECKS: dict[int, tuple[subprocess.Popen, object]] = {}


@pytest.fixture(scope="module", autouse=True)
def _start_exact_mode_checks():
    """Start the per-mode subprocesses at module start, so their JIT builds overlap
    this process's build and tests instead of running after them."""
    if not torch.cuda.is_available():
        yield
        return
    # This process runs the default EXACT mode; the other two each need their own process.
    for mode in (MODE_FAST, MODE_DETERMINISTIC):
        log = tempfile.TemporaryFile(mode="w+")
        with envs.SGLANG_TOPK_V2_MODE.override(mode):
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--mode-check"],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        _EXACT_MODE_CHECKS[mode] = (proc, log)
    yield
    for proc, log in _EXACT_MODE_CHECKS.values():
        if proc.poll() is None:
            proc.kill()
        log.close()


@pytest.mark.parametrize("mode", [MODE_FAST, MODE_DETERMINISTIC])
def test_topk_v2_modes(mode: int) -> None:
    """FAST / DETERMINISTIC in their own process, since one process builds one mode."""
    proc, log = _EXACT_MODE_CHECKS[mode]
    proc.wait(timeout=1800)
    log.seek(0)
    assert proc.returncode == 0, log.read()[-8000:]


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

    # The kernel may mask the at-most-three alignment columns immediately
    # before a window. Everything else, including other rows' storage, must be
    # left untouched.
    changed = (scores != before).cpu()
    for i, (start, length) in enumerate(rows):
        allowed = torch.zeros(scores.shape[1], dtype=torch.bool)
        if length > k:
            allowed[start - start % 4 : start] = True
        stray = (changed[i] & ~allowed).nonzero().flatten().tolist()
        assert not stray, f"row {i} ({name}) wrote outside its masked head: {stray[:8]}"


def _assert_topk_values(window, indices, k):
    indices = indices.cpu().long()
    window = window.cpu()
    assert indices.numel() == k
    assert ((indices >= 0) & (indices < window.numel())).all(), indices
    assert indices.unique().numel() == k
    expected = window.topk(k).values.sort().values
    actual = window[indices].sort().values
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_ties", [48, 96])
@torch.inference_mode()
def test_topk_v2_negative_infinity_ties(num_ties: int) -> None:
    """Inactive entries must not displace valid -inf scores or leave slots unwritten."""
    k = 16
    length = num_ties + 3
    scores = torch.full((1, (length + 3) & ~3), -torch.inf, device="cuda")
    scores[0, :3] = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    out = torch.full((1, k), -2, dtype=torch.int32, device="cuda")

    topk_transform_paged_v2(scores, lengths, None, out, PAGE_SIZE, _plan(lengths))

    _assert_topk_values(scores[0, :length], out[0], k)


@pytest.mark.parametrize("length", [257, 8193, 16385])
@torch.inference_mode()
def test_topk_v2_ragged_negative_infinity(length: int) -> None:
    """Columns before an unaligned window must never beat its valid -inf scores."""
    k = 16
    scores = torch.full((3, (length + 6) & ~3), OUTSIDE_SCORE, device="cuda")
    starts = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    lengths = torch.full((3,), length, dtype=torch.int32, device="cuda")
    offsets = starts + 1024
    out = torch.full((3, k), -2, dtype=torch.int32, device="cuda")
    for row, start in enumerate((1, 2, 3)):
        scores[row, start : start + length] = -torch.inf
        scores[row, start : start + 3] = torch.tensor([1.0, 2.0, 3.0], device="cuda")

    topk_transform_ragged_v2(
        scores, lengths, out_offsets=offsets, out_indices=out, row_starts=starts
    )

    for row, start in enumerate((1, 2, 3)):
        _assert_topk_values(
            scores[row, start : start + length], out[row] - offsets[row], k
        )


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


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize(
    "extend_lens",
    [
        [7],  # one request
        [4, 4],  # equal row counts
        [1, 13, 2],  # ragged, including a single-row request
    ],
)
@torch.inference_mode()
def test_topk_v2_packed_rows(extend_lens: list[int], k: int) -> None:
    """DSA extend layout: batch-global packed scores + shared page-table rows.

    Rows are causal within a request; a distinct page-table permutation per request
    catches row/request index mix-ups, and the ragged case leaves most window starts
    off the 16-byte load boundary (the production case).
    """
    torch.manual_seed(4242 + k + len(extend_lens))
    device = "cuda"

    # Keep every row longer than k so no row takes the trivial path.
    prefix = k + 1024
    kv_lens = [prefix + e for e in extend_lens]
    k_offsets = [0]
    for kv in kv_lens[:-1]:
        k_offsets.append(k_offsets[-1] + kv)
    total_kv = sum(kv_lens)

    row_starts, lengths, row_to_batch = [], [], []
    for i, e in enumerate(extend_lens):
        for local in range(e):
            row_starts.append(k_offsets[i])
            lengths.append(kv_lens[i] - e + local + 1)
            row_to_batch.append(i)
    rows = len(lengths)

    width = (total_kv + 3) & ~3
    scores = torch.randn(rows, width, dtype=torch.float32, device=device)[:, :total_kv]
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    row_starts_t = torch.tensor(row_starts, dtype=torch.int32, device=device)
    row_to_batch_t = torch.tensor(row_to_batch, dtype=torch.int32, device=device)

    num_pages = (max(kv_lens) + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(
        len(extend_lens), num_pages, "perm", device, per_row=True
    )

    out = torch.full((rows, k), -1, dtype=torch.int32, device=device)
    # The kernel masks in place, so reference values must be read before the call.
    scores_cpu = scores.cpu()
    topk_transform_packed_v2(
        scores,
        lengths_t,
        page_table,
        out,
        PAGE_SIZE,
        row_starts=row_starts_t,
        row_to_batch=row_to_batch_t,
    )
    torch.cuda.synchronize()

    out_cpu = out.cpu().tolist()
    for r in range(rows):
        L, start, req = lengths[r], row_starts[r], row_to_batch[r]
        window = scores_cpu[r, start : start + L]
        ref = torch.topk(window, k, sorted=False).indices.tolist()
        our = _invert(out_cpu[r], inv_cpu[req])
        _assert_topk_close(
            window.unsqueeze(0),
            [ref],
            [our],
            1,
            [L],
            k,
            max_permit_error=PACKED_PERMIT_ERROR,
        )


@pytest.mark.parametrize("residue", [1, 2, 3])
@pytest.mark.parametrize("boundary", [8192, 16384])
@torch.inference_mode()
def test_topk_v2_packed_level_boundary(boundary: int, residue: int) -> None:
    """Rows whose length sits on an implementation's max_seq_len boundary.

    The masked head widens the problem by ``residue``, so a row of exactly
    ``boundary`` tokens spills past the register implementation sized for it. The
    packed kernel picks the implementation per row from the widened length, so
    these must still be exact; a compile-time choice made from the un-widened
    length would overflow.
    """
    torch.manual_seed(boundary + residue)
    device = "cuda"
    k = 512

    # Request 0 exists only to push request 1's window off the 16-byte boundary.
    kv_lens = [residue, boundary + 1]
    lengths = [residue] + [boundary - 1, boundary, boundary + 1]
    row_starts = [0] + [residue] * 3
    row_to_batch = [0, 1, 1, 1]
    rows = len(lengths)
    total_kv = sum(kv_lens)

    width = (total_kv + 3) & ~3
    scores = torch.randn(rows, width, dtype=torch.float32, device=device)[:, :total_kv]
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    row_starts_t = torch.tensor(row_starts, dtype=torch.int32, device=device)
    row_to_batch_t = torch.tensor(row_to_batch, dtype=torch.int32, device=device)

    num_pages = (max(kv_lens) + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(
        len(kv_lens), num_pages, "perm", device, per_row=True
    )

    out = torch.full((rows, k), -1, dtype=torch.int32, device=device)
    scores_cpu = scores.cpu()
    topk_transform_packed_v2(
        scores,
        lengths_t,
        page_table,
        out,
        PAGE_SIZE,
        row_starts=row_starts_t,
        row_to_batch=row_to_batch_t,
    )
    torch.cuda.synchronize()

    out_cpu = out.cpu().tolist()
    for r in range(rows):
        L, start, req = lengths[r], row_starts[r], row_to_batch[r]
        window = scores_cpu[r, start : start + L]
        our = _invert(out_cpu[r], inv_cpu[req])
        if L <= k:
            # Trivial path: every position, then -1 padding.
            assert sorted(our[:L]) == list(range(L)), f"row {r} trivial output wrong"
            assert all(v == -1 for v in out_cpu[r][L:]), f"row {r} padding wrong"
            continue
        ref = torch.topk(window, k, sorted=False).indices.tolist()
        _assert_topk_close(
            window.unsqueeze(0),
            [ref],
            [our],
            1,
            [L],
            k,
            max_permit_error=PACKED_PERMIT_ERROR,
        )


@pytest.mark.skipif(is_hip(), reason="ROCm split path keeps its own tie merge")
@pytest.mark.parametrize("num_ties", [129, 700, 1500])
@pytest.mark.parametrize("seq", [5000, 12000, 40000, 131072])
@pytest.mark.parametrize("path", ["paged", "ragged", "packed"])
@torch.inference_mode()
def test_topk_v2_ties_by_index(path: str, seq: int, num_ties: int) -> None:
    """With deterministic, equal scores at the cut resolve to the smallest
    indices, as a stable sort would, also past the 128 ties the warp paths
    handle. A run of
    ``num_ties`` equal scores straddles the k-th place of each row; it stays
    below the 2048-candidate tie buffer, past which the choice among the
    candidates of the threshold bin is not defined."""
    k, batch = 2048, 2
    g = torch.Generator(device="cuda").manual_seed(seq + num_ties)
    scores = torch.randn(batch, seq, generator=g, device="cuda")
    for row in range(batch):
        cut = scores[row].topk(k - num_ties // 2).values[-1].item()
        tied = torch.randperm(seq, generator=g, device="cuda")[:num_ties]
        scores[row, tied] = cut
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    out = torch.full((batch, k), -2, dtype=torch.int32, device="cuda")
    zeros = torch.zeros(batch, dtype=torch.int32, device="cuda")
    if path == "paged":
        plan = _plan(seq_lens)
        topk_transform_paged_v2(
            scores.clone(), seq_lens, None, out, PAGE_SIZE, plan, deterministic=True
        )
    elif path == "ragged":
        topk_transform_ragged_v2(
            scores.clone(),
            seq_lens,
            out_offsets=zeros,
            out_indices=out,
            deterministic=True,
        )
    else:
        identity = torch.arange(-(-seq // PAGE_SIZE), dtype=torch.int32, device="cuda")
        topk_transform_packed_v2(
            scores.clone(),
            seq_lens,
            identity.expand(batch, -1).contiguous(),
            out,
            PAGE_SIZE,
            row_starts=zeros,
            deterministic=True,
        )
    for row in range(batch):
        expected = torch.sort(-scores[row], stable=True).indices[:k].sort().values
        actual = out[row].long().sort().values
        assert torch.equal(actual, expected), f"row {row}: tie order not stable"


@pytest.mark.skipif(is_hip(), reason="CUDA candidate epilogue")
@pytest.mark.parametrize("seq", [100, 2048, 5000, 12000, 40000])
@pytest.mark.parametrize("with_row_starts", [False, True])
@torch.inference_mode()
def test_topk_v2_candidates_match_ragged(seq: int, with_row_starts: bool) -> None:
    """The candidate epilogue selects what the ragged kernel selects and emits
    (score bits, position * dcp_size + dcp_rank), (0, -1) padded."""
    k, batch, dcp_size, dcp_rank = 2048, 3, 8, 5
    g = torch.Generator(device="cuda").manual_seed(seq)
    width = seq + 16
    scores = torch.randn(batch, width, generator=g, device="cuda")
    scores[:, 7:400] = scores[:, 7:8]  # ties across the window
    seq_lens = torch.tensor(
        [seq, max(seq // 3, 1), 1], dtype=torch.int32, device="cuda"
    )
    starts = (
        torch.tensor([0, 5, 13], dtype=torch.int32, device="cuda")
        if with_row_starts
        else None
    )
    ragged = torch.empty(batch, k, dtype=torch.int32, device="cuda")
    topk_transform_ragged_v2(
        scores.clone(),
        seq_lens,
        out_offsets=torch.zeros(batch, dtype=torch.int32, device="cuda"),
        out_indices=ragged,
        row_starts=starts,
        deterministic=True,
    )
    cand = torch.empty(batch, k, 2, dtype=torch.int32, device="cuda")
    topk_transform_candidates_v2(
        scores.clone(),
        seq_lens,
        out_candidates=cand,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        row_starts=starts,
        deterministic=True,
    )
    # Output slot order is not part of either contract; compare the sets.
    for row in range(batch):
        start = 0 if starts is None else int(starts[row])
        idx = ragged[row].long()
        idx = idx[idx >= 0].sort().values
        pos = cand[row, :, 1].long()
        valid = pos >= 0
        assert torch.equal(pos[valid].sort().values, idx * dcp_size + dcp_rank)
        local = (pos[valid] - dcp_rank) // dcp_size
        want_bits = scores[row, start + local].view(torch.int32)
        assert torch.equal(cand[row, valid, 0], want_bits)
        assert bool((cand[row, ~valid, 0] == 0).all())


def _candidate_sets(scores, seq_lens, k, dcp_size, dcp_rank, **kwargs):
    cand = torch.full(
        (scores.shape[0], k, 2), -7, dtype=torch.int32, device=scores.device
    )
    topk_transform_candidates_v2(
        scores.clone(),
        seq_lens,
        out_candidates=cand,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        deterministic=True,
        **kwargs,
    )
    torch.cuda.synchronize()
    return cand


_needs_cluster_candidates = pytest.mark.skipif(
    is_hip()
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="CUDA cluster candidate top-k (sm90 / sm100)",
)


@_needs_cluster_candidates
@pytest.mark.parametrize("cluster_size", [2, 4])
@torch.inference_mode()
def test_topk_v2_candidates_cluster_long_row_ties(cluster_size: int) -> None:
    k, dcp_size, dcp_rank, n = 32, 8, 3, (1 << 20) + 64
    scores = torch.zeros(1, n, device="cuda")
    scores[0, :64] = 1
    scores[0, -64:] = 1
    seq_lens = torch.tensor([n], dtype=torch.int32, device="cuda")
    got = _candidate_sets(
        scores,
        seq_lens,
        k,
        dcp_size,
        dcp_rank,
        cluster_size=cluster_size,
        cluster_floor=0,
    )
    want = torch.arange(k, dtype=torch.int32, device="cuda") * dcp_size + dcp_rank
    assert torch.equal(got[0, :, 1].sort().values, want)


@_needs_cluster_candidates
@pytest.mark.parametrize("cluster_size", [2, 4])
@pytest.mark.parametrize(
    "dist", ["randn", "ties", "dense_top", "top_bin"] + NARROW_DISTRIBUTIONS
)
@torch.inference_mode()
def test_topk_v2_candidates_cluster_matches_one_block(
    dist: str, cluster_size: int
) -> None:
    k, dcp_size, dcp_rank = 2048, 8, 3
    lens = [65536, 40000, 16385, 12000, 8193, 4097, 2049, 2048, 100, 0]
    width = max(lens) + 64
    torch.manual_seed(len(dist) * 31 + cluster_size)
    if dist == "randn":
        scores = torch.randn(len(lens), width, device="cuda")
    elif dist == "ties":
        scores = torch.randint(0, 200, (len(lens), width), device="cuda").float() / 8
    elif dist == "dense_top":
        scores = 10 + 0.3 * torch.randn(len(lens), width, device="cuda")
    elif dist == "top_bin":  # the threshold lands in the highest coarse bin
        scores = torch.zeros(len(lens), width, device="cuda")
        for row, n in enumerate(lens):
            scores[row, max(n - k, 0) : n] = 70000
    else:
        scores = _single_bin_scores(dist, len(lens), width, "cuda")
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    ref = _candidate_sets(scores, seq_lens, k, dcp_size, dcp_rank, cluster_size=0)
    for floor in (0, 16384):
        got = _candidate_sets(
            scores,
            seq_lens,
            k,
            dcp_size,
            dcp_rank,
            cluster_size=cluster_size,
            cluster_floor=floor,
        )
        for row, n in enumerate(lens):
            msg = (dist, floor, row, n)
            got_pos, got_order = got[row, :, 1].sort()
            valid = got_pos >= 0
            assert int(valid.sum()) == min(n, k), msg
            local = (got_pos[valid] - dcp_rank) // dcp_size
            assert bool((local >= 0).all() and (local < n).all()), msg
            assert local.unique().numel() == local.numel(), msg
            assert torch.equal(
                got[row, got_order[valid], 0],
                scores[row, local.long()].view(torch.int32),
            ), msg
            assert bool((got[row, got_order[~valid], 0] == 0).all()), msg
            if n == 0:
                continue
            want = scores[row, :n].topk(min(n, k)).values
            kth_copies = int((scores[row, :n] == want[-1]).sum())
            if n > k and kth_copies > k:
                assert torch.equal(
                    scores[row, local.long()].sort(descending=True).values, want
                ), msg
            else:
                want_pos, _ = ref[row, :, 1].sort()
                assert torch.equal(got_pos, want_pos), msg


@pytest.mark.skipif(not is_hip(), reason="ROCm split path")
@pytest.mark.parametrize("nvals", [64, 1024, 16384])
@pytest.mark.parametrize("batch,seq", [(1, 131072), (8, 131072)])
@torch.inference_mode()
def test_topk_v2_split_duplicate_scores(batch: int, seq: int, nvals: int) -> None:
    """Duplicate-heavy rows: the cross-block tie merge must stay exact.

    The split path resolves the threshold bin in whichever block arrives last,
    out of a tie list that every block appended to; a bin that overflows the
    list is rescanned instead (test_topk_v2_single_coarse_bin_is_exact). Rows
    of ``nvals`` distinct scores put ``seq / nvals`` exact duplicates at the
    cut, which is what fills it; the shapes are long and narrow enough to reach
    the split gate. CUDA runs its cluster path here, whose coarser histogram
    makes the same bin four times as wide, so this is the ROCm path's test.
    """
    k = 2048
    g = torch.Generator(device="cuda").manual_seed(nvals * 7 + seq)
    scores = torch.randint(0, nvals, (batch, seq), generator=g, device="cuda").float()
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    out = torch.full((batch, k), -2, dtype=torch.int32, device="cuda")

    topk_transform_paged_v2(scores, seq_lens, None, out, PAGE_SIZE, _plan(seq_lens))

    for row in range(batch):
        _assert_topk_values(scores[row], out[row], k)


@pytest.mark.skipif(not is_hip(), reason="ROCm zero-candidate overflow shortcut")
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (8, 8192),  # Register2
        (8, 16384),  # Register4
        (68, 65536),  # Streaming: below this batch's split floor
        (56, 262400),  # Split: late values land in the last long chunk
    ],
)
@torch.inference_mode()
def test_topk_v2_zero_candidates_graph_replay(batch: int, seq: int, k: int) -> None:
    """Only a proven all-+0 candidate bucket may bypass overflow refinement.

    The row itself is not constant: k-1 positive scores are above the bucket,
    and negative scores are below it. More than 2048 +0 candidates precede
    the final positions, where a small positive value must win the last slot.
    These positions are also late within the last split chunk; atomic staging
    order is deliberately not assumed. Reuse one graph across +0, nonzero,
    and +0 inputs to exercise the persistent split counters as well.
    """
    device = "cuda"
    scores_cpu = torch.zeros((batch, seq), dtype=torch.float32)
    scores_cpu[:, : k - 1] = 2.0
    scores_cpu[:, k + 128 : k + 640] = -1.0
    assert (scores_cpu[0] == 0).sum() > 2048
    scores = scores_cpu.to(device)
    lengths = torch.full((batch,), seq, dtype=torch.int32, device=device)
    page_table = (
        torch.arange((seq + PAGE_SIZE - 1) // PAGE_SIZE, device=device)
        .to(torch.int32)
        .expand(batch, -1)
        .contiguous()
    )
    metadata = _plan(lengths)
    out = torch.full((batch, k), -2, dtype=torch.int32, device=device)

    def transform():
        topk_transform_paged_v2(scores, lengths, page_table, out, PAGE_SIZE, metadata)

    # Compile and initialize persistent scratch outside graph capture.
    for _ in range(3):
        transform()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transform()

    # Both positives round to binary16 +0 but must survive exact fp32 ranking.
    # Write bits directly so CPU DAZ/FTZ cannot erase the subnormal test input.
    scores_cpu_bits = scores_cpu.view(torch.int32)
    for late_bits, negative_zero in [
        (0, False),
        (0x31800000, False),  # 2**-28
        (0, False),
        (1, False),  # smallest positive fp32 subnormal
        (0, False),
        (0, True),
        (0, False),
        (0x31800000, True),
        (0, False),
    ]:
        scores_cpu_bits[:, -2] = -2147483648 if negative_zero else 0
        scores_cpu_bits[:, -1] = late_bits
        scores.copy_(scores_cpu)
        assert torch.equal(
            scores[:, -2:].view(torch.int32).cpu(),
            scores_cpu[:, -2:].view(torch.int32),
        )
        out.fill_(-2)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        # Identity page mapping makes outputs raw indices. The CPU oracle
        # preserves subnormals and permits any legal, unique equal-score set.
        out_cpu = out.cpu()
        if late_bits != 0:
            assert (out_cpu == seq - 1).any(dim=1).all()
        for row in range(batch):
            _assert_topk_values(scores_cpu[row], out_cpu[row], k)


if __name__ == "__main__":
    if "--mode-check" in sys.argv:
        _mode_check()
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-v"]))
