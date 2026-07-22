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
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.jit_kernel.dsv4.topk import (
    plan_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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
        assert len(our) == min(
            k, L
        ), f"b={i} L={L} k={k}: {len(our)} valid != {min(k, L)}"
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


def _run(scores, seq_lens, page_table, inv_cpu, k):
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()
    out_cpu = out.cpu().tolist()
    return [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]


def _run_raw(scores, seq_lens, page_table, k):
    """Run the kernel and return its optional raw (pre-transform) top-k index
    output per row, dropping -1 padding -- the selected positions themselves,
    NOT the page-table transform of them."""
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()
    raw_cpu = raw.cpu().tolist()
    return [[v for v in raw_cpu[i] if v != -1] for i in range(batch)]


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


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (8, 256),  # trivial
        (8, 4096),  # register
        (4, 131072),  # fused small-batch cluster
        (64, 131072),  # persistent cluster + main<3> epilogue
        (256, 131072),  # non-cluster streaming
    ],
)
@torch.inference_mode()
def test_topk_v2_raw_indices(batch: int, seq: int, page_mode: str) -> None:
    """The optional raw-index output must be the pre-transform position of each
    transformed output slot (out[j] == page_to_indices(raw[j])), and -1 aligns."""
    k = 512
    torch.manual_seed(batch * 131 + seq)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=device)

    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()

    out_cpu, raw_cpu = out.cpu().tolist(), raw.cpu().tolist()
    for i in range(batch):
        for j in range(k):
            o, r = out_cpu[i][j], raw_cpu[i][j]
            if o == -1:
                assert r == -1, f"b={i} j={j}: out=-1 but raw={r}"
            else:
                inv = (int(inv_cpu[i][o >> PAGE_BITS]) << PAGE_BITS) | (o & PAGE_MASK)
                assert r == inv, f"b={i} j={j}: raw={r} != inverse(out)={inv}"


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2_output_indices(batch: int, seq: int, k: int) -> None:
    """Validate the raw (pre-transform) index output DIRECTLY against torch.topk.

    Unlike ``test_topk_v2`` -- which checks the page-transformed output and inverts
    it through the page table -- this exercises the selected indices themselves, so
    it isolates the top-k selection from the page-table transform. A permuted page
    table is used so raw != out, catching any bug that leaks transformed page
    indices into the raw buffer. Covers every dispatch template/boundary.
    """
    torch.manual_seed(batch * 100003 + seq * 7 + k + 1)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


# Negative lengths model DP-padded / idle-companion rows (e.g. -4 from GLM 5.2
# MTP draft-extend metadata, see #30378; DP-attention idle rows are the same
# class -- issue #25574). The kernels must treat them as empty (trivial
# all-(-1) output, via signed length comparisons) instead of reinterpreting
# them as ~4e9-token rows -- which was an illegal memory access, or a silently
# garbage output row when the unsigned chunk arithmetic happened to wrap. One
# config per dispatch shape so every device-side seq_len read (main kernel,
# fused small-batch cluster kernel, and the plan kernel's pool routing) is
# exercised.
NEGATIVE_LENGTH_CONFIGS = [
    [4096, 4096, -4, 400],  # level 0 (Register2)
    [9000, 9000, 9000, -4],  # level 1 (Register4)
    [40000] * 16 + [-1048576, 400],  # level 2 (Streaming; batch > 15 => floor 65536)
    [131072, 9000, 9000, -1],  # fused small-batch cluster
    [131072, 9000, 9000, -4],  # fused small-batch cluster
    [131072, 9000, 9000, -1048576],  # fused small-batch cluster
    [131072] * 20 + [400] * 10 + [-4],  # persistent pool + main<3>
    [131072] * 20 + [400] * 10 + [-1048576],  # persistent pool + main<3>
]


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("lens", NEGATIVE_LENGTH_CONFIGS)
@torch.inference_mode()
def test_topk_v2_negative_lengths(lens: list, k: int) -> None:
    """Negative rows -> all-(-1) out AND raw; non-negative rows unaffected."""
    torch.manual_seed(abs(sum(lens)) + k)
    device = "cuda"
    batch = len(lens)
    width = (max(lens) + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, "perm", device)
    out = torch.full((batch, k), 7, dtype=torch.int32, device=device)  # poison
    raw = torch.full((batch, k), 7, dtype=torch.int32, device=device)  # poison

    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()

    # Every row is fully written by the kernel (trivial rows pad with -1), so
    # the poison also proves the negative rows were actually processed.
    for i, L in enumerate(lens):
        if L < 0:
            assert (out[i] == -1).all(), f"row {i} (len={L}): out not all -1"
            assert (raw[i] == -1).all(), f"row {i} (len={L}): raw not all -1"

    # Non-negative rows must still match torch.topk (negative rows reference
    # as length 0, which _assert_topk_close checks as "no valid entries").
    clamped = torch.tensor([max(L, 0) for L in lens], dtype=torch.int32)
    out_cpu = out.cpu().tolist()
    our_raw = [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]
    ref_raw = _reference(scores, clamped, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, clamped, k)


@torch.inference_mode()
def test_topk_v2_plan_negative_lengths() -> None:
    """A negative row must not be counted by the plan's threshold heuristic.

    30 rows just above the 65536 candidate sit exactly at its cap (30). If the
    plan's counting loop read the negative row unsigned (~4e9 > every
    candidate), the count would become 31 > 30 and cluster_threshold would be
    lifted to the next candidate (98304); it must stay at the floor, and the
    negative row must not be routed to the persistent pool.
    """
    device = "cuda"
    lens = [98000] * 30 + [-4]
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    metadata = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()
    threshold, num_items = metadata[0].tolist()
    assert threshold == FLOOR, f"cluster_threshold {threshold} != {FLOOR}"
    assert num_items == 30, f"num_cluster_items {num_items} != 30"

    # A full-uint32 static threshold (> INT32_MAX) must not resurrect the
    # unsigned routing: the compaction loop proves rows positive (signed)
    # before the unsigned threshold compare, so nothing is routed here --
    # neither the negative row nor the 98000-token rows.
    metadata = plan_topk_v2(seq_lens, static_threshold=2**31)
    torch.cuda.synchronize()
    threshold_u32 = metadata[0, 0].item() & 0xFFFFFFFF
    assert threshold_u32 == 2**31, f"static threshold not honored: {threshold_u32}"
    assert metadata[0, 1].item() == 0, "rows routed past a 2^31 threshold"


@torch.inference_mode()
def test_topk_v1_negative_lengths() -> None:
    """The v1 kernel dispatches on the same per-row lengths; negative
    (padded) rows must take the same trivial all-(-1) path as in v2."""
    k = 512
    torch.manual_seed(20260714)
    device = "cuda"
    lens = [9000, 9000, -4, -1048576]
    batch = len(lens)
    width = (max(lens) + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, "perm", device)
    out = torch.full((batch, k), 7, dtype=torch.int32, device=device)  # poison

    topk_transform_512(scores, seq_lens, page_table, out, PAGE_SIZE)
    torch.cuda.synchronize()

    # naive_transform fills every slot >= length with -1, so the poison also
    # proves the negative rows were actually processed (not skipped).
    for i, L in enumerate(lens):
        if L < 0:
            assert (out[i] == -1).all(), f"row {i} (len={L}): out not all -1"
    clamped = torch.tensor([max(L, 0) for L in lens], dtype=torch.int32)
    out_cpu = out.cpu().tolist()
    our_raw = [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]
    ref_raw = _reference(scores, clamped, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, clamped, k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
