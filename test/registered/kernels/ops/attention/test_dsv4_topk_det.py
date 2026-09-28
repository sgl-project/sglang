"""DeepSeek-V4 indexer top-k (v1 and v2 JIT kernels): tie handling.

The selected set must equal a CPU stable sort by (score desc, position asc) and be
identical across runs, on tied rows, on tie-free rows and (v1) on rows whose
threshold bin overflows the 8192-entry candidate buffer. page_table is the
identity, so the output page indices equal raw positions.
"""

import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4 import topk as topk_mod
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOPK = 512
PAGE = 64
RUNS = 20
dev = "cuda"


def identity_page_table(B, L):
    npages = (L + PAGE - 1) // PAGE
    return torch.arange(npages, dtype=torch.int32, device=dev).repeat(B, 1).contiguous()


def cpu_reference(scores_cpu, lens):
    """Stable top-k SET per row: sort by (score desc, position asc)."""
    B = scores_cpu.shape[0]
    ref = []
    for r in range(B):
        n = int(lens[r])
        s = scores_cpu[r, :n]
        if n <= TOPK:
            ref.append(
                np.sort(
                    np.concatenate([np.arange(n), -np.ones(TOPK - n, dtype=np.int64)])
                )
            )
            continue
        # primary key -s (score desc), secondary key position (asc)
        order = np.lexsort((np.arange(n), -s))
        ref.append(np.sort(order[:TOPK]))
    return np.stack(ref)


def sorted_sets(out):
    return torch.sort(out, dim=1).values.cpu().numpy()


def make_tied_rows(g, B, Lmin, Lmax):
    """Heavily tied logits: ReLU-like with many exact zeros, and coarse quantization."""
    lens = torch.randint(Lmin, Lmax + 1, (B,), generator=g).to(torch.int32)
    L = int(lens.max())
    scores = torch.zeros(B, L, dtype=torch.float32)
    for r in range(B):
        n = int(lens[r])
        kind = r % 4
        if kind == 0:
            # relu of gaussian, quantized to 1/8 -> many ties incl. exact zeros
            x = torch.randn(n, generator=g).clamp_min(0)
            x = torch.round(x * 8) / 8
        elif kind == 1:
            # < 512 positives, rest exactly zero (threshold at the zero bin)
            x = torch.zeros(n)
            npos = int(torch.randint(50, 500, (1,), generator=g))
            idx = torch.randperm(n, generator=g)[:npos]
            x[idx] = torch.rand(npos, generator=g) + 0.1
        elif kind == 2:
            # a few distinct levels only
            x = torch.randint(0, 6, (n,), generator=g).float() * 0.5
        else:
            # tie exactly at rank 512/513: top-600 all equal 3.0
            x = torch.rand(n, generator=g)
            idx = torch.randperm(n, generator=g)[:600]
            x[idx] = 3.0
        scores[r, :n] = x
    return scores, lens


def make_tiefree_rows(g, B, Lmin, Lmax):
    lens = torch.randint(Lmin, Lmax + 1, (B,), generator=g).to(torch.int32)
    L = int(lens.max())
    scores = torch.zeros(B, L, dtype=torch.float32)
    for r in range(B):
        n = int(lens[r])
        x = (
            torch.randperm(n, generator=g).float() * 1e-3
            + torch.rand(n, generator=g) * 1e-7
        )
        scores[r, :n] = x
    # ensure no exact duplicates per row
    for r in range(B):
        n = int(lens[r])
        assert torch.unique(scores[r, :n]).numel() == n, "tie-free construction failed"
    return scores, lens


def make_overflow_rows(g, B, L=17000):
    """Threshold bin with > 8192 candidates (SMEM_INPUT_SIZE), two ways."""
    lens = torch.full((B,), L, dtype=torch.int32)
    scores = torch.zeros(B, L, dtype=torch.float32)
    for r in range(B):
        if r % 3 == 0:
            # 400 positives + 16600 exact zeros -> need 112 zeros from a
            # 16600-candidate bin
            x = torch.zeros(L)
            idx = torch.randperm(L, generator=g)[:400]
            x[idx] = torch.rand(400, generator=g) + 0.5
        elif r % 3 == 1:
            # 100 large + 9000 exactly 1.0 + rest small distinct: stage-1 bin(1.0)
            # has 9000 > 8192
            x = torch.rand(L, generator=g) * 0.01
            perm = torch.randperm(L, generator=g)
            x[perm[:100]] = 5.0 + torch.rand(100, generator=g)
            x[perm[100:9100]] = 1.0
        else:
            # 9000 DISTINCT values inside ONE fp16 bin [1.0, 1.2) at the threshold
            # -> stage-1 overflow with distinct candidates: the re-scan + sub-bin
            # histogram must pick the 412 largest of them exactly, not refine an
            # arrival-ordered 8192-subset.
            x = torch.rand(L, generator=g) * 0.01
            perm = torch.randperm(L, generator=g)
            x[perm[:100]] = 5.0 + torch.rand(100, generator=g)
            x[perm[100:9100]] = 1.0 + torch.rand(9000, generator=g) * 0.2
        scores[r] = x
    return scores, lens


def check_rows(name, out_sets, ref, lens):
    bad = np.nonzero((out_sets != ref).any(axis=1))[0]
    print(
        f"  {name}: rows={len(ref)} mismatching_rows={len(bad)}"
        + (
            f" first={bad[:5].tolist()} lens={lens[bad[:5]].tolist()}"
            if len(bad)
            else ""
        )
    )
    return len(bad) == 0


def pad4(x):  # v2 needs score_stride % 4 == 0
    return F.pad(x, (0, (-x.shape[1]) % 4))


def coarse_bin12(x):
    h = torch.tensor(x).to(torch.float16).view(torch.int16).numpy().astype(np.uint16)
    key = np.where(h & 0x8000, ~h, h | 0x8000).astype(np.uint16)
    return key >> 4


def threshold_bin_count(s, n, k=512):
    b = coarse_bin12(s[:n])
    order = np.argsort(-b, kind="stable")
    thr = b[order[k - 1]]  # bin of the k-th largest key
    return int((b == thr).sum())


def run(version, scores, seq_lens, page_table, out):
    if version == "v1":
        topk_mod.topk_transform_paged(scores, seq_lens, page_table, out, PAGE, None)
    else:
        meta = topk_mod.plan_topk_v2(seq_lens, static_threshold=-1)
        topk_mod.topk_transform_paged_v2(
            scores, seq_lens, page_table, out, PAGE, meta, None
        )


def prepare(version, scores_cpu, lens):
    B = scores_cpu.shape[0]
    scores = (pad4(scores_cpu) if version == "v2" else scores_cpu).to(dev).contiguous()
    pt = identity_page_table(B, scores.shape[1])
    out = torch.empty(B, TOPK, dtype=torch.int32, device=dev)
    return scores, lens.to(dev), pt, out


def check_tied_rows(version, B, Lmin, Lmax):
    g = torch.Generator().manual_seed(1234)
    scores_cpu, lens = make_tied_rows(g, B, Lmin, Lmax)
    scores, seq_lens, pt, out = prepare(version, scores_cpu, lens)
    ref = cpu_reference(scores_cpu.numpy(), lens.numpy())
    rows = np.ones(B, dtype=bool)
    if version == "v2":
        # v2 keeps an arrival-ordered subset when the threshold coarse bin holds more
        # than 2048 candidates (collect pass); only rows below that are covered.
        cnt = np.array(
            [threshold_bin_count(scores_cpu[r].numpy(), int(lens[r])) for r in range(B)]
        )
        rows = cnt <= 2048
        assert rows.any()

    run(version, scores, seq_lens, pt, out)
    s0 = sorted_sets(out)
    assert check_rows(version, s0[rows], ref[rows], lens.numpy()[rows])
    for _ in range(RUNS):
        run(version, scores, seq_lens, pt, out)
        assert np.array_equal(sorted_sets(out)[rows], s0[rows])


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_tied_rows_match_stable_sort_and_are_run_to_run_identical(version):
    check_tied_rows(version, 512, 513, 17500)


@pytest.mark.parametrize("seq_len", [65536, 131072])
def test_v2_tied_rows_cluster_path_deterministic(seq_len):
    # A batch <= 15 with rows above the static cluster floor (24576 on Blackwell,
    # 32768 on Hopper) takes the small-batch cluster kernel; ties there must still
    # resolve by lowest position.
    check_tied_rows("v2", 12, seq_len, seq_len)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_tiefree_rows_match_reference(version):
    g = torch.Generator().manual_seed(1234)
    scores_cpu, lens = make_tiefree_rows(g, 256, 300, 17500)
    scores, seq_lens, pt, out = prepare(version, scores_cpu, lens)
    ref = cpu_reference(scores_cpu.numpy(), lens.numpy())
    run(version, scores, seq_lens, pt, out)
    assert check_rows(version, sorted_sets(out), ref, lens.numpy())


def test_v1_overflow_rows_exact():
    g = torch.Generator().manual_seed(1234)
    scores_cpu, lens = make_overflow_rows(g, 33)
    scores, seq_lens, pt, out = prepare("v1", scores_cpu, lens)
    ref = cpu_reference(scores_cpu.numpy(), lens.numpy())
    run("v1", scores, seq_lens, pt, out)
    s0 = sorted_sets(out)
    assert check_rows("v1", s0, ref, lens.numpy())
    for _ in range(RUNS):
        run("v1", scores, seq_lens, pt, out)
        assert np.array_equal(sorted_sets(out), s0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
