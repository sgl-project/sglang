"""Correctness tests for the flashinfer GVR top-k DSv4 decode path.

Exercises ``gvr_topk_transform_decode`` (srt/layers/attention/dsv4/gvr_topk.py)
- the exact production state machine behind ``--dsa-topk-backend
flashinfer-gvr`` - over chained decode steps: gather previous-step hints per
request slot, run ``flashinfer.top_k_varlen(backend="gvr", compress_ratio=4)``,
scatter the output back as next-step hints, and page-transform the raw indices.

Derived properties pinned down here:

* GVR output is EXACT top-k regardless of hint quality (cold iota hints,
  chained unordered hints, stale clamped hints) - the guess/hint only seeds
  the threshold search. Verified by value-multiset comparison against
  ``torch.topk`` (immune to equal-score tie swaps).
* The documented ``pre_idx[:, 0] == argmax`` convention is a speed hint, not a
  correctness requirement: chaining GVR's own unordered output (which does not
  keep the argmax in column 0) must still be exact. If this test ever fails
  while the col0-forced variant passes, the production path needs a col0
  fixup.
* Cross-step hint bookkeeping: ``state.hints[layer, slot]`` must hold exactly
  the step's selected indices. A broken scatter would be *silent* in
  end-to-end output (hints only affect speed), so it is asserted directly.
* Rows with ``c4_len <= top_k`` emit exactly positions ``0..c4_len-1`` and pad
  with -1 (the downstream flash_mla contract); surplus kernel writes must not
  leak past the range mask.

Requires Blackwell (sm_100+) and a flashinfer build with ``top_k_varlen``
(flashinfer PR #3901); skipped otherwise.
"""

from __future__ import annotations

import sys

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from sglang.srt.layers.attention.dsv4.indexer import (
    GvrTopkState,
    gvr_topk_transform_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
NUM_LAYERS = 2


def _require_gvr():
    if not hasattr(flashinfer, "top_k_varlen"):
        pytest.skip("flashinfer build lacks top_k_varlen (PR #3901)")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("GVR top-k requires Blackwell (sm_100+)")


def _make_page_table(bs, num_pages, mode, device):
    """Return (page_table, inverse) like the DSv4 c4 page table.

    ``perm`` uses a distinct permutation per row so raw and transformed
    indices differ - catching raw/page index confusion in the transform.
    """
    if mode == "identity":
        pt = (
            torch.arange(num_pages, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(bs, -1)
            .contiguous()
        )
        return pt, pt.cpu()
    full = torch.stack(
        [torch.randperm(num_pages, device=device) for _ in range(bs)]
    ).to(torch.int32)
    inv = torch.empty_like(full)
    ar = torch.arange(num_pages, dtype=torch.int32, device=device)
    for r in range(bs):
        inv[r, full[r].long()] = ar
    return full, inv.cpu()


def _invert_row(out_row, inv_row):
    """Undo the page transform for one row, dropping -1 padding."""
    return [
        (int(inv_row[v >> PAGE_BITS]) << PAGE_BITS) | (v & PAGE_MASK)
        for v in out_row
        if v != -1
    ]


def _assert_row_exact(logits_row_cpu, sel, c4_len, top_k, ctx):
    """Value-multiset comparison vs torch.topk (tie-order agnostic)."""
    if c4_len <= top_k:
        assert sorted(sel) == list(range(c4_len)), (
            f"{ctx}: short row must emit exactly 0..{c4_len - 1}, got "
            f"{len(sel)} entries"
        )
        return
    assert len(sel) == top_k, f"{ctx}: {len(sel)} valid entries != {top_k}"
    assert len(set(sel)) == top_k, f"{ctx}: duplicate indices"
    assert max(sel) < c4_len, f"{ctx}: index {max(sel)} >= c4_len {c4_len}"
    ours = torch.sort(logits_row_cpu[sel], descending=True).values
    ref = torch.topk(logits_row_cpu[:c4_len], top_k).values
    assert torch.equal(ours, ref), f"{ctx}: selected value multiset != torch.topk"


def _run_chain(
    *,
    bs,
    n_pages,
    top_k,
    steps,
    device,
    page_mode="identity",
    fix_col0_argmax=False,
    start_lens=None,
    seed=0,
):
    """Drive the production state machine for ``steps`` chained decode steps."""
    torch.manual_seed(seed)
    N = n_pages * PAGE_SIZE
    num_slots = 2 * bs
    state = GvrTopkState(
        num_layers=NUM_LAYERS, num_slots=num_slots, top_k=top_k, device=device
    )
    # Non-trivial slot mapping (reversed odd slots) to exercise gather/scatter.
    req_pool_indices = torch.arange(
        num_slots - 1, -1, -2, dtype=torch.int64, device=device
    )
    page_table, inv_cpu = _make_page_table(bs, n_pages, page_mode, device)
    if start_lens is None:
        start_lens = torch.randint(
            top_k + 1, N - steps, (bs,), dtype=torch.int32, device=device
        )
    c4_seq_lens = start_lens.clone()

    layer_id = 1  # non-zero: catches hints indexed by the wrong layer axis
    for t in range(steps):
        logits = torch.randn(bs, N, dtype=torch.float32, device=device)
        if fix_col0_argmax:
            valid = torch.arange(N, device=device).unsqueeze(0) < c4_seq_lens.unsqueeze(
                1
            )
            argmax = (
                logits.masked_fill(~valid, float("-inf")).argmax(dim=1).to(torch.int32)
            )
            state.hints[layer_id, req_pool_indices, 0] = argmax
        out = torch.full((bs, top_k), -1, dtype=torch.int32, device=device)
        gvr_topk_transform_decode(
            logits=logits,
            c4_seq_lens=c4_seq_lens,
            page_table=page_table,
            out_page_indices=out,
            c4_page_size=PAGE_SIZE,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            state=state,
            top_k=top_k,
        )
        torch.cuda.synchronize()

        logits_cpu = logits.cpu()
        out_cpu = out.cpu().tolist()
        lens_cpu = c4_seq_lens.cpu().tolist()
        hints_cpu = state.hints[layer_id][req_pool_indices].cpu().tolist()
        for i in range(bs):
            ctx = f"step={t} row={i} L={lens_cpu[i]} k={top_k}"
            sel = _invert_row(out_cpu[i], inv_cpu[i])
            _assert_row_exact(logits_cpu[i], sel, lens_cpu[i], top_k, ctx)
            # Hint bookkeeping: silent-if-broken (hints only affect speed),
            # so pin the scattered content to this step's selection.
            if lens_cpu[i] > top_k:
                assert sorted(hints_cpu[i]) == sorted(sel), f"{ctx}: hints != output"
        c4_seq_lens = c4_seq_lens + 1  # KV grows; prev indices stay valid (cr=4)


@pytest.mark.parametrize("top_k", [512, 1024])
@torch.inference_mode()
def test_gvr_chained_decode_steps(top_k):
    """Chained steps with GVR's own unordered output as the next hint
    (production behavior: no argmax-in-column-0 fixup)."""
    _require_gvr()
    _run_chain(bs=8, n_pages=64, top_k=top_k, steps=6, device="cuda", seed=top_k)


@torch.inference_mode()
def test_gvr_chained_decode_steps_col0_argmax():
    """Same chain with the documented pre_idx[:,0]=argmax convention enforced.

    Passing here while test_gvr_chained_decode_steps fails would show the
    convention is load-bearing and the production path needs a col0 fixup.
    """
    _require_gvr()
    _run_chain(
        bs=8,
        n_pages=64,
        top_k=512,
        steps=6,
        device="cuda",
        fix_col0_argmax=True,
        seed=1,
    )


@torch.inference_mode()
def test_gvr_page_transform_permuted():
    """Distinct per-row page-table permutations: raw != transformed, so any
    leak of raw indices into the output (or vice versa) fails inversion."""
    _require_gvr()
    _run_chain(
        bs=4, n_pages=64, top_k=512, steps=3, device="cuda", page_mode="perm", seed=2
    )


@torch.inference_mode()
def test_gvr_cold_start_short_rows():
    """Rows shorter than top_k from iota cold-start hints (clamp path): the
    output must be exactly 0..L-1 plus -1 padding, nothing leaking past the
    range mask."""
    _require_gvr()
    device = "cuda"
    bs, top_k = 4, 512
    start = torch.tensor([1, 16, 64, 300], dtype=torch.int32, device=device)
    _run_chain(
        bs=bs,
        n_pages=16,
        top_k=top_k,
        steps=3,
        device=device,
        start_lens=start,
        seed=3,
    )


@torch.inference_mode()
def test_gvr_stale_hint_reset():
    """Slot reuse: hints hold indices from a longer evicted request, beyond the
    new batch's logits width. Bug regression: clamping such hints in-range
    produced all-duplicate pre_idx rows, which degenerate GVR's threshold
    search into emitting the first-k positions instead of the top-k. The
    production path must reset stale rows to iota and stay exact."""
    _require_gvr()
    device = "cuda"
    bs, top_k, n_pages = 4, 512, 16
    N = n_pages * PAGE_SIZE
    state = GvrTopkState(
        num_layers=NUM_LAYERS, num_slots=bs, top_k=top_k, device=device
    )
    # Poison every hint with positions far past the new logits width.
    state.hints.fill_(10 * N)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)
    page_table, inv_cpu = _make_page_table(bs, n_pages, "identity", device)
    torch.manual_seed(4)
    logits = torch.randn(bs, N, dtype=torch.float32, device=device)
    c4_seq_lens = torch.full((bs,), N - 8, dtype=torch.int32, device=device)
    out = torch.full((bs, top_k), -1, dtype=torch.int32, device=device)
    gvr_topk_transform_decode(
        logits=logits,
        c4_seq_lens=c4_seq_lens,
        page_table=page_table,
        out_page_indices=out,
        c4_page_size=PAGE_SIZE,
        layer_id=0,
        req_pool_indices=req_pool_indices,
        state=state,
        top_k=top_k,
    )
    torch.cuda.synchronize()
    logits_cpu = logits.cpu()
    out_cpu = out.cpu().tolist()
    for i in range(bs):
        sel = _invert_row(out_cpu[i], inv_cpu[i])
        _assert_row_exact(logits_cpu[i], sel, N - 8, top_k, f"stale row={i}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
