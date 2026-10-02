# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of the fused Triton sparse-MLA prefill kernel against an
fp32 reference: base path, ragged ``-1`` padding, the union path's ownership mask
and dedup contract, int64 addressing, determinism, and the shared-memory step-down.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
    _FIT_TILE,
    _union_dedup,
    sparse_mla_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

D_QK, D_V, SM_SCALE = 576, 512, 0.0625


def _reference(q, kv, indices, d_v=D_V):
    """fp32 reference: each token attends over its own valid selected rows."""
    T, h, _ = q.shape
    S = kv.shape[0]
    out = torch.empty(T, h, d_v, dtype=torch.float32, device=q.device)
    qf, kf = q.float(), kv.float()
    for t in range(T):
        idx = indices[t]
        idx = idx[(idx >= 0) & (idx < S)]
        if idx.numel() == 0:
            out[t] = 0.0
            continue
        k = kf[idx]
        p = torch.softmax((qf[t] @ k.T) * SM_SCALE, dim=-1)
        out[t] = p @ k[:, :d_v]
    return out.to(torch.bfloat16)


def _qkv(T, S, h, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, h, D_QK, dtype=torch.bfloat16, device="cuda", generator=g)
    kv = torch.randn(S, D_QK, dtype=torch.bfloat16, device="cuda", generator=g)
    return q, kv, g


def _random_indices(T, topk, S, g, pad_frac=0.0):
    idx = torch.full((T, topk), -1, dtype=torch.int32, device="cuda")
    n = max(1, int(topk * (1.0 - pad_frac)))
    for t in range(T):
        idx[t, :n] = torch.randperm(S, device="cuda", generator=g)[:n].to(torch.int32)
    return idx


def _overlapping_indices(T, topk, S, g):
    """Selections that mostly agree between neighbouring tokens, as the indexer's
    do; uniform-random sets are nearly disjoint and would hide ownership-mask bugs.
    Rows stay unique within a token, which the union path relies on."""
    perm = torch.randperm(S, device="cuda", generator=g)
    pool, spare = perm[:topk], perm[topk:]
    n_keep = topk * 3 // 4
    n_fresh = topk - n_keep
    assert spare.numel() >= n_fresh, "S must exceed topk enough to vary the set"
    idx = torch.empty(T, topk, dtype=torch.int32, device="cuda")
    for t in range(T):
        keep = pool[torch.randperm(topk, device="cuda", generator=g)[:n_keep]]
        fresh = spare[
            torch.randperm(spare.numel(), device="cuda", generator=g)[:n_fresh]
        ]
        idx[t] = torch.cat([keep, fresh]).to(torch.int32)
    return idx


def _assert_matches(case, out, ref, tag, cos_min=0.999, max_abs=0.05):
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    mabs = (out.float() - ref.float()).abs().max().item()
    case.assertGreater(cos, cos_min, f"{tag}: cosine {cos:.6f}")
    case.assertLess(mabs, max_abs, f"{tag}: max_abs {mabs:.4f}")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestDSATritonSparseMLAPrefill(CustomTestCase):
    def test_base_path(self):
        for T, topk in ((512, 512), (2048, 2048), (37, 2048)):
            S = max(T, topk + 8)
            q, kv, g = _qkv(T, S, 8, seed=T)
            idx = _random_indices(T, topk, S, g)
            _assert_matches(
                self,
                sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V),
                _reference(q, kv, idx),
                f"base T={T} topk={topk}",
            )

    def test_ragged_minus_one_padding(self):
        T, topk, S = 1024, 2048, 2056
        q, kv, g = _qkv(T, S, 8, seed=7)
        idx = _random_indices(T, topk, S, g, pad_frac=0.6)
        _assert_matches(
            self,
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V),
            _reference(q, kv, idx),
            "ragged -1 padding",
        )

    def test_union_is_exact_on_overlapping_selections(self):
        # The union tile is only exercised when neighbouring tokens share rows;
        # each token must still see exactly its own set through the mask.
        for group in (2, 4):
            T, topk, S = (2048 // group) * group, 2048, 4096
            q, kv, g = _qkv(T, S, 8, seed=100 + group)
            idx = _overlapping_indices(T, topk, S, g)
            _assert_matches(
                self,
                sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, union=group),
                _reference(q, kv, idx),
                f"union G={group}",
            )

    def test_large_head_count_steps_down_instead_of_oom(self):
        # An oversized tile makes the first candidate fail on every GPU (on the
        # CI H100's 228 KB the tuned tile simply fits and nothing would step
        # down), and the cache is cleared so an earlier test's fit cannot be
        # moved to the front. The recorded fit must not be the requested tile.
        _FIT_TILE.clear()
        T, topk, S = 256, 512, 520
        q, kv, g = _qkv(T, S, 32, seed=9)
        idx = _random_indices(T, topk, S, g)
        _assert_matches(
            self,
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, config=(256, 8, 4)),
            _reference(q, kv, idx),
            "h=32 smem fallback",
        )
        self.assertNotEqual(_FIT_TILE[("base", 32, 0, q.device.index)], (256, 4))

    def test_union_large_head_count_steps_down(self):
        # Same for the union launcher, whose Q tile is num_heads * G rows and
        # runs out of shared memory sooner (16 heads at G=2 overflows SM120's
        # 100 KB with the tuned tile). The oversized tile forces the step-down
        # on every GPU; the union tile that fit must not be the requested one.
        _FIT_TILE.clear()
        T, topk, S = 512, 512, 1024
        q, kv, g = _qkv(T, S, 16, seed=21)
        idx = _overlapping_indices(T, topk, S, g)
        _assert_matches(
            self,
            sparse_mla_prefill(
                q, kv, idx, SM_SCALE, D_V, union=2, union_config=(256, 4, 4)
            ),
            _reference(q, kv, idx),
            "union G=2 h=16 smem fallback",
        )
        self.assertNotEqual(_FIT_TILE[("union", 16, 2, q.device.index)], (256, 4))

    def test_union_dedup_contract(self):
        # Each group's rows must come out exactly once, with ownership bits equal
        # to the OR of the tokens that selected them; -1 slots must be dropped,
        # not counted. Checked against a Python set reference so a membership
        # slip cannot hide behind the kernel's cosine gate.
        G, K, S = 4, 64, 256
        g = torch.Generator(device="cuda").manual_seed(77)
        idx = _overlapping_indices(3 * G, K, S, g)
        idx[1, 40:] = -1  # a ragged row in the first group
        idx[9, :] = -1  # a token with no selection at all
        uidx, ubits, ulen = _union_dedup(idx, G)
        self.assertEqual(tuple(uidx.shape), (3, G * K))
        for grp in range(3):
            rows = {}
            for tok in range(G):
                for v in idx[grp * G + tok].tolist():
                    if v >= 0:
                        rows[v] = rows.get(v, 0) | (1 << tok)
            n = int(ulen[grp])
            self.assertEqual(n, len(rows), f"group {grp}: unique count")
            got = dict(zip(uidx[grp, :n].tolist(), ubits[grp, :n].tolist()))
            self.assertEqual(len(got), n, f"group {grp}: a row was emitted twice")
            self.assertEqual(got, rows, f"group {grp}: rows or bits")

    def test_union_falls_back_when_tile_is_illegal(self):
        # Regression: h=4 with G=2 is an 8-row tile, which tl.dot cannot run;
        # the launcher only catches OutOfResources, so the compile error used to
        # reach the caller. The kernel entry must take the per-token path instead
        # and still match the reference.
        T, topk, S = 256, 256, 512
        q, kv, g = _qkv(T, S, 4, seed=91)
        idx = _overlapping_indices(T, topk, S, g)
        _assert_matches(
            self,
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, union=2),
            _reference(q, kv, idx),
            "union G=2 h=4 falls back",
        )

    def test_int64_indexing_matches_int32(self):
        # A KV pool past ~3.7M rows overflows int32 element offsets and the
        # launcher switches to int64 gather addressing. No test can allocate
        # that pool, so the mode is forced here instead: the two must agree
        # bitwise, or a large deployment silently reads the wrong rows.
        T, topk, S = 512, 512, 1024
        q, kv, g = _qkv(T, S, 8, seed=61)
        idx = _random_indices(T, topk, S, g)
        a = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, int64_indexing=False)
        b = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, int64_indexing=True)
        self.assertTrue(torch.equal(a, b), "int64 addressing changed the result")
        _assert_matches(self, b, _reference(q, kv, idx), "int64 indexing")

    def test_deterministic(self):
        # No split-K / atomics / partial merge, so repeated calls must be
        # bitwise identical -- the property that lets a served model be
        # reproduced run to run.
        T, topk, S = 1024, 2048, 2056
        q, kv, g = _qkv(T, S, 8, seed=11)
        idx = _random_indices(T, topk, S, g)
        a = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V)
        b = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V)
        self.assertTrue(torch.equal(a, b), "kernel output is not deterministic")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestDSATritonPrefillBackendAdapter(CustomTestCase):
    """The backend method driving the real kernel: values against the fp32
    reference and the `[num_tokens, num_heads, v_head_dim]` output contract
    that `_forward_flashmla_sparse` returns to the same caller."""

    def _backend(self, *, union=0):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_triton_union = union
        return backend

    def test_forward_matches_reference_and_output_contract(self):
        T, topk, S, h = 1024, 512, 2048, 8
        q, kv, g = _qkv(T, S, h, seed=51)
        idx = _overlapping_indices(T, topk, S, g)
        ref = _reference(q, kv, idx)

        for union in (0, 2, 4):
            with self.subTest(union=union):
                out = self._backend(union=union)._forward_triton_sparse_mla(
                    q_all=q,
                    kv_cache=kv,
                    page_table_1=idx,
                    sm_scale=SM_SCALE,
                    v_head_dim=D_V,
                )
                self.assertEqual(
                    tuple(out.shape),
                    (T, h, D_V),
                    "must match the [num_tokens, num_heads, v_head_dim] that "
                    "_forward_flashmla_sparse returns to the same caller",
                )
                self.assertEqual(out.dtype, torch.bfloat16)
                _assert_matches(self, out, ref, f"backend union={union}")


if __name__ == "__main__":
    unittest.main()
