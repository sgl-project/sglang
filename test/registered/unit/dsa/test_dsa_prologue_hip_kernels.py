"""The fused DSA prologue kernels must write exactly what the stock launches they replace
write: the index-K page bytes (FP8 preshuffled or MXFP4, DCP-owner rows only), the query
codes / scales / head-gate weights, and the fp8_ds_mla KV rows. Runs the Triton kernels
in the interpreter on CPU against the stock Triton kernels (the reference of AITER's FP8
index-K store is index_k_norm_store, which mirrors it)."""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import unittest

import torch

from sglang.kernels.ops.attention.dsa import dsa_prologue_hip as fused
from sglang.kernels.ops.attention.dsa import fp4_index_hip, head_gate
from sglang.kernels.ops.attention.dsa.fp4_index_hip import store_fp4_index_k
from sglang.kernels.ops.attention.dsa.index_k_norm_store import index_k_norm_store
from sglang.kernels.ops.attention.fla.layernorm_gated import (
    _layer_norm_fwd_1pass_kernel,
)
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache_separate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

HEADS, D, PAGE, PAGES = 32, 128, 64, 6
EPS = 1e-6
SOFTMAX = D**-0.5


def _inputs(tokens, seed, front_cols=3808):
    """Query [T, 32, 128]; key and head gate as column slices of a merged GEMM output."""
    g = torch.Generator().manual_seed(seed)
    front = (torch.randn(tokens, front_cols, generator=g) * 3).to(torch.bfloat16)
    key = front[:, 2112 : 2112 + D]
    gate = front[:, 2240 : 2240 + HEADS]
    query = (torch.randn(tokens, HEADS, D, generator=g) * 2).to(torch.bfloat16)
    w = 1 + 0.1 * torch.randn(D, generator=g)
    b = 0.1 * torch.randn(D, generator=g)
    return query, gate, key, w, b


def _loc(tokens, seed, dcp_size):
    g = torch.Generator().manual_seed(seed + 100)
    loc = torch.randperm(PAGES * PAGE * dcp_size, generator=g)[:tokens].to(torch.int64)
    loc[::5] = -1
    return loc


def _local(loc, dcp_size, dcp_rank):
    # dsa_indexer_hip._localize_store_loc
    if dcp_size == 1:
        return loc
    owned = (loc >= 0) & (loc % dcp_size == dcp_rank)
    return torch.where(owned, loc // dcp_size, -1)


def _stock_q_fp8(query, gate, scale_fmt):
    """indexer_q_quant_head_gate's launch (its wrapper asserts a GPU tensor)."""
    tokens, heads, _ = query.shape
    q8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    weights = torch.empty((tokens, heads, 1), dtype=torch.float32)
    head_gate._indexer_q_quant_head_gate_kernel[(-(-tokens * heads // 32),)](
        query,
        gate,
        q8,
        weights,
        tokens * heads,
        heads,
        gate.stride(0),
        heads**-0.5,
        SOFTMAX,
        scale_fmt is not None,
        32,
    )
    return q8, weights


def _stock_q_fp4(query, gate):
    """indexer_q_fp4_head_gate's launch."""
    tokens, heads, _ = query.shape
    payload = torch.empty((tokens, heads, 64), dtype=torch.uint8)
    scale = torch.empty((tokens, 1, 4, 16, 4), dtype=torch.uint8)
    weights = torch.empty((tokens, heads, 1), dtype=torch.float32)
    fp4_index_hip._q_fp4_head_gate_kernel[(tokens, 4)](
        query,
        gate,
        payload,
        scale,
        weights,
        heads,
        gate.stride(0),
        heads**-0.5,
        SOFTMAX,
        16,
    )
    return payload, scale, weights


def _stock_layer_norm(key, w, b):
    """Fp32LayerNormHip's launch (rms_norm_gated, no gate, one row per program)."""
    m = key.shape[0]
    out = torch.empty((m, D), dtype=key.dtype)
    stats = torch.empty(m, dtype=torch.float32)
    _layer_norm_fwd_1pass_kernel[(m, 1)](
        key,
        out,
        w,
        b,
        None,
        stats,
        stats.clone(),
        key.stride(0),
        out.stride(0),
        0,
        0,
        0,
        m,
        D,
        EPS,
        BLOCK_N=D,
        ROWS_PER_BLOCK=1,
        HAS_BIAS=True,
        HAS_Z=False,
        Z_IS_3D=False,
        Z_HEADS=1,
        NORM_BEFORE_GATE=True,
        IS_RMS_NORM=False,
        ACTIVATION="swish",
        num_warps=1,
    )
    return out


class TestIndexPrepFp8(CustomTestCase):
    def _run(self, tokens, dcp_size, dcp_rank, scale_fmt="ue8m0", seed=0):
        query, gate, key, w, b = _inputs(tokens, seed)
        loc = _loc(tokens, seed, dcp_size)
        cache = torch.randint(0, 255, (PAGES, PAGE * (D + 4)), dtype=torch.uint8)
        ref_cache = cache.clone()

        q8, weights = fused.index_qk_prep_fp8(
            query=query,
            gate=gate,
            softmax_scale=SOFTMAX,
            scale_fmt=scale_fmt,
            key=key,
            norm_weight=w,
            norm_bias=b,
            eps=EPS,
            index_k_cache=cache,
            loc=loc,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        ref_q8, ref_weights = _stock_q_fp8(query, gate, scale_fmt)
        index_k_norm_store(
            key.contiguous(),
            w,
            b,
            EPS,
            ref_cache.view(torch.float8_e4m3fn),
            _local(loc, dcp_size, dcp_rank),
        )
        self.assertTrue(torch.equal(q8.view(torch.uint8), ref_q8.view(torch.uint8)))
        self.assertTrue(torch.equal(weights, ref_weights))
        self.assertTrue(torch.equal(cache, ref_cache))
        return cache

    def test_matches_stock_launches(self):
        # verify rows 8 (C1 x 8) and 48 (C12-class), and DCP8 owner rows
        self._run(tokens=8, dcp_size=1, dcp_rank=0)
        self._run(tokens=48, dcp_size=8, dcp_rank=3, seed=1)

    def test_other_ranks_rows_untouched(self):
        query, gate, key, w, b = _inputs(16, 7)
        loc = torch.arange(16, dtype=torch.int64) * 8 + 5  # all owned by rank 5
        cache = torch.zeros((PAGES, PAGE * (D + 4)), dtype=torch.uint8)
        fused.index_qk_prep_fp8(
            query=query,
            gate=gate,
            softmax_scale=SOFTMAX,
            scale_fmt="ue8m0",
            key=key,
            norm_weight=w,
            norm_bias=b,
            eps=EPS,
            index_k_cache=cache,
            loc=loc,
            dcp_size=8,
            dcp_rank=2,
        )
        self.assertEqual(int(cache.count_nonzero()), 0)


class TestIndexPrepFp4(CustomTestCase):
    def test_matches_stock_launches(self):
        for tokens, dcp_size, dcp_rank in ((8, 1, 0), (40, 8, 6)):
            query, gate, key, w, b = _inputs(tokens, tokens)
            loc = _loc(tokens, tokens, dcp_size)
            payload = torch.randint(0, 255, (PAGES, 4096), dtype=torch.uint8)
            scale = torch.randint(0, 255, (PAGES, 256), dtype=torch.uint8)
            ref_payload, ref_scale = payload.clone(), scale.clone()

            q, weights = fused.index_qk_prep_fp4(
                query=query,
                gate=gate,
                softmax_scale=SOFTMAX,
                key=key,
                norm_weight=w,
                norm_bias=b,
                eps=EPS,
                payload=payload,
                scale=scale,
                loc=loc,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            ref_payload_q, ref_scale_q, ref_weights = _stock_q_fp4(query, gate)
            store_fp4_index_k(
                _stock_layer_norm(key, w, b),
                ref_payload,
                ref_scale,
                _local(loc, dcp_size, dcp_rank),
            )
            self.assertTrue(torch.equal(q.payload, ref_payload_q))
            self.assertTrue(torch.equal(q.scale, ref_scale_q))
            self.assertTrue(torch.equal(weights, ref_weights))
            self.assertTrue(torch.equal(payload, ref_payload))
            self.assertTrue(torch.equal(scale, ref_scale))


class TestFp8DsMlaStore(CustomTestCase):
    def _reference(self, buf, loc, nope, rope, ue8m0, dcp_size, dcp_rank):
        """quantize_k_cache_separate, then set_mla_kv_buffer_kernel's owner rule."""
        nope_b, rope_b = quantize_k_cache_separate(nope, rope, scale_ue8m0=ue8m0)
        rows = torch.cat([nope_b, rope_b], dim=-1).squeeze(1)
        for t, l in enumerate(loc.tolist()):
            if l != 0 and l % dcp_size == dcp_rank:
                buf[l // dcp_size, 0] = rows[t]

    def test_matches_stock_launches(self):
        g = torch.Generator().manual_seed(0)
        for tokens, ue8m0, dcp_size, dcp_rank in ((8, True, 1, 0), (48, False, 8, 2)):
            latent = (torch.randn(tokens, 2112, generator=g) * 4).to(torch.bfloat16)
            nope = latent[:, 1536:2048].unsqueeze(1)
            rope = latent[:, 2048:2112].unsqueeze(1)
            loc = torch.randperm(64 * dcp_size, generator=g)[:tokens]
            loc[0] = 0  # the reserved padding slot is never written
            buf = torch.randint(0, 255, (64, 1, 656), dtype=torch.uint8)
            ref = buf.clone()
            fused.fp8_ds_mla_store(
                buf,
                loc,
                nope,
                rope,
                scale_ue8m0=ue8m0,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            self._reference(ref, loc, nope, rope, ue8m0, dcp_size, dcp_rank)
            self.assertTrue(torch.equal(buf, ref))


if __name__ == "__main__":
    unittest.main()
