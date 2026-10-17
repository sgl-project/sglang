"""Unit tests for how KV-cache scales reach MHATokenToKVPool.set_kv_buffer.

The pool divides cache_k / cache_v by the scales it is given in place, so a caller
that reads K/V after the write must scale out of place and hand the pool no scale,
and a wrapper pool must not substitute a scale when its caller passed none.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_TOKENS, KV_HEADS, HEAD_DIM, Q_HEADS = 37, 2, 64, 8


class _Stop(Exception):
    pass


class _RecordingPool:
    def __init__(self, stop_after_write=False):
        self.calls = []
        self.stop_after_write = stop_after_write

    def set_kv_buffer(
        self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, **kwargs
    ):
        self.calls.append(
            SimpleNamespace(
                cache_k=cache_k.clone(),
                cache_v=cache_v.clone(),
                k_scale=k_scale,
                v_scale=v_scale,
            )
        )
        if self.stop_after_write:
            raise _Stop


def _make_kv():
    g = torch.Generator().manual_seed(0)
    # Contiguous views into one fused buffer, as the QKV projection hands them over.
    kv = torch.randn(2 * NUM_TOKENS, KV_HEADS, HEAD_DIM, generator=g).bfloat16()
    return kv[:NUM_TOKENS], kv[NUM_TOKENS:]


def _wrapper_cases():
    loc = torch.arange(NUM_TOKENS)
    for cls in (SWAKVPool, UnifiedSWAKVPool):
        for is_swa in (False, True):
            full, swa = _RecordingPool(), _RecordingPool()
            pool = SimpleNamespace(
                full_kv_pool=full, swa_kv_pool=swa, layers_mapping={0: (0, is_swa)}
            )
            inner = swa if is_swa else full
            yield (
                f"{cls.__name__}(swa={is_swa})",
                cls,
                pool,
                inner,
                KVWriteLoc(loc, loc),
            )
    inner = _RecordingPool()
    pool = SimpleNamespace(
        full_kv_pool=inner,
        use_mla=False,
        _transfer_full_attention_id=lambda layer_id: layer_id,
    )
    yield "HybridLinearKVPool", HybridLinearKVPool, pool, inner, KVWriteLoc(loc)


class TestWrapperPoolScaleForwarding(CustomTestCase):
    def test_unscaled_write_stays_unscaled(self):
        # A substituted 1.0 makes the MHA pool run a full div_ pass over K and V.
        k, v = _make_kv()
        for name, cls, pool, inner, loc in _wrapper_cases():
            with self.subTest(pool=name):
                cls.set_kv_buffer(pool, SimpleNamespace(layer_id=0), loc, k, v)
                self.assertEqual(len(inner.calls), 1)
                self.assertIsNone(inner.calls[0].k_scale)
                self.assertIsNone(inner.calls[0].v_scale)

    def test_explicit_scale_is_forwarded(self):
        k, v = _make_kv()
        k_scale, v_scale = torch.tensor(0.73), torch.tensor(0.61)
        for name, cls, pool, inner, loc in _wrapper_cases():
            with self.subTest(pool=name):
                cls.set_kv_buffer(
                    pool, SimpleNamespace(layer_id=0), loc, k, v, k_scale, v_scale
                )
                self.assertIs(inner.calls[0].k_scale, k_scale)
                self.assertIs(inner.calls[0].v_scale, v_scale)


class TestTritonExtendScaledKVWrite(CustomTestCase):
    def _save_kv(self, use_mla):
        k, v = _make_kv()
        k_before, v_before = k.clone(), v.clone()
        q = torch.randn(NUM_TOKENS, Q_HEADS * HEAD_DIM).bfloat16()
        # 0-dim fp32 parameters, as BaseKVCacheMethod creates them.
        layer = SimpleNamespace(
            layer_id=0,
            k_scale=torch.nn.Parameter(torch.tensor(0.73), requires_grad=False),
            v_scale=torch.nn.Parameter(torch.tensor(0.61), requires_grad=False),
        )
        pool = _RecordingPool(stop_after_write=True)
        backend = object.__new__(TritonAttnBackend)
        backend.use_dense_fp8_chunked_prefill = False
        backend.dcp_size = 1
        backend.use_mla = use_mla
        backend.token_to_kv_pool = pool
        backend.forward_metadata = SimpleNamespace(
            custom_mask=None, swa_out_cache_loc=None, out_cache_loc_full_physical=None
        )
        forward_batch = SimpleNamespace(
            out_cache_loc=torch.arange(NUM_TOKENS),
            out_cache_loc_is_physical=False,
            attn_attend_prefix_cache=None,
            _attn_output=torch.empty_like(q),
        )
        with self.assertRaises(_Stop):
            TritonAttnBackend.forward_extend(backend, q, k, v, layer, forward_batch)
        # The attention kernel that runs after the write reads the caller's k / v.
        self.assertTrue(torch.equal(k, k_before))
        self.assertTrue(torch.equal(v, v_before))
        self.assertEqual(len(pool.calls), 1)
        return pool.calls[0], k_before, v_before, layer

    def test_mha_write_is_prescaled_and_unscaled_at_pool(self):
        call, k, v, layer = self._save_kv(use_mla=False)
        self.assertIsNone(call.k_scale)
        self.assertIsNone(call.v_scale)
        # Same values, same dtype as the pool's in-place divide would produce.
        self.assertTrue(torch.equal(call.cache_k, k.clone().div_(layer.k_scale)))
        self.assertTrue(torch.equal(call.cache_v, v.clone().div_(layer.v_scale)))

    def test_mla_write_is_prescaled(self):
        call, k, v, layer = self._save_kv(use_mla=True)
        self.assertTrue(torch.equal(call.cache_k, k.clone().div_(layer.k_scale)))
        self.assertTrue(torch.equal(call.cache_v, v))


if __name__ == "__main__":
    unittest.main()
