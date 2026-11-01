"""trtllm_mla's cached eager-decode path must return what flashinfer returns.

``cached_trtllm_mla_decode`` keeps only flashinfer's autotuner tactic per shape
and builds the runner on every call from the caller's tensors and exact
``max_seq_len``, so eager decode skips the per-call tuning work of
``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla``. Outputs must match
the public API bit for bit, the cache must hold no tensors, one entry must serve
all layers and a growing ``max_seq_len``, and LSE requests must go through the
public API.

Runs on SM100 only, where these kernels exist.

    python -m pytest test/registered/attention/test_trtllm_mla_decode_cache.py -v
"""

import gc
import math
import unittest
import weakref
from unittest import mock

import torch

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")

_PAGE_SIZE = 64
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_QK_NOPE_HEAD_DIM = 128
_SCALE = 1.0 / math.sqrt(_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)


def _make_inputs(batch, heads, seq_lens, pages_per_seq, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    num_pages = batch * pages_per_seq
    q = torch.randn(
        (batch, 1, heads, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        generator=gen,
    )
    kv_cache = torch.randn(
        (num_pages, 1, _PAGE_SIZE, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        generator=gen,
    )
    # Shuffled physical pages so the page table is exercised.
    perm = torch.randperm(num_pages, device="cuda", generator=gen).to(torch.int32)
    block_tables = perm.view(batch, pages_per_seq).contiguous()
    seq_lens_t = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return q, kv_cache, block_tables, seq_lens_t


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm100_supported(), "requires SM100"
)
class TestTrtllmMlaDecodeCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import trtllm_mla_decode_cache

        cls.mod = trtllm_mla_decode_cache
        cls.ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def setUp(self):
        self.mod._TACTIC_CACHE.clear()

    def _kwargs(self, q, kv_cache, block_tables, seq_lens, max_seq_len, ws=None):
        return dict(
            query=q,
            kv_cache=kv_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=_SCALE,
            qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
            kv_lora_rank=_KV_LORA_RANK,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            workspace_buffer=self.ws if ws is None else ws,
        )

    def _cached(self, *args, **extra):
        return self.mod.cached_trtllm_mla_decode(**self._kwargs(*args), **extra)

    def _public(self, *args, **extra):
        return self.mod._public_decode(
            **self._kwargs(*args),
            bmm2_scale=1.0,
            multi_ctas_kv_counter_buffer=None,
            out=None,
            enable_pdl=True,
            backend="trtllm-gen",
            **extra,
        )

    def test_matches_public_api_bitwise(self):
        for heads in (16, 128):
            for batch, seq_len in ((1, 1024), (8, 2048), (64, 4096)):
                args = _make_inputs(
                    batch,
                    heads,
                    [seq_len] * batch,
                    seq_len // _PAGE_SIZE,
                    batch * heads,
                )
                ref = self._public(*args, seq_len, return_lse=False)
                for _ in range(3):  # the first call fills the cache, later calls hit it
                    out = self._cached(*args, seq_len)
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_non_power_of_two_max_seq_len_is_bitwise_exact(self):
        # Eager decode passes the batch maximum and 4096-token page tables. A
        # bucketed max_seq_len (1500 -> 2048) used to change the KV split here.
        for batch, heads, lens in ((32, 16, None), (3, 128, [1500, 1, 777])):
            seq_lens = lens or [1500] + [1500 - 7 * i for i in range(1, batch)]
            args = _make_inputs(batch, heads, seq_lens, 4096 // _PAGE_SIZE, seed=7)
            ref = self._public(*args, 1500, return_lse=False)
            for _ in range(2):
                out = self._cached(*args, 1500)
                torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_one_entry_for_growing_max_seq_len_and_all_layers(self):
        layers = [
            _make_inputs(4, 16, [1030] * 4, 2048 // _PAGE_SIZE, seed=11 + i)
            for i in range(3)
        ]
        # Steps 1030..1125 stay inside one 128-token K-tile (tile 9) and one
        # seq_len bucket (2048), so all 3 layers x 96 steps share 1 cache entry.
        for max_seq_len in range(1030, 1126):
            for args in layers:
                self._cached(*args, max_seq_len)
        self.assertEqual(len(self.mod._TACTIC_CACHE), 1)
        # Crossing the 1152 K-tile boundary adds at most one more entry when
        # cute-dsl is an eligible candidate.
        for max_seq_len in range(1126, 1200):
            for args in layers:
                self._cached(*args, max_seq_len)
        self.assertLessEqual(len(self.mod._TACTIC_CACHE), 2)
        q, kv, bt, sl = layers[1]
        ref = self._public(q, kv, bt, sl, 1199, return_lse=False)
        out = self._cached(q, kv, bt, sl, 1199)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_autotuned_auto_candidate_parity(self):
        from flashinfer.autotuner import AutoTuner, autotune

        tuner = AutoTuner.get()
        tuner.clear_cache()
        try:
            for i, (batch, heads, seq_len) in enumerate(
                ((1, 128, 1024), (8, 128, 2048), (32, 16, 1500))
            ):
                args = _make_inputs(
                    batch,
                    heads,
                    [seq_len] * batch,
                    4096 // _PAGE_SIZE,
                    seed=200 + i,
                )
                with autotune(True):
                    self._public(*args, seq_len, return_lse=False)
                torch.cuda.synchronize()
                ref = self._public(*args, seq_len, return_lse=False)
                self.mod._TACTIC_CACHE.clear()
                miss = self._cached(*args, seq_len)
                hit = self._cached(*args, seq_len)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(miss, ref))
                self.assertTrue(torch.equal(hit, ref))
        finally:
            tuner.clear_cache()

    def test_explicit_cute_dsl_backend_is_bitwise_exact(self):
        args = _make_inputs(8, 128, [1500] * 8, 4096 // _PAGE_SIZE, seed=301)
        ref = self.mod._public_decode(
            **self._kwargs(*args, 1500),
            bmm2_scale=1.0,
            multi_ctas_kv_counter_buffer=None,
            out=None,
            return_lse=False,
            enable_pdl=True,
            backend="cute-dsl",
        )
        miss = self._cached(*args, 1500, backend="cute-dsl")
        hit = self._cached(*args, 1500, backend="cute-dsl")
        self.assertTrue(torch.equal(miss, ref))
        self.assertTrue(torch.equal(hit, ref))

    def test_cache_holds_no_tensors(self):
        q, kv, bt, sl = _make_inputs(8, 128, [2048] * 8, 2048 // _PAGE_SIZE, seed=13)
        ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        for _ in range(2):
            self.mod.cached_trtllm_mla_decode(
                **self._kwargs(q, kv, bt, sl, 2048, ws=ws)
            )
        torch.cuda.synchronize()
        self.assertEqual(len(self.mod._TACTIC_CACHE), 1)
        kv_ref, ws_ref = weakref.ref(kv), weakref.ref(ws)
        del kv, ws
        gc.collect()
        self.assertIsNone(kv_ref())
        self.assertIsNone(ws_ref())

    def test_tuning_mode_bypasses_cache(self):
        args = _make_inputs(2, 16, [1024] * 2, 1024 // _PAGE_SIZE, seed=17)
        ref = self._public(*args, 1024, return_lse=False)
        with mock.patch.object(self.mod, "_tuner_token", return_value=None):
            out = self._cached(*args, 1024)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        self.assertEqual(len(self.mod._TACTIC_CACHE), 0)

    def test_return_lse_goes_through_public_api(self):
        args = _make_inputs(8, 128, [2048] * 8, 2048 // _PAGE_SIZE, seed=3)
        ref, ref_lse = self._public(*args, 2048, return_lse=True)
        for _ in range(3):
            out, lse = self._cached(*args, 2048, return_lse=True)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            torch.testing.assert_close(lse, ref_lse, rtol=0, atol=0)
        self.assertEqual(len(self.mod._TACTIC_CACHE), 0)

    def test_missing_runner_api_falls_back_to_public_api(self):
        args = _make_inputs(2, 16, [1024] * 2, 1024 // _PAGE_SIZE, seed=5)
        ref = self._public(*args, 1024, return_lse=False)
        with mock.patch.object(self.mod, "_FI_RUNNER_API", False):
            out = self._cached(*args, 1024)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        self.assertEqual(len(self.mod._TACTIC_CACHE), 0)

    def test_tuner_token_tracks_file_configs_and_active_contexts(self):
        fake = mock.MagicMock(
            is_tuning_mode=False,
            _active_tuning_contexts=0,
            _protocol_cache_generation=0,
            _dirty_seq=0,
            profiling_cache={},
            _file_configs={},
        )
        t0 = self.mod._tuner_token(fake)
        self.assertIsNotNone(t0)
        fake._file_configs = {"k": 1}
        t1 = self.mod._tuner_token(fake)
        self.assertNotEqual(t0, t1)
        fake._dirty_seq = 2
        t2 = self.mod._tuner_token(fake)
        self.assertNotEqual(t1, t2)
        fake._active_tuning_contexts = 1
        self.assertIsNone(self.mod._tuner_token(fake))

    def test_non_sm10x_delegates_to_public_api(self):
        args = _make_inputs(2, 16, [1024] * 2, 1024 // _PAGE_SIZE, seed=19)
        ref = self._public(*args, 1024, return_lse=False)
        with mock.patch.object(self.mod, "_device_info", return_value=(12, 188)):
            out = self._cached(*args, 1024)
        self.assertTrue(torch.equal(out, ref))
        self.assertEqual(len(self.mod._TACTIC_CACHE), 0)

    def test_invalid_and_boundary_inputs_match_public_api_contract(self):
        q, kv, bt, sl = _make_inputs(2, 16, [1024] * 2, 1024 // _PAGE_SIZE, seed=23)
        ref = self.mod._public_decode(
            **self._kwargs(q, kv, bt, sl, 1024),
            bmm2_scale=1.0,
            multi_ctas_kv_counter_buffer=None,
            out=None,
            return_lse=False,
            enable_pdl=None,
            backend="trtllm-gen",
        )
        out = self._cached(q, kv, bt, sl, 1024, enable_pdl=None)
        self.assertTrue(torch.equal(out, ref))
        for bad_kwargs in (
            dict(query=q.squeeze(1)),
            dict(seq_lens=None),
            dict(
                out=torch.empty(
                    (2, 1, 16, _KV_LORA_RANK), dtype=torch.float16, device="cuda"
                )
            ),
            dict(kv_lora_rank=128),
        ):
            kw = self._kwargs(q, kv, bt, sl, 1024)
            kw.update(bad_kwargs)
            with self.assertRaises((ValueError, TypeError, RuntimeError)):
                self.mod.cached_trtllm_mla_decode(**kw)

    def test_rejects_unknown_backend(self):
        args = _make_inputs(1, 16, [64], 1, seed=1)
        with self.assertRaises(ValueError):
            self._cached(*args, 64, backend="auto")


if __name__ == "__main__":
    unittest.main()
