"""Unit tests for sliding-window layers on the MLX radix/pool KV path.

The shared ``MlxAttentionKVPool`` stores full-attention layers only,
sliding-window layers keep window-bounded per-request storage, and a
radix prefix hit on an SWA model recomputes the whole prefix.  Scheduler
bookkeeping stays in the unclamped coordinates, so these tests drive
``MlxModelRunner`` directly with hand-built slot ids, mirroring how the
tp_worker calls it.
"""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm"

if _HAS_MLX:
    import mlx.core as mx
    from mlx_lm.models import gpt_oss

    from sglang.srt.hardware_backend.mlx.aot import (
        MlxAOTKernelContext,
        MlxAOTKernelSet,
        MlxAOTRoPEContext,
        MlxAOTRoPEKernel,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        BatchedDecodeContext,
        ContiguousAttentionKVCache,
        MlxAttentionKVPool,
        MLXAttentionWrapper,
        WindowedAttentionKVCache,
        find_attention_layers,
        get_layer_window_sizes,
        patch_model_attention,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache.layout import MlxModelCacheLayout
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

TINY_WINDOW = 8


def _tiny_gpt_oss_model():
    """Randomly initialized 4-layer gpt_oss with alternating sliding/full layers.

    Mirrors test_windowed_kv_cache.py's builder (kept local: the registered
    unit-test directory is not an importable package).
    """
    args = gpt_oss.ModelArgs(
        num_hidden_layers=4,
        num_local_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=TINY_WINDOW,
        rope_theta=150000,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096,
            "truncate": False,
        },
    )
    return gpt_oss.Model(args)


def _stub_runner(model, disable_radix_cache, pool_size=64):
    """Surgically build a runner around an already-loaded tiny model."""
    layers, attrs = find_attention_layers(model)
    runner = MlxModelRunner.__new__(MlxModelRunner)
    runner.model = model
    runner.disable_radix_cache = disable_radix_cache
    runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
        layers, attrs, layer_window_sizes=get_layer_window_sizes(model)
    )
    runner._max_seq_len = 64
    runner._cache_pool = []
    runner._req_caches = {}
    runner._req_token_ids = {}
    runner._req_sampling = {}
    runner._req_pool_idx = {}
    runner._req_synced_offset = {}
    runner._req_to_token_pool = None
    runner._attention_kv_pool = None
    runner._decode_step_ct = 0
    runner._clear_steps = 0
    runner._aot_kernels = MlxAOTKernelSet()
    runner._pool_size = pool_size
    if not disable_radix_cache:
        runner.init_cache_pools(None)
    return runner


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSwaLayoutAndPoolContract(CustomTestCase):
    def _layout(self, with_windows=True):
        model = _tiny_gpt_oss_model()
        layers, attrs = find_attention_layers(model)
        return MlxModelCacheLayout.from_attention_discovery(
            layers, attrs, get_layer_window_sizes(model) if with_windows else None
        )

    def test_partition_and_dense_full_pool_index(self):
        layout = self._layout()
        self.assertEqual(layout.attention_layer_indices, (0, 1, 2, 3))
        self.assertEqual(layout.swa_attention_layer_indices, (0, 2))
        self.assertEqual(layout.full_attention_layer_indices, (1, 3))
        # Dense over full layers only, so it differs from the cache index.
        self.assertEqual(layout.full_kv_pool_index_by_layer, {1: 0, 3: 1})
        self.assertEqual(layout.attention_pool_index_by_layer, {0: 0, 1: 1, 2: 2, 3: 3})
        with self.assertRaises(KeyError):
            layout.full_kv_pool_index(0)

        # Without a window map the two indices coincide (pre-SWA behavior).
        plain = self._layout(with_windows=False)
        self.assertFalse(plain.has_sliding_window_layers)
        self.assertEqual(
            plain.full_kv_pool_index_by_layer, plain.attention_pool_index_by_layer
        )

    def test_sliding_window_model_gets_no_pool(self):
        # An SWA prefix hit recomputes the prefix instead of gathering it, so
        # the shared pool would have no reader. Allocating it would burn the
        # whole auto-sized KV budget on a write-only buffer.
        runner = _stub_runner(_tiny_gpt_oss_model(), disable_radix_cache=False)
        self.assertTrue(runner._cache_layout.has_sliding_window_layers)
        self.assertIsNone(runner._attention_kv_pool)
        # The layer-type split still resolves -- it is the seam the shared
        # window-aware SWA pool will build on.
        self.assertEqual(runner._cache_layout.full_kv_pool_index_by_layer, {1: 0, 3: 1})

    def test_pool_covers_every_layer_without_sliding_windows(self):
        model = _tiny_gpt_oss_model()
        layers, attrs = find_attention_layers(model)
        runner = _stub_runner(model, disable_radix_cache=True)
        runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            layers, attrs
        )
        runner.disable_radix_cache = False
        runner.init_cache_pools(None)
        self.assertEqual(runner._attention_kv_pool.num_layers, 4)
        self.assertEqual(runner._attention_kv_pool.pool_size, 65)

    def test_all_sliding_window_model_gets_no_pool(self):
        # An all-SWA model has nothing to pool. Pool construction must skip
        # out, and pool sizing must still land on a finite slot count rather
        # than dividing by zero bytes per slot.
        runner = _stub_runner(_tiny_gpt_oss_model(), disable_radix_cache=False)
        layers, attrs = find_attention_layers(runner.model)
        runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            layers, attrs, {idx: TINY_WINDOW for idx in range(4)}
        )
        self.assertEqual(runner._cache_layout.full_attention_layer_indices, ())
        self.assertEqual(runner._cache_layout.full_kv_pool_index_by_layer, {})

        runner._attention_kv_pool = None
        runner.init_cache_pools(None)
        self.assertIsNone(runner._attention_kv_pool)

        runner._mem_fraction_static = 0.5
        self.assertGreater(runner._compute_pool_size(None), 0)

    def test_sliding_flag_without_window_map_still_rejected(self):
        model = _tiny_gpt_oss_model()
        patch_model_attention(model)
        model.model.layers[0].self_attn._inner.is_sliding = True
        runner = _stub_runner(model, disable_radix_cache=True)
        # With the container window map the flagged layer is bounded: fine.
        runner._get_attn_config()
        # Without a resolvable window the layer cannot be bounded: reject.
        layers, attrs = find_attention_layers(model)
        runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            layers, attrs
        )
        with self.assertRaises(NotImplementedError):
            runner._get_attn_config()

    def test_sync_writes_full_layers_only(self):
        runner = _stub_runner(_tiny_gpt_oss_model(), disable_radix_cache=False)
        # init_cache_pools skips the pool on an SWA model (see
        # test_sliding_window_model_gets_no_pool), so attach one by hand: the
        # layer-type filtering in _sync_new_kv_to_pool is what the shared
        # window-aware SWA pool will rely on, and it must stay correct.
        self.assertIsNone(runner._attention_kv_pool)
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=runner._pool_size + 1,
            num_layers=runner._cache_layout.num_full_attention_layers,
            n_kv_heads=2,
            head_dim=16,
            dtype=mx.float32,
        )
        cache = runner._new_native_cache()
        per_layer_k = {}
        for layer_idx in range(4):
            k = mx.full((1, 2, 5, 16), float(layer_idx + 1))
            cache[layer_idx].update_and_fetch(k, -k)
            per_layer_k[layer_idx] = k
        slot_ids = [7, 9, 11]
        runner._sync_new_kv_to_pool(cache, cache_start=2, slot_ids=slot_ids)
        for layer_idx, pool_idx in ((1, 0), (3, 1)):
            got_k, got_v = runner._attention_kv_pool.get_kv(
                pool_idx, mx.array(slot_ids, dtype=mx.int32)
            )
            want = per_layer_k[layer_idx][0, :, 2:5, :].transpose(1, 0, 2)
            self.assertTrue(mx.array_equal(got_k, want).item())
            self.assertTrue(mx.array_equal(got_v, -want).item())
        # Untouched pool slots stay zero (nothing wrote outside the slots).
        rest_k, _ = runner._attention_kv_pool.get_kv(
            0, mx.array([1, 2, 3], dtype=mx.int32)
        )
        self.assertEqual(mx.abs(rest_k).max().item(), 0.0)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSwaRadixPath(CustomTestCase):
    """Radix-path prefill/decode is token-identical to the per-request path.

    Both runners share one tiny gpt-oss (same random weights).  The
    reference runner uses the ``disable_radix_cache`` per-request path
    pinned by test_windowed_kv_cache.py; the radix runner replays the
    same requests through pool sync, prefix hits, and prefix recomputes.
    """

    DECODE_STEPS = 6

    @classmethod
    def setUpClass(cls):
        mx.random.seed(7)
        cls.model = _tiny_gpt_oss_model()
        patch_model_attention(cls.model)

    def setUp(self):
        self.reference = _stub_runner(self.model, disable_radix_cache=True)
        self.radix = _stub_runner(self.model, disable_radix_cache=False)

    def _greedy(self, runner, rid, full_ids, new_ids, prefix_slots, new_slots):
        tokens = [
            runner.prefill(
                req_id=rid,
                new_token_ids=list(new_ids),
                full_token_ids=list(full_ids),
                prefix_slot_ids=list(prefix_slots),
                new_slot_ids=list(new_slots),
                req_pool_idx=0,
            )
        ]
        for _ in range(self.DECODE_STEPS):
            tokens.extend(runner.decode_batch([rid]))
        return tokens

    def _reference_stream(self, prompt):
        tokens = self._greedy(
            self.reference, "ref", prompt, prompt, prefix_slots=(), new_slots=()
        )
        self.reference.remove_request("ref")
        return tokens

    def _seed(self, prompt, slots):
        self._greedy(self.radix, "seed", prompt, prompt, (), slots)
        self.radix.remove_request("seed")

    def _assert_windowed_bounded(self, rid):
        cache = self.radix._req_caches[rid]
        for layer_idx in self.radix._cache_layout.swa_attention_layer_indices:
            entry = cache[layer_idx]
            self.assertIsInstance(entry, WindowedAttentionKVCache)
            self.assertLessEqual(entry.get_kv()[0].shape[2], 2 * TINY_WINDOW)

    def test_cold_prefill_matches_reference(self):
        prompt = [(i * 7 + 3) % 128 for i in range(20)]
        want = self._reference_stream(prompt)
        got = self._greedy(
            self.radix, "cold", prompt, prompt, (), range(1, len(prompt) + 1)
        )
        self.assertEqual(got, want)
        self._assert_windowed_bounded("cold")

    def test_partial_prefix_hit_recomputes_exactly(self):
        # Prefix (20) is well past the window (8): the hit must recompute the
        # whole prefix rather than gather it, a chunked extend continues on
        # top, and the stream must match one cold reference over the same
        # tokens with every cache left at the unclamped absolute position.
        prefix = [(i * 7 + 3) % 128 for i in range(20)]
        chunk_a, chunk_b = [9, 42, 77, 5], [11, 13, 17]
        prefix_slots = list(range(1, len(prefix) + 1))
        self._seed(prefix, prefix_slots)
        want = self._reference_stream(prefix + chunk_a + chunk_b)

        gathers = []
        self.radix._cache_with_pool_backed_attention = lambda slots, n: gathers.append(
            n
        )
        self.radix.prefill(
            req_id="hit",
            new_token_ids=chunk_a,
            full_token_ids=prefix + chunk_a,
            prefix_slot_ids=prefix_slots,
            new_slot_ids=list(range(30, 34)),
            req_pool_idx=0,
        )
        self.assertEqual(gathers, [], "SWA prefix hits must recompute, not gather")

        got = [self.radix.extend("hit", chunk_b, list(range(34, 37)))]
        for _ in range(self.DECODE_STEPS):
            got.extend(self.radix.decode_batch(["hit"]))
        self.assertEqual(got, want)
        self._assert_windowed_bounded("hit")
        expected = len(prefix + chunk_a + chunk_b) + self.DECODE_STEPS
        for layer_idx in range(4):
            self.assertEqual(self.radix._req_caches["hit"][layer_idx].offset, expected)

    def test_full_prefix_hit_without_new_tokens(self):
        # An exact hit leaves no extend tokens: the prefix rebuild supplies
        # run tokens ending on the last prefix token, whose logits predict
        # the next token.
        prompt = [(i * 5 + 11) % 128 for i in range(20)]
        prefix_slots = list(range(1, len(prompt) + 1))
        self._seed(prompt, prefix_slots)
        want = self._reference_stream(prompt)
        got = self._greedy(self.radix, "exact", prompt, [], prefix_slots, ())
        self.assertEqual(got, want)

    def test_fused_aot_kernel_serves_full_layers_by_full_pool_index(self):
        # The fused RoPE+pool-scatter kernel must skip sliding-window layers
        # and address the pool by the full-attention index, not the cache one.
        sliding_wrapper = self.model.model.layers[0].self_attn
        full_wrapper = self.model.model.layers[1].self_attn

        recorded = []
        original = MLXAttentionWrapper._rope_custom_aot

        def _recording_rope(queries, keys, values, positions, pool_idx, rope_ctx):
            recorded.append(pool_idx)
            return queries, keys

        MLXAttentionWrapper._rope_custom_aot = staticmethod(_recording_rope)
        try:
            win = WindowedAttentionKVCache(TINY_WINDOW)
            contig = ContiguousAttentionKVCache(
                n_kv_heads=2, head_dim=16, max_seq_len=32, dtype=mx.float32
            )
            ctx = BatchedDecodeContext(
                batch_size=1,
                seq_lens=[0],
                attention_layer_caches=[[win], [contig]],
                attention_pool_index_by_layer={0: 0, 1: 1},
                full_kv_pool_index_by_layer={1: 0},
                aot=MlxAOTKernelContext(
                    rope=MlxAOTRoPEContext(kernel=MlxAOTRoPEKernel(), kv_pool=None)
                ),
            )
            x = mx.random.normal((1, 1, 64))
            mx.eval(sliding_wrapper._batched_decode(x, ctx))
            self.assertEqual(recorded, [], "SWA layer must not hit the fused kernel")
            mx.eval(full_wrapper._batched_decode(x, ctx))
            # Cache index for layer 1 is 1; its full-pool index is 0.
            self.assertEqual(recorded, [0], "full layer needs the full-pool index")
        finally:
            MLXAttentionWrapper._rope_custom_aot = original


if __name__ == "__main__":
    unittest.main()
