"""Unit tests for sliding-window layers on the MLX radix/pool KV path.

Phase 1 of the SWA pool split (mirroring the CUDA path's ``SWAKVPool``
layer-type separation): the shared ``MlxAttentionKVPool`` stores
full-attention layers only, sliding-window layers keep window-bounded
per-request ``WindowedAttentionKVCache`` storage on the radix path too,
and a radix prefix hit on an SWA model recomputes the whole prefix (no
cross-request SWA KV exists in Phase 1, and recomputing only a trailing
band is inexact because window receptive fields chain backwards through
layers).  Scheduler bookkeeping stays in the unclamped coordinates, so
these tests drive ``MlxModelRunner`` directly with hand-built slot ids,
mirroring how the tp_worker calls it.

Correctness claims pinned here:

1. Layout level: the full/SWA partition and the dense full-pool index
   derive from the container window map, and default to the old
   all-full behavior without one.
2. Pool level: only full-attention layers get pool buffers, and
   ``_sync_new_kv_to_pool`` writes exactly those layers.
3. Prefill level: SWA-model prefix hits recompute instead of gathering
   pool history; radix-path greedy decoding after cold, partial-hit,
   and full-hit prefills is token-identical to the per-request
   reference path.
4. Decode level: the fused AOT RoPE+scatter kernel is routed to
   full-attention layers only, addressed by the full-pool index.
"""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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


def _layout_for(model, with_windows=True):
    layers, attrs = find_attention_layers(model)
    window_map = get_layer_window_sizes(model) if with_windows else None
    return MlxModelCacheLayout.from_attention_discovery(
        layers, attrs, layer_window_sizes=window_map
    )


def _stub_runner(model, disable_radix_cache, pool_size=64):
    """Surgically build a runner around an already-loaded tiny model."""
    runner = MlxModelRunner.__new__(MlxModelRunner)
    runner.model = model
    runner.disable_radix_cache = disable_radix_cache
    runner._layer_window_sizes = get_layer_window_sizes(model)
    runner._cache_layout = _layout_for(model)
    runner._max_seq_len = 64
    runner._cache_pool = []
    runner._req_caches = {}
    runner._req_token_ids = {}
    runner._req_pool_idx = {}
    runner._req_synced_offset = {}
    runner._req_to_token_pool = None
    runner._attention_kv_pool = None
    runner._decode_step_ct = 0
    runner._clear_steps = 0
    runner._aot_kernels = MlxAOTKernelSet()
    runner._pool_size = pool_size
    if not disable_radix_cache:
        n_kv_heads, head_dim, dtype = runner._get_attn_config()
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=pool_size + 1,
            num_layers=runner._cache_layout.num_full_attention_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
    return runner


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSwaLayoutPartition(CustomTestCase):
    def test_partition_from_container_window_map(self):
        layout = _layout_for(_tiny_gpt_oss_model())
        self.assertEqual(layout.attention_layer_indices, (0, 1, 2, 3))
        self.assertEqual(layout.swa_attention_layer_indices, (0, 2))
        self.assertEqual(layout.full_attention_layer_indices, (1, 3))
        self.assertEqual(layout.num_full_attention_layers, 2)
        self.assertEqual(layout.full_kv_pool_index_by_layer, {1: 0, 3: 1})
        self.assertEqual(layout.max_swa_window, TINY_WINDOW)
        self.assertEqual(layout.full_kv_pool_index(3), 1)
        with self.assertRaises(KeyError):
            layout.full_kv_pool_index(0)

    def test_no_window_map_keeps_all_full_partition(self):
        layout = _layout_for(_tiny_gpt_oss_model(), with_windows=False)
        self.assertEqual(layout.swa_attention_layer_indices, ())
        self.assertEqual(
            layout.full_attention_layer_indices, layout.attention_layer_indices
        )
        self.assertEqual(
            layout.full_kv_pool_index_by_layer, layout.attention_pool_index_by_layer
        )
        self.assertIsNone(layout.max_swa_window)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSwaRunnerPoolContract(CustomTestCase):
    def test_attn_config_resolves_for_windowed_layers(self):
        model = _tiny_gpt_oss_model()
        runner = _stub_runner(model, disable_radix_cache=True)
        n_kv_heads, head_dim, _ = runner._get_attn_config()
        self.assertEqual(n_kv_heads, 2)
        self.assertEqual(head_dim, 16)

    def test_sliding_flag_without_window_map_still_rejected(self):
        model = _tiny_gpt_oss_model()
        patch_model_attention(model)
        model.model.layers[0].self_attn._inner.is_sliding = True
        runner = _stub_runner(model, disable_radix_cache=True)
        # With the container window map the flagged layer is bounded: fine.
        runner._get_attn_config()
        # Without a resolvable window the layer cannot be bounded: reject.
        runner._layer_window_sizes = {}
        with self.assertRaises(NotImplementedError):
            runner._get_attn_config()

    def test_pool_covers_full_attention_layers_only(self):
        model = _tiny_gpt_oss_model()
        runner = _stub_runner(model, disable_radix_cache=False)
        self.assertEqual(len(runner._attention_kv_pool.k_buffer), 2)
        self.assertEqual(runner._attention_kv_pool.num_layers, 2)

    def test_init_cache_pools_builds_full_layer_pool(self):
        model = _tiny_gpt_oss_model()
        runner = _stub_runner(model, disable_radix_cache=True)  # no pool yet
        runner.disable_radix_cache = False
        runner._pool_size = 16
        runner.init_cache_pools(None)
        self.assertEqual(runner._attention_kv_pool.num_layers, 2)
        self.assertEqual(runner._attention_kv_pool.pool_size, 17)

    def test_native_cache_windowed_on_radix_path(self):
        model = _tiny_gpt_oss_model()
        runner = _stub_runner(model, disable_radix_cache=False)
        cache = runner._new_native_cache()
        self.assertIsInstance(cache[0], WindowedAttentionKVCache)
        self.assertIsInstance(cache[1], ContiguousAttentionKVCache)
        self.assertIsInstance(cache[2], WindowedAttentionKVCache)
        self.assertIsInstance(cache[3], ContiguousAttentionKVCache)

    def test_windowed_cache_offset_seeding(self):
        # Pool-path prefix restoration primitive: an empty windowed cache
        # can start at an absolute position (Phase 2 seeds it from a shared
        # SWA pool gather).
        win = WindowedAttentionKVCache(TINY_WINDOW, offset=12)
        self.assertEqual(win.offset, 12)
        self.assertEqual(win._local, 0)
        mask = win.make_mask(4, window_size=TINY_WINDOW)
        self.assertEqual(mask.shape[-1], 4)  # no kept prefix yet
        k = mx.random.normal((1, 2, 4, 16))
        out, _ = win.update_and_fetch(k, k)
        self.assertEqual(out.shape[2], 4)
        self.assertEqual(win.offset, 16)
        with self.assertRaises(ValueError):
            WindowedAttentionKVCache(TINY_WINDOW, offset=-1)

    def test_swa_prefix_hit_recomputes_instead_of_pool_gather(self):
        mx.random.seed(5)
        model = _tiny_gpt_oss_model()
        patch_model_attention(model)
        runner = _stub_runner(model, disable_radix_cache=False)
        prompt = [(i * 7 + 3) % 128 for i in range(20)]
        slots = list(range(1, 21))
        runner.prefill(
            req_id="seed",
            new_token_ids=prompt,
            full_token_ids=prompt,
            prefix_slot_ids=[],
            new_slot_ids=slots,
            req_pool_idx=0,
        )
        runner.remove_request("seed")

        calls = []
        original = runner._cache_with_pool_backed_attention

        def spy(prefix_slot_ids, prefix_len):
            calls.append(prefix_len)
            return original(prefix_slot_ids, prefix_len)

        runner._cache_with_pool_backed_attention = spy
        runner.prefill(
            req_id="hit",
            new_token_ids=[42],
            full_token_ids=prompt + [42],
            prefix_slot_ids=slots,
            new_slot_ids=[25],
            req_pool_idx=1,
        )
        self.assertEqual(calls, [], "SWA prefix hits must recompute, not gather")
        # The whole sequence re-ran: every layer cache is at the full length.
        cache = runner._req_caches["hit"]
        for layer_idx in (0, 1, 2, 3):
            self.assertEqual(cache[layer_idx].offset, len(prompt) + 1)

    def test_sync_writes_full_layers_only(self):
        model = _tiny_gpt_oss_model()
        runner = _stub_runner(model, disable_radix_cache=False)
        n_kv_heads, head_dim = 2, 16
        cache = runner._new_native_cache()
        seq_len = 5
        per_layer_k = {}
        for layer_idx in (0, 1, 2, 3):
            k = mx.full((1, n_kv_heads, seq_len, head_dim), float(layer_idx + 1))
            v = -k
            cache[layer_idx].update_and_fetch(k, v)
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
class TestSwaRadixPrefillEquivalence(CustomTestCase):
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
        token = runner.prefill(
            req_id=rid,
            new_token_ids=list(new_ids),
            full_token_ids=list(full_ids),
            prefix_slot_ids=list(prefix_slots),
            new_slot_ids=list(new_slots),
            req_pool_idx=0,
        )
        tokens = [token]
        for _ in range(self.DECODE_STEPS):
            tokens.extend(runner.decode_batch([rid]))
        return tokens

    def _reference_stream(self, prompt):
        tokens = self._greedy(
            self.reference, "ref", prompt, prompt, prefix_slots=(), new_slots=()
        )
        self.reference.remove_request("ref")
        return tokens

    def _assert_windowed_bounded(self, runner, rid):
        cache = runner._req_caches[rid]
        for layer_idx in runner._cache_layout.swa_attention_layer_indices:
            entry = cache[layer_idx]
            self.assertIsInstance(entry, WindowedAttentionKVCache)
            self.assertLessEqual(entry._local, 2 * TINY_WINDOW)
            self.assertLessEqual(entry.get_kv()[0].shape[2], 2 * TINY_WINDOW)

    def test_cold_prefill_matches_reference(self):
        prompt = [(i * 7 + 3) % 128 for i in range(20)]
        want = self._reference_stream(prompt)
        got = self._greedy(
            self.radix,
            "cold",
            prompt,
            prompt,
            prefix_slots=(),
            new_slots=range(1, len(prompt) + 1),
        )
        self.assertEqual(got, want)
        self._assert_windowed_bounded(self.radix, "cold")

    def test_partial_prefix_hit_recomputes_exactly(self):
        # Prefix (20) is well past the window (8): the hit recomputes the
        # whole prefix plus the suffix and must match the reference stream.
        prefix = [(i * 7 + 3) % 128 for i in range(20)]
        suffix = [9, 42, 77, 5]
        prefix_slots = list(range(1, len(prefix) + 1))
        self._greedy(
            self.radix, "seed", prefix, prefix, prefix_slots=(), new_slots=prefix_slots
        )
        self.radix.remove_request("seed")

        want = self._reference_stream(prefix + suffix)
        got = self._greedy(
            self.radix,
            "hit",
            prefix + suffix,
            suffix,
            prefix_slots=prefix_slots,
            new_slots=range(30, 30 + len(suffix)),
        )
        self.assertEqual(got, want)
        self._assert_windowed_bounded(self.radix, "hit")
        # The windowed caches were rebuilt at the right absolute position.
        cache = self.radix._req_caches["hit"]
        expected_offset = len(prefix) + len(suffix) + self.DECODE_STEPS
        for layer_idx in (0, 1, 2, 3):
            self.assertEqual(cache[layer_idx].offset, expected_offset)

    def test_full_prefix_hit_scheduler_shape(self):
        # Production full hits arrive with the last token left as extend
        # input (the scheduler truncates prefix_indices by one).
        prompt = [(i * 5 + 11) % 128 for i in range(20)]
        prefix_slots = list(range(1, len(prompt) + 1))
        self._greedy(
            self.radix, "seed", prompt, prompt, prefix_slots=(), new_slots=prefix_slots
        )
        self.radix.remove_request("seed")

        want = self._reference_stream(prompt)
        got = self._greedy(
            self.radix,
            "fullhit",
            prompt,
            prompt[-1:],
            prefix_slots=prefix_slots[:-1],
            new_slots=prefix_slots[-1:],
        )
        self.assertEqual(got, want)
        self._assert_windowed_bounded(self.radix, "fullhit")

    def test_full_prefix_hit_without_new_tokens(self):
        # The runner also serves exact hits with no extend tokens: the
        # window rebuild supplies run tokens ending on the last prefix
        # token, whose logits predict the next token.
        prompt = [(i * 5 + 11) % 128 for i in range(20)]
        prefix_slots = list(range(1, len(prompt) + 1))
        self._greedy(
            self.radix, "seed", prompt, prompt, prefix_slots=(), new_slots=prefix_slots
        )
        self.radix.remove_request("seed")

        want = self._reference_stream(prompt)
        got = self._greedy(
            self.radix,
            "exact",
            prompt,
            [],
            prefix_slots=prefix_slots,
            new_slots=(),
        )
        self.assertEqual(got, want)

    def test_short_prefix_hit_rebuilds_everything(self):
        # Prefix (6) inside the window (8): nothing is trusted, the whole
        # prompt re-runs, and no pool-backed gather happens.
        prefix = [(i * 3 + 1) % 128 for i in range(6)]
        suffix = [64, 33]
        prefix_slots = list(range(40, 40 + len(prefix)))
        self._greedy(
            self.radix, "seed", prefix, prefix, prefix_slots=(), new_slots=prefix_slots
        )
        self.radix.remove_request("seed")

        want = self._reference_stream(prefix + suffix)
        got = self._greedy(
            self.radix,
            "short",
            prefix + suffix,
            suffix,
            prefix_slots=prefix_slots,
            new_slots=range(50, 50 + len(suffix)),
        )
        self.assertEqual(got, want)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSwaAotDecodeRouting(CustomTestCase):
    """The fused RoPE+pool-scatter kernel only serves full-attention layers."""

    def _decode_once(self, wrapper, cache, ctx):
        x = mx.random.normal((1, 1, 64))
        out = wrapper._batched_decode(x, ctx)
        mx.eval(out)

    def test_swa_layer_skips_fused_kernel_full_layer_uses_pool_index(self):
        mx.random.seed(3)
        model = _tiny_gpt_oss_model()
        patch_model_attention(model)
        sliding_wrapper = model.model.layers[0].self_attn
        full_wrapper = model.model.layers[1].self_attn

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
            aot_ctx = MlxAOTKernelContext(
                rope=MlxAOTRoPEContext(kernel=MlxAOTRoPEKernel(), kv_pool=None)
            )
            ctx = BatchedDecodeContext(
                batch_size=1,
                seq_lens=[0],
                attention_layer_caches=[[win], [contig]],
                attention_pool_index_by_layer={0: 0, 1: 1},
                full_kv_pool_index_by_layer={1: 0},
                aot=aot_ctx,
            )
            self._decode_once(sliding_wrapper, win, ctx)
            self.assertEqual(recorded, [], "SWA layer must not hit the fused kernel")
            self._decode_once(full_wrapper, contig, ctx)
            self.assertEqual(
                recorded, [0], "full layer must scatter via its full-pool index"
            )
        finally:
            MLXAttentionWrapper._rope_custom_aot = original


if __name__ == "__main__":
    unittest.main()
