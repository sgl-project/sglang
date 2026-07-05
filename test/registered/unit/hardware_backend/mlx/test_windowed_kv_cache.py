"""Unit tests for the MLX windowed per-request attention KV cache.

``WindowedAttentionKVCache`` keeps only the trailing ``window`` tokens of a
sliding-window layer instead of the full sequence (PR #30050 stored full
history and windowed at read time; this is the promised follow-up for the
``disable_radix_cache`` per-request path).  Correctness rests on three
claims, each pinned here against the full-history path:

1. Cache level: the arrays returned by ``update_and_fetch``/``get_kv`` are
   bit-identical to the trailing slice of a ``ContiguousAttentionKVCache``
   fed the same K/V, across chunk patterns, compaction boundaries, and
   buffer shrink after oversized prefill chunks; ``make_mask`` width always
   equals the returned key length, and the clamped mask equals the trailing
   columns of the full-history banded mask.
2. Model level (container path): chunked prefill and greedy decode through
   a tiny gpt-oss produce the same tokens with windowed caches on sliding
   layers as with full-history caches.
3. Wrapper level (batched decode path): ``MLXAttentionWrapper`` decode over
   windowed caches matches the full-KV banded-mask reference for multiple
   consecutive steps, riding through the 2*window compaction boundary.

The reference comparisons deliberately mirror tensor shapes where the
tolerance is tight (see TestBatchedDecodeSlidingWindow in
test_sliding_window_attention.py for why): MLX kernels pick different code
paths per input shape.  Where key lengths necessarily differ (windowed vs
full SDPA), assertions rely on greedy-token equality plus a loose numeric
bound instead.
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
    from mlx_lm.models.base import create_causal_mask

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        BatchedDecodeContext,
        ContiguousAttentionKVCache,
        MLXAttentionWrapper,
        WindowedAttentionKVCache,
        find_attention_layers,
        get_layer_window_sizes,
        make_attention_mask,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache.layout import MlxModelCacheLayout

TINY_WINDOW = 8


def _tiny_gpt_oss_model():
    """Randomly initialized 4-layer gpt_oss with alternating sliding/full layers.

    Mirrors test_sliding_window_attention.py's builder (kept local: the
    registered unit-test directory is not an importable package).
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


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestWindowedCacheEquivalence(CustomTestCase):
    """Cache-level equivalence against the full-history trailing slice.

    Both caches receive identical K/V arrays and only copy them, so the
    comparisons are exact (``mx.array_equal``), not tolerance-based.
    """

    W = 4
    H = 2
    D = 8

    def _run_pattern(self, chunk_sizes, decode_steps):
        full = ContiguousAttentionKVCache(max_seq_len=128)
        win = WindowedAttentionKVCache(self.W)
        for S in chunk_sizes:
            k = mx.random.normal((1, self.H, S, self.D))
            v = mx.random.normal((1, self.H, S, self.D))
            mask = win.make_mask(S, window_size=self.W)
            fk, fv = full.update_and_fetch(k, v)
            wk, wv = win.update_and_fetch(k, v)
            self.assertTrue(
                mx.array_equal(wk, fk[:, :, -wk.shape[2] :, :]).item(),
                f"K mismatch after chunk S={S} in {chunk_sizes}",
            )
            self.assertTrue(
                mx.array_equal(wv, fv[:, :, -wv.shape[2] :, :]).item(),
                f"V mismatch after chunk S={S} in {chunk_sizes}",
            )
            # make_mask runs before update_and_fetch in a forward pass; its
            # width must match the keys update_and_fetch then returns.
            self.assertEqual(mask.shape[-1], wk.shape[2])
            self.assertEqual(win.offset, full.offset)
        for i in range(decode_steps):
            k = mx.random.normal((1, self.H, 1, self.D))
            v = mx.random.normal((1, self.H, 1, self.D))
            full.write_token(k, v)
            win.write_token(k, v)
            fk, _ = full.get_kv()
            wk, wv = win.get_kv()
            trailing = min(win.offset, self.W)
            self.assertTrue(
                mx.array_equal(wk[:, :, -trailing:, :], fk[:, :, -trailing:, :]).item(),
                f"decode K mismatch at step {i} for {chunk_sizes}",
            )
            self.assertLessEqual(wk.shape[2], 2 * self.W)
            self.assertEqual(win.offset, full.offset)
        return win

    def test_chunk_patterns_match_full_trailing_slice(self):
        mx.random.seed(0)
        for pattern in [
            (3,),  # stays inside the window
            (4,),  # lands exactly on the window
            (5,),  # first chunk already crosses the window
            (6, 3),  # second chunk forces prefix normalisation
            (2, 2, 2, 2),  # repeated small chunks
            (1, 1, 1),  # degenerate single-token chunks
            (10, 1, 7),  # chunk larger than 2*window, then mixed
        ]:
            self._run_pattern(pattern, decode_steps=3 * self.W)

    def test_decode_capacity_stays_bounded(self):
        mx.random.seed(1)
        win = self._run_pattern((10,), decode_steps=4 * self.W)
        # Steady-state decode buffer is exactly 2*window despite the
        # window+10 buffer the oversized prefill chunk allocated.
        self.assertEqual(win.keys.shape[2], 2 * self.W)

    def test_clamped_mask_equals_trailing_columns_of_full_mask(self):
        # Full-history banded mask over 9 keys for rows at abs 6..8 vs the
        # windowed mask over the kept 4 + 3 keys: identical after dropping
        # the columns the window can never reach (band arithmetic is
        # shift-invariant).
        full_mask = make_attention_mask(3, 6, window_size=self.W)
        win_mask = make_attention_mask(3, min(6, self.W), window_size=self.W)
        self.assertEqual(win_mask.shape[-1], self.W + 3)
        self.assertTrue(
            mx.array_equal(full_mask[..., -win_mask.shape[-1] :], win_mask).item()
        )

    def test_full_context_mask_raises_after_drop(self):
        win = WindowedAttentionKVCache(self.W)
        for S in (6, 3):
            win.update_and_fetch(
                mx.random.normal((1, self.H, S, self.D)),
                mx.random.normal((1, self.H, S, self.D)),
            )
        with self.assertRaises(RuntimeError):
            win.make_mask(2)

    def test_full_context_mask_raises_after_single_oversized_chunk(self):
        # Nothing has been dropped yet (offset == local), but the next
        # update_and_fetch normalises the prefix to the window, so a
        # full-context mask is already a promise the cache cannot keep.
        win = WindowedAttentionKVCache(self.W)
        win.update_and_fetch(
            mx.random.normal((1, self.H, 10, self.D)),
            mx.random.normal((1, self.H, 10, self.D)),
        )
        with self.assertRaises(RuntimeError):
            win.make_mask(2)

    def test_full_context_mask_allowed_before_drop(self):
        win = WindowedAttentionKVCache(self.W)
        win.update_and_fetch(
            mx.random.normal((1, self.H, 3, self.D)),
            mx.random.normal((1, self.H, 3, self.D)),
        )
        self.assertEqual(win.make_mask(2), "causal")

    def test_reset_keeps_buffers_and_replays(self):
        win = WindowedAttentionKVCache(self.W)
        win.update_and_fetch(
            mx.random.normal((1, self.H, 6, self.D)),
            mx.random.normal((1, self.H, 6, self.D)),
        )
        win.reset()
        self.assertEqual(win.offset, 0)
        self.assertIsNotNone(win.keys)
        k = mx.random.normal((1, self.H, 2, self.D))
        out, _ = win.update_and_fetch(k, mx.random.normal((1, self.H, 2, self.D)))
        self.assertEqual(out.shape[2], 2)
        self.assertTrue(mx.array_equal(out, k).item())

    def test_window_validation(self):
        with self.assertRaises(ValueError):
            WindowedAttentionKVCache(0)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestWindowedModelForward(CustomTestCase):
    """Container-path (prefill/extend) equivalence on a tiny gpt-oss.

    Windowed caches on sliding layers vs full-history caches everywhere.
    SDPA key lengths necessarily differ once the window drops history, so
    greedy-token equality is the primary assertion, with a loose numeric
    bound on the final-chunk logits as a sanity check.
    """

    PROMPT_LEN = 20  # 2.5x TINY_WINDOW

    def _cache_lists(self, model):
        window_map = get_layer_window_sizes(model)
        self.assertEqual(
            [window_map[i] for i in range(4)],
            [TINY_WINDOW, None, TINY_WINDOW, None],
        )
        windowed = [
            (
                WindowedAttentionKVCache(window_map[i])
                if window_map.get(i) is not None
                else ContiguousAttentionKVCache(max_seq_len=64)
            )
            for i in range(4)
        ]
        full = [ContiguousAttentionKVCache(max_seq_len=64) for _ in range(4)]
        return windowed, full

    def test_chunked_prefill_matches_full_history(self):
        mx.random.seed(0)
        model = _tiny_gpt_oss_model()
        windowed, full = self._cache_lists(model)
        ids = (mx.arange(self.PROMPT_LEN) * 7 + 3) % 128
        split = 12  # second chunk starts beyond the window
        logits = {}
        for name, cache in (("windowed", windowed), ("full", full)):
            first = model(ids[None, :split], cache=cache)
            second = model(ids[None, split:], cache=cache)
            mx.eval(first, second)
            logits[name] = second
        diff = mx.abs(logits["windowed"] - logits["full"]).max().item()
        self.assertLess(diff, 1e-3, f"final-chunk logits diverge by {diff}")
        self.assertEqual(
            mx.argmax(logits["windowed"][:, -1, :], axis=-1).item(),
            mx.argmax(logits["full"][:, -1, :], axis=-1).item(),
        )

    def test_greedy_decode_matches_full_history(self):
        mx.random.seed(0)
        model = _tiny_gpt_oss_model()
        windowed, full = self._cache_lists(model)
        ids = (mx.arange(self.PROMPT_LEN) * 5 + 11) % 128
        tokens = {}
        for name, cache in (("windowed", windowed), ("full", full)):
            out = model(ids[None], cache=cache)
            token = mx.argmax(out[:, -1, :], axis=-1)
            seq = [token.item()]
            for _ in range(2 * TINY_WINDOW):
                out = model(token[None], cache=cache)
                token = mx.argmax(out[:, -1, :], axis=-1)
                seq.append(token.item())
            tokens[name] = seq
        self.assertEqual(
            tokens["windowed"],
            tokens["full"],
            "greedy decode diverges between windowed and full-history caches",
        )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestWindowedBatchedDecode(CustomTestCase):
    """Wrapper batched-decode over windowed caches vs the full-KV reference.

    Shadow ``ContiguousAttentionKVCache`` instances receive the same K/V so
    the reference can attend over full history with mlx_lm's banded mask.
    Shapes mirror the wrapper exactly (batched projections + RoPE, one SDPA
    per request), keeping the tolerance at 1e-5 — see
    test_sliding_window_attention.py for the shape-sensitivity rationale.
    """

    HIDDEN = 64
    N_KV_HEADS = 2
    HEAD_DIM = 16

    def _prefill_both(self, attn, x, window):
        win = WindowedAttentionKVCache(window)
        shadow = ContiguousAttentionKVCache(
            n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=64,
            dtype=mx.float32,
        )
        prefix = x[:, :-1, :]
        for cache in (win, shadow):
            attn(
                prefix,
                make_attention_mask(prefix.shape[1], 0, window_size=window),
                cache=cache,
            )
        return win, shadow

    def _project_step(self, attn, x_step, offsets):
        B, D = x_step.shape[0], self.HEAD_DIM
        q = attn.q_proj(x_step).reshape(B, 1, -1, D).transpose(0, 2, 1, 3)
        k = attn.k_proj(x_step).reshape(B, 1, -1, D).transpose(0, 2, 1, 3)
        v = attn.v_proj(x_step).reshape(B, 1, -1, D).transpose(0, 2, 1, 3)
        off = mx.array(offsets, dtype=mx.int32)
        return attn.rope(q, offset=off), attn.rope(k, offset=off), v

    def _reference_step(self, attn, x_step, shadows, offsets, window):
        """Full-KV banded reference; also writes the new token into shadows."""
        q, k_new, v_new = self._project_step(attn, x_step, offsets)
        outs = []
        for i, shadow in enumerate(shadows):
            k_prefix, v_prefix = shadow.get_kv()
            k = mx.concatenate([k_prefix, k_new[i : i + 1]], axis=2)
            v = mx.concatenate([v_prefix, v_new[i : i + 1]], axis=2)
            mask = create_causal_mask(1, offsets[i], window_size=window)
            outs.append(
                mx.fast.scaled_dot_product_attention(
                    q[i : i + 1],
                    k,
                    v,
                    scale=attn.sm_scale,
                    mask=mask,
                    sinks=attn.sinks,
                )
            )
            shadow.write_token(k_new[i : i + 1], v_new[i : i + 1])
        out = mx.concatenate(outs, axis=0)
        out = out.transpose(0, 2, 1, 3).reshape(len(shadows), 1, -1)
        return attn.o_proj(out)

    def test_multi_step_decode_rides_compaction_boundary(self):
        mx.random.seed(0)
        model = _tiny_gpt_oss_model()
        attn = model.model.layers[0].self_attn
        window = TINY_WINDOW
        lens = [12, 5]  # one past the window, one inside
        xs = [mx.random.normal((1, L, self.HIDDEN)) for L in lens]
        pairs = [self._prefill_both(attn, x, window) for x in xs]
        wins = [p[0] for p in pairs]
        shadows = [p[1] for p in pairs]
        wrapper = MLXAttentionWrapper(attn, layer_idx=0, window_size=window)

        # 12 steps push the longer request's local buffer across the
        # 2*window=16 compaction boundary mid-run.
        for step in range(12):
            offsets = [w.offset for w in wins]
            x_step = mx.random.normal((len(lens), 1, self.HIDDEN))
            ref = self._reference_step(attn, x_step, shadows, offsets, window)
            ctx = BatchedDecodeContext(
                batch_size=len(lens),
                seq_lens=offsets,
                attention_layer_caches=[wins],
            )
            got = wrapper._batched_decode(x_step, ctx)
            mx.eval(got, ref)
            for i in range(len(lens)):
                diff = mx.abs(got[i : i + 1] - ref[i : i + 1]).max().item()
                self.assertLess(
                    diff,
                    1e-5,
                    f"step {step} request {i} diverges by {diff}",
                )
            for win, shadow in zip(wins, shadows):
                self.assertEqual(win.offset, shadow.offset)

    def test_chained_graph_build_across_compaction(self):
        """Two in-flight lazy graphs spanning a compaction stay correct.

        The MLX overlap scheduler builds decode step N+1's graph before
        step N materialises (decode_batch_start_chained).  Compaction
        allocates a fresh buffer (never mutates in place), so step N's
        returned views must stay valid.  Steps are built in pairs and
        evaluated together; the oversized-prefill shrink fires inside the
        first pair and the steady-state 2*window rebuild inside a later
        pair (start local 14 -> shrink to 9 -> hits 16 at step 8).
        """
        mx.random.seed(1)
        model = _tiny_gpt_oss_model()
        attn = model.model.layers[0].self_attn
        window = TINY_WINDOW
        lens = [15, 6]
        xs = [mx.random.normal((1, L, self.HIDDEN)) for L in lens]
        pairs = [self._prefill_both(attn, x, window) for x in xs]
        wins = [p[0] for p in pairs]
        shadows = [p[1] for p in pairs]
        wrapper = MLXAttentionWrapper(attn, layer_idx=0, window_size=window)

        def build_step():
            offsets = [w.offset for w in wins]
            x_step = mx.random.normal((len(lens), 1, self.HIDDEN))
            ref = self._reference_step(attn, x_step, shadows, offsets, window)
            ctx = BatchedDecodeContext(
                batch_size=len(lens),
                seq_lens=offsets,
                attention_layer_caches=[wins],
            )
            return wrapper._batched_decode(x_step, ctx), ref

        for pair in range(6):
            got_a, ref_a = build_step()
            got_b, ref_b = build_step()  # built before got_a materialises
            mx.eval(got_a, got_b)
            for label, got, ref in (("a", got_a, ref_a), ("b", got_b, ref_b)):
                for i in range(len(lens)):
                    diff = mx.abs(got[i : i + 1] - ref[i : i + 1]).max().item()
                    self.assertLess(
                        diff,
                        1e-5,
                        f"pair {pair}{label} request {i} diverges by {diff}",
                    )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestModelRunnerCacheWiring(CustomTestCase):
    """_new_native_cache/_acquire_cache wiring without loading a model."""

    def _stub_runner(self, window_map):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        model = _tiny_gpt_oss_model()
        layers, attrs = find_attention_layers(model)
        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            layers, attrs
        )
        runner._layer_window_sizes = window_map
        runner._max_seq_len = 4096
        runner._cache_pool = []
        return runner

    def test_windowed_for_sliding_layers_on_per_request_path(self):
        model = _tiny_gpt_oss_model()
        runner = self._stub_runner(get_layer_window_sizes(model))
        cache = runner._new_native_cache()
        self.assertIsInstance(cache[0], WindowedAttentionKVCache)
        self.assertIsInstance(cache[1], ContiguousAttentionKVCache)
        self.assertIsInstance(cache[2], WindowedAttentionKVCache)
        self.assertIsInstance(cache[3], ContiguousAttentionKVCache)
        self.assertEqual(cache[0].window, TINY_WINDOW)

    def test_all_contiguous_when_window_map_empty(self):
        # Radix/pool path: MlxModelRunner.__init__ leaves the map empty, so
        # pool-backed conversions keep absolute slicing intact.
        runner = self._stub_runner({})
        cache = runner._new_native_cache()
        for c in cache:
            self.assertIsInstance(c, ContiguousAttentionKVCache)

    def test_acquire_cache_resets_windowed_state(self):
        model = _tiny_gpt_oss_model()
        runner = self._stub_runner(get_layer_window_sizes(model))
        cache = runner._new_native_cache()
        k = mx.random.normal((1, 2, 10, 16))
        cache[0].update_and_fetch(k, k)
        cache[1].update_and_fetch(
            k.transpose(0, 1, 2, 3), k
        )  # contiguous accepts (1, H, S, D) too
        runner._release_cache(cache)
        reused = runner._acquire_cache()
        self.assertIs(reused, cache)
        for c in reused:
            self.assertEqual(c.offset, 0)
        self.assertEqual(reused[0]._local, 0)
        out, _ = reused[0].update_and_fetch(k, k)
        self.assertEqual(out.shape[2], 10)


if __name__ == "__main__":
    unittest.main()
