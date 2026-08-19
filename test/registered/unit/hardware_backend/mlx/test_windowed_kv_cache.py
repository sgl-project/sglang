"""Unit tests for the MLX windowed per-request attention KV cache.

``WindowedAttentionKVCache`` keeps only the trailing ``window`` tokens of a
sliding-window layer.  Every level is pinned against the full-history path it
replaces: the cache arrays against a ``ContiguousAttentionKVCache`` trailing
slice, the container forward against full-history caches, and
``MLXAttentionWrapper`` batched decode against the same wrapper driven by
full-history caches.

Sliding-window layers use this storage on both KV paths; how it composes
with the shared pool and radix prefix hits is pinned in
test_swa_radix_pool.py.
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

WINDOW = 8
HIDDEN, N_KV_HEADS, HEAD_DIM = 64, 2, 16


def _dense_mask(mask, n_queries: int, offset: int):
    """Densify the cheap ``"causal"`` / ``None`` mask forms.

    ``make_attention_mask`` returns those instead of a materialised band
    whenever the window cannot bind, so their width lives in the key tensor
    rather than in the mask.  Densifying keeps width and content checkable
    for both forms.
    """
    if mask is None or isinstance(mask, str):
        return create_causal_mask(n_queries, offset)
    return mask


def _tiny_gpt_oss_model():
    """Random-weight 4-layer gpt_oss, alternating sliding/full layers."""
    return gpt_oss.Model(
        gpt_oss.ModelArgs(
            num_hidden_layers=4,
            num_local_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
            hidden_size=HIDDEN,
            intermediate_size=64,
            head_dim=HEAD_DIM,
            num_attention_heads=4,
            num_key_value_heads=N_KV_HEADS,
            sliding_window=WINDOW,
        )
    )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestWindowedCacheEquivalence(CustomTestCase):
    """Storage equivalence with the full-history trailing slice.

    Both caches receive identical K/V and only copy it, so the comparisons
    are exact (``mx.array_equal``), not tolerance-based.
    """

    W, H, D = 4, 2, 8

    def _kv(self, S):
        return mx.random.normal((1, self.H, S, self.D))

    def test_chunk_patterns_match_full_trailing_slice(self):
        """Windowed storage == trailing slice of full history, everywhere.

        Also pins what the forward pass depends on: the mask built before
        ``update_and_fetch`` is exactly as wide as the keys it then returns,
        the kept prefix still covers a full window, ``offset`` stays
        absolute, and the decode buffer stays bounded by ``2 * window``.
        """
        mx.random.seed(0)
        for chunks in [
            (3,),  # stays inside the window
            (4,),  # lands exactly on the window
            (5,),  # first chunk already crosses the window
            (6, 3),  # second chunk forces prefix normalisation
            (2, 2, 2, 2),  # repeated small chunks
            (1, 1, 1),  # degenerate single-token chunks
            (10, 1, 7),  # chunk larger than 2*window, then mixed
        ]:
            full = ContiguousAttentionKVCache(max_seq_len=128)
            win = WindowedAttentionKVCache(self.W)
            for S in chunks:
                k, v = self._kv(S), self._kv(S)
                mask = win.make_mask(S, window_size=self.W)  # runs before update
                fk, fv = full.update_and_fetch(k, v)
                wk, wv = win.update_and_fetch(k, v)
                at = f"chunk S={S} of {chunks}"
                self.assertTrue(mx.array_equal(wk, fk[:, :, -wk.shape[2] :, :]), at)
                self.assertTrue(mx.array_equal(wv, fv[:, :, -wv.shape[2] :, :]), at)
                # When the window cannot bind, make_mask returns the cheap
                # "causal"/None form whose width is implicit in the key
                # tensor; densify so the invariant stays checkable either way.
                dense = _dense_mask(mask, S, wk.shape[2] - S)
                self.assertEqual(dense.shape[-1], wk.shape[2], f"mask width, {at}")
                self.assertGreaterEqual(
                    wk.shape[2] - S, min(win.offset - S, self.W), f"prefix, {at}"
                )
                self.assertEqual(win.offset, full.offset, at)

            for step in range(4 * self.W):
                k, v = self._kv(1), self._kv(1)
                full.write_token(k, v)
                win.write_token(k, v)
                fk, _ = full.get_kv()
                wk, _ = win.get_kv()
                t = min(win.offset, self.W)
                at = f"decode step {step} after {chunks}"
                self.assertTrue(mx.array_equal(wk[:, :, -t:, :], fk[:, :, -t:, :]), at)
                self.assertEqual(win.offset, full.offset, at)
                self.assertLessEqual(win.keys.shape[2], 2 * self.W, at)

    def test_decode_reallocates_amortised_not_per_token(self):
        """Compaction must stay amortised O(1) on both write paths."""
        for write in ("write_token", "update_and_fetch"):
            win = WindowedAttentionKVCache(self.W)
            big = self._kv(5 * self.W)
            win.update_and_fetch(big, big)
            buf, reallocs = win.keys, 0
            for _ in range(10 * self.W):
                getattr(win, write)(self._kv(1), self._kv(1))
                if win.keys is not buf:
                    buf, reallocs = win.keys, reallocs + 1
            self.assertLessEqual(reallocs, 12, f"{write} reallocated {reallocs}x")

    def test_full_context_mask_raises_once_history_is_unservable(self):
        win = WindowedAttentionKVCache(self.W)
        win.update_and_fetch(self._kv(3), self._kv(3))
        self.assertEqual(win.make_mask(2), "causal")  # nothing dropped yet
        # One oversized chunk is enough: the next update normalises the
        # prefix to the window, so full context is already unservable.
        win.update_and_fetch(self._kv(10), self._kv(10))
        with self.assertRaises(RuntimeError):
            win.make_mask(2)

    def test_reset_keeps_buffers_and_replays(self):
        win = WindowedAttentionKVCache(self.W)
        win.update_and_fetch(self._kv(6), self._kv(6))
        win.reset()
        self.assertEqual(win.offset, 0)
        self.assertIsNotNone(win.keys)  # buffer kept for reuse
        k = self._kv(2)
        out, _ = win.update_and_fetch(k, k)
        self.assertEqual(out.shape[2], 2)  # no stale prefix survived
        self.assertTrue(mx.array_equal(out, k))


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestWindowedModelForward(CustomTestCase):
    """Container path: chunked prefill + greedy decode on a tiny gpt-oss."""

    def test_chunked_prefill_and_greedy_decode_match_full_history(self):
        mx.random.seed(0)
        model = _tiny_gpt_oss_model()
        windows = get_layer_window_sizes(model)
        self.assertEqual([windows[i] for i in range(4)], [WINDOW, None, WINDOW, None])
        ids = (mx.arange(20) * 7 + 3) % 128  # 2.5x window
        split = 12  # second chunk starts beyond the window

        tokens = {}
        for name in ("windowed", "full"):
            cache = [
                (
                    WindowedAttentionKVCache(windows[i])
                    if name == "windowed" and windows[i] is not None
                    else ContiguousAttentionKVCache(max_seq_len=64)
                )
                for i in range(4)
            ]
            model(ids[None, :split], cache=cache)
            out = model(ids[None, split:], cache=cache)
            seq = []
            for _ in range(2 * WINDOW):  # crosses the compaction boundary
                token = mx.argmax(out[:, -1, :], axis=-1)
                seq.append(token.item())
                out = model(token[None], cache=cache)
            tokens[name] = seq

        self.assertEqual(
            tokens["windowed"],
            tokens["full"],
            "windowed caches diverge from full history",
        )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestWindowedBatchedDecode(CustomTestCase):
    """The production decode path (``MLXAttentionWrapper._batched_decode``).

    Windowed and full-history caches are driven through the *same* wrapper
    with the same inputs.  The wrapper slices the trailing window off
    whatever ``get_kv`` returns, so both runs must build byte-identical SDPA
    inputs and the outputs must be bit-equal, not merely close.
    """

    def test_chained_decode_across_compaction_boundary(self):
        """Decode steps built in still-lazy pairs, riding a compaction.

        Pairs mirror ``decode_batch_start_chained``: step N+1's graph is
        built before step N materialises.  Compaction allocates a fresh
        buffer instead of mutating in place, so step N's returned views must
        stay valid.  Prefill lengths 15 and 6 put the oversized-chunk shrink
        in the first pair and a steady-state rebuild in a later one.
        """
        mx.random.seed(1)
        attn = _tiny_gpt_oss_model().model.layers[0].self_attn
        wrapper = MLXAttentionWrapper(attn, layer_idx=0, window_size=WINDOW)
        wins, fulls = [], []
        for length in (15, 6):
            x = mx.random.normal((1, length, HIDDEN))
            win = WindowedAttentionKVCache(WINDOW)
            full = ContiguousAttentionKVCache(max_seq_len=64)
            for cache in (win, full):
                attn(x, make_attention_mask(length, 0, window_size=WINDOW), cache=cache)
            wins.append(win)
            fulls.append(full)

        def build_step(x_step, caches):
            ctx = BatchedDecodeContext(
                batch_size=len(caches),
                seq_lens=[c.offset for c in caches],
                attention_layer_caches=[caches],
            )
            return wrapper._batched_decode(x_step, ctx)

        for pair in range(6):
            steps = [mx.random.normal((len(wins), 1, HIDDEN)) for _ in range(2)]
            # Both graphs are built before either materialises.
            got = [(build_step(x, wins), build_step(x, fulls)) for x in steps]
            mx.eval(got)
            for tag, (windowed, full) in zip("ab", got):
                self.assertTrue(
                    mx.array_equal(windowed, full), f"pair {pair}{tag} diverges"
                )
            for win, full in zip(wins, fulls):
                self.assertEqual(win.offset, full.offset)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestModelRunnerCacheWiring(CustomTestCase):
    """``_new_native_cache``/``_acquire_cache`` wiring without loading weights."""

    def _stub_runner(self, window_map):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        layers, attrs = find_attention_layers(_tiny_gpt_oss_model())
        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            layers, attrs, layer_window_sizes=window_map
        )
        runner._max_seq_len = 4096
        runner._cache_pool = []
        return runner

    def test_windowed_only_for_sliding_layers_and_reset_on_reuse(self):
        runner = self._stub_runner(get_layer_window_sizes(_tiny_gpt_oss_model()))
        cache = runner._new_native_cache()
        self.assertEqual(
            [type(c) for c in cache],
            [
                WindowedAttentionKVCache,
                ContiguousAttentionKVCache,
                WindowedAttentionKVCache,
                ContiguousAttentionKVCache,
            ],
        )
        self.assertEqual(cache[0].window, WINDOW)

        # Models without container windows have an empty map, so every
        # attention layer keeps a contiguous full-history cache.
        for c in self._stub_runner({})._new_native_cache():
            self.assertIsInstance(c, ContiguousAttentionKVCache)

        k = mx.random.normal((1, N_KV_HEADS, 10, HEAD_DIM))
        cache[0].update_and_fetch(k, k)
        cache[1].update_and_fetch(k, k)
        runner._release_cache(cache)
        reused = runner._acquire_cache()
        self.assertIs(reused, cache)
        for c in reused:
            self.assertEqual(c.offset, 0)
        # A stale local buffer would prepend the previous request's KV.
        out, _ = reused[0].update_and_fetch(k, k)
        self.assertEqual(out.shape[2], 10)


if __name__ == "__main__":
    unittest.main()
