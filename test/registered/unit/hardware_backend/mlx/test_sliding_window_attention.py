"""Unit tests for MLX sliding-window attention support (gpt-oss style models).

gpt-oss interleaves sliding-window and full-attention layers, names its
softmax scale ``sm_scale``, and adds per-head attention sinks. These tests
pin the three seams that make such models work on the MLX backend:

1. The attention contract accepts ``sm_scale`` and exposes per-layer window
   sizes read from the mlx-lm container convention (``layer_types`` +
   ``window_size``).
2. The cache shims' ``make_mask`` mirrors mlx_lm's
   ``cache.create_attention_mask`` exactly — in particular ``window_size``
   must produce a banded mask (including for N == 1) instead of being
   silently dropped, or sliding-window layers degrade to full attention.
3. ``MLXAttentionWrapper._batched_decode`` applies the window by truncating
   each request's KV to the trailing window, passes ``sinks`` through, and
   uses the contract scale helper.

The AOT RoPE kernel gating is also pinned: YarnRoPE (used by gpt-oss) bakes
its base and scaling into precomputed ``_freqs`` plus an ``mscale`` factor,
so the vanilla-RoPE Metal kernel must reject it rather than silently compute
with base=10000 and no yarn scaling.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm"

if _HAS_MLX:
    import mlx.core as mx
    from mlx_lm.models import gpt_oss
    from mlx_lm.models.base import create_causal_mask
    from mlx_lm.models.cache import KVCache

    import sglang.srt.hardware_backend.mlx.aot as mlx_aot
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        AttentionOffsetCache,
        BatchedDecodeContext,
        ContiguousAttentionKVCache,
        MLXAttentionWrapper,
        PoolBackedAttentionKVCache,
        find_attention_layers,
        get_attention_scale,
        get_layer_window_sizes,
        is_attention_module,
        make_attention_mask,
        patch_model_attention,
    )

TINY_WINDOW = 8


def _tiny_gpt_oss_model():
    """Randomly initialized 4-layer gpt_oss with alternating sliding/full layers."""
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
class TestGptOssAttentionContract(CustomTestCase):
    def test_gpt_oss_attention_passes_contract(self):
        model = _tiny_gpt_oss_model()
        attn = model.model.layers[0].self_attn

        self.assertFalse(hasattr(attn, "scale"))
        self.assertTrue(hasattr(attn, "sm_scale"))
        self.assertTrue(is_attention_module(attn))

        layers, attrs = find_attention_layers(model)
        self.assertEqual(len(layers), 4)
        self.assertEqual(attrs, ["self_attn"] * 4)

    def test_module_without_any_scale_attr_fails_contract(self):
        attn = _tiny_gpt_oss_model().model.layers[0].self_attn
        scaleless = SimpleNamespace(
            q_proj=attn.q_proj,
            k_proj=attn.k_proj,
            v_proj=attn.v_proj,
            o_proj=attn.o_proj,
            rope=attn.rope,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.assertFalse(is_attention_module(scaleless))

    def test_get_attention_scale_prefers_scale_over_sm_scale(self):
        self.assertEqual(get_attention_scale(SimpleNamespace(scale=0.5)), 0.5)
        self.assertEqual(get_attention_scale(SimpleNamespace(sm_scale=0.25)), 0.25)
        self.assertEqual(
            get_attention_scale(SimpleNamespace(scale=0.5, sm_scale=0.25)), 0.5
        )
        self.assertIsNone(get_attention_scale(SimpleNamespace()))

    def test_get_layer_window_sizes_reads_gpt_oss_container(self):
        windows = get_layer_window_sizes(_tiny_gpt_oss_model())
        self.assertEqual(windows, {0: TINY_WINDOW, 1: None, 2: TINY_WINDOW, 3: None})

    def test_get_layer_window_sizes_reads_sliding_window_alias(self):
        # olmo3/llama-style containers name the scalar ``sliding_window``.
        model = SimpleNamespace(
            model=SimpleNamespace(
                layer_types=["sliding_attention", "full_attention"],
                sliding_window=16,
            )
        )
        self.assertEqual(get_layer_window_sizes(model), {0: 16, 1: None})

    def test_get_layer_window_sizes_defaults_to_empty(self):
        self.assertEqual(get_layer_window_sizes(SimpleNamespace()), {})
        no_window = SimpleNamespace(
            model=SimpleNamespace(layer_types=["sliding_attention"])
        )
        self.assertEqual(get_layer_window_sizes(no_window), {})

    def test_patch_warns_when_window_declared_but_unmapped(self):
        # A container that declares a scalar window without layer_types
        # (gemma3-style pattern models): prefill masks honor the window but
        # batched decode cannot; the mismatch must be surfaced.
        model = _tiny_gpt_oss_model()
        model.model.layer_types = []
        with self.assertLogs(
            "sglang.srt.hardware_backend.mlx.kv_cache.model_patching",
            level="WARNING",
        ) as logs:
            patch_model_attention(model)
        self.assertTrue(any("sliding window" in msg for msg in logs.output))
        wrappers = [layer.self_attn for layer in model.model.layers]
        self.assertEqual([w._window_size for w in wrappers], [None] * 4)

    def test_patch_model_attention_assigns_window_sizes(self):
        model = _tiny_gpt_oss_model()
        self.assertEqual(patch_model_attention(model), 4)
        wrappers = [layer.self_attn for layer in model.model.layers]
        self.assertTrue(all(isinstance(w, MLXAttentionWrapper) for w in wrappers))
        self.assertEqual(
            [w._window_size for w in wrappers],
            [TINY_WINDOW, None, TINY_WINDOW, None],
        )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestShimMakeMask(CustomTestCase):
    """The shims must return exactly what mlx_lm's own KVCache.make_mask returns."""

    def _shims(self, offset):
        contig = ContiguousAttentionKVCache(
            n_kv_heads=1, head_dim=2, max_seq_len=64, dtype=mx.float32
        )
        contig.offset = offset
        pool_backed = PoolBackedAttentionKVCache(
            pool=None, layer_idx=0, slots=None, prefix_len=offset
        )
        return (AttentionOffsetCache(offset=offset), contig, pool_backed)

    def _assert_same_mask(self, got, ref, msg):
        if ref is None or isinstance(ref, str):
            self.assertEqual(got, ref, msg)
        else:
            self.assertTrue(
                isinstance(got, mx.array) and mx.array_equal(got, ref).item(),
                msg,
            )

    def test_shims_match_mlx_lm_reference(self):
        cases = [
            (N, offset, window, return_array)
            for N in (1, 4)
            for offset in (0, 3, 9)
            for window in (None, 4)
            for return_array in (False, True)
        ]
        for N, offset, window, return_array in cases:
            reference = KVCache()
            reference.offset = offset
            ref = reference.make_mask(N, return_array=return_array, window_size=window)
            for shim in self._shims(offset):
                got = shim.make_mask(N, return_array=return_array, window_size=window)
                self._assert_same_mask(
                    got,
                    ref,
                    f"{type(shim).__name__} mismatch for N={N} offset={offset} "
                    f"window={window} return_array={return_array}",
                )

    def test_windowed_mask_is_banded_including_self(self):
        # Query at absolute position 6 with W=4 may attend to keys 3..6
        # (j in [i - W + 1, i], the window includes the query itself).
        mask = make_attention_mask(1, 6, window_size=4)
        self.assertEqual(mask.shape, (1, 7))
        self.assertEqual(
            [bool(v) for v in mask[0]],
            [False, False, False, True, True, True, True],
        )

    def test_single_token_windowed_mask_is_not_none(self):
        # N == 1 must still produce a banded mask when a window is set —
        # returning None here silently disables the window during decode.
        mask = make_attention_mask(1, 200, window_size=128)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (1, 201))
        self.assertEqual(mx.sum(mask).item(), 128)

    def test_prefill_mask_bands_each_query_row(self):
        # N=5 rows starting at offset 7, W=4: row i allows [i+4, i+7].
        offset, N, window = 7, 5, 4
        mask = make_attention_mask(N, offset, window_size=window)
        self.assertEqual(mask.shape, (N, offset + N))
        for i in range(N):
            allowed = {j for j in range(offset + N) if bool(mask[i, j])}
            expected = set(range(offset + i - window + 1, offset + i + 1))
            self.assertEqual(allowed, expected, f"row {i}")

    def test_defaults_without_window_are_unchanged(self):
        self.assertIsNone(make_attention_mask(1, 5))
        self.assertEqual(make_attention_mask(4, 5), "causal")


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestBatchedDecodeSlidingWindow(CustomTestCase):
    """Batched decode must match a hand-built decode-step reference.

    The reference recomputes the decode step from the inner module's own
    projections and RoPE, attending over the full (untruncated) KV with
    mlx_lm's own ``create_causal_mask`` band. The wrapper instead truncates
    KV to the trailing window and pads ragged requests — mathematically
    identical, so the comparison is float-tight.

    The reference deliberately mirrors the wrapper's tensor shapes
    (projections and RoPE batched over B, not per request, and no
    full-sequence forward): MLX matmul/SDPA kernels pick different code
    paths per input shape, and e.g. a (2, 1, H) vs (1, 1, H) linear alone
    differs by ~1e-3 on this tiny model — far above the tolerance that
    makes this test able to catch real bugs.
    """

    HIDDEN = 64
    N_KV_HEADS = 2
    HEAD_DIM = 16

    def _prefill_cache(self, attn, x, window):
        cache = ContiguousAttentionKVCache(
            n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM,
            max_seq_len=32,
            dtype=mx.float32,
        )
        prefix = x[:, :-1, :]
        attn(
            prefix,
            make_attention_mask(prefix.shape[1], 0, window_size=window),
            cache=cache,
        )
        return cache

    def _project_last_tokens(self, attn, xs, offsets):
        B, D = len(xs), self.HEAD_DIM
        x_last = mx.concatenate([x[:, -1:, :] for x in xs], axis=0)
        q = attn.q_proj(x_last).reshape(B, 1, -1, D).transpose(0, 2, 1, 3)
        k = attn.k_proj(x_last).reshape(B, 1, -1, D).transpose(0, 2, 1, 3)
        v = attn.v_proj(x_last).reshape(B, 1, -1, D).transpose(0, 2, 1, 3)
        off = mx.array(offsets, dtype=mx.int32)
        return attn.rope(q, offset=off), attn.rope(k, offset=off), v

    def _reference_decode(self, attn, xs, caches, window):
        """Full-KV banded-mask decode; must be called before the wrapper
        writes the decode token into the shared caches."""
        offsets = [x.shape[1] - 1 for x in xs]
        q, k_new, v_new = self._project_last_tokens(attn, xs, offsets)
        outs = []
        for i, cache in enumerate(caches):
            k_prefix, v_prefix = cache.get_kv()
            k = mx.concatenate([k_prefix, k_new[i : i + 1]], axis=2)
            v = mx.concatenate([v_prefix, v_new[i : i + 1]], axis=2)
            mask = (
                create_causal_mask(1, offsets[i], window_size=window)
                if window is not None
                else None
            )
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
        out = mx.concatenate(outs, axis=0)
        out = out.transpose(0, 2, 1, 3).reshape(len(xs), 1, -1)
        return attn.o_proj(out)

    def _wrapper_decode(self, attn, window, xs, caches):
        wrapper = MLXAttentionWrapper(attn, layer_idx=0, window_size=window)
        ctx = BatchedDecodeContext(
            batch_size=len(xs),
            seq_lens=[x.shape[1] - 1 for x in xs],
            attention_layer_caches=[caches],
        )
        x_last = mx.concatenate([x[:, -1:, :] for x in xs], axis=0)
        out = wrapper._batched_decode(x_last, ctx)
        mx.eval(out)
        return out

    def _assert_matches_reference(self, attn, window, lens):
        mx.random.seed(0)
        xs = [mx.random.normal((1, L, self.HIDDEN)) for L in lens]
        caches = [self._prefill_cache(attn, x, window) for x in xs]
        ref = self._reference_decode(attn, xs, caches, window)
        got = self._wrapper_decode(attn, window, xs, caches)
        for i in range(len(xs)):
            diff = mx.abs(got[i : i + 1] - ref[i : i + 1]).max().item()
            self.assertLess(
                diff,
                1e-5,
                f"request {i} (len={lens[i]}, window={window}) diverges "
                f"from the manual decode reference by {diff}",
            )

    def test_sliding_layer_ragged_batch_matches_reference(self):
        # Request 0 crosses the window (12 > 8), request 1 stays inside (5 < 8):
        # covers trailing-window truncation and the local padding mask at once.
        model = _tiny_gpt_oss_model()
        attn = model.model.layers[0].self_attn
        self._assert_matches_reference(attn, TINY_WINDOW, lens=[12, 5])

    def test_sliding_layer_all_past_window_matches_reference(self):
        # Both requests exceed the window with unequal true lengths: the
        # shared context reports padding but the windowed lengths are all
        # equal, so the correct local pad is zero. Reusing the full-length
        # ctx metadata here would inject spurious padding.
        model = _tiny_gpt_oss_model()
        attn = model.model.layers[0].self_attn
        self._assert_matches_reference(attn, TINY_WINDOW, lens=[12, 10])

    def test_sliding_layer_single_request_matches_reference(self):
        # B=1 with truncation: the windowed no-padding branch (mask stays None).
        model = _tiny_gpt_oss_model()
        attn = model.model.layers[0].self_attn
        self._assert_matches_reference(attn, TINY_WINDOW, lens=[12])

    def test_full_attention_layer_matches_reference(self):
        # Full-attention gpt_oss layer: sinks + sm_scale on the unwindowed path.
        model = _tiny_gpt_oss_model()
        attn = model.model.layers[1].self_attn
        self._assert_matches_reference(attn, None, lens=[12, 5])


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestSdpaSinksSemantics(CustomTestCase):
    def test_sdpa_sinks_match_manual_softmax_with_sink_column(self):
        # Pin mx.fast.scaled_dot_product_attention(sinks=...) to the reference
        # semantics gpt-oss relies on: append one per-head sink logit to the
        # softmax and drop its probability column afterwards.
        mx.random.seed(0)
        B, H, Lq, Lk, D = 1, 4, 5, 9, 16
        scale = D**-0.5
        q = mx.random.normal((B, H, Lq, D))
        k = mx.random.normal((B, H, Lk, D))
        v = mx.random.normal((B, H, Lk, D))
        sinks = mx.random.normal((H,))
        offset, window = Lk - Lq, 4

        rinds = mx.arange(Lk)
        linds = mx.arange(offset, offset + Lq)
        mask = (linds[:, None] >= rinds[None]) & (linds[:, None] < rinds[None] + window)

        out_fast = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask, sinks=sinks
        )

        scores = (q * scale) @ k.transpose(0, 1, 3, 2)
        scores = mx.where(mask, scores, mx.finfo(mx.float32).min)
        sink_col = mx.broadcast_to(sinks[None, :, None, None], (B, H, Lq, 1))
        probs = mx.softmax(mx.concatenate([scores, sink_col], axis=-1), axis=-1)
        out_manual = probs[..., :-1] @ v

        diff = mx.abs(out_fast - out_manual).max().item()
        self.assertLess(diff, 1e-6)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestAotRopeKernelGating(CustomTestCase):
    """The vanilla-RoPE Metal kernel must reject scaled RoPE variants."""

    def _build_kernel(self, attn, head_dim=2, n_kv_heads=1):
        original_loader = mlx_aot._load_metal_rope_pool_fused
        mlx_aot._load_metal_rope_pool_fused = lambda: object()
        try:
            return mlx_aot._build_rope_kernel(
                mlx_aot.MlxAOTKernelBuildInputs(
                    sample_attn=attn,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                )
            )
        finally:
            mlx_aot._load_metal_rope_pool_fused = original_loader

    def test_vanilla_rope_is_accepted(self):
        attn = SimpleNamespace(
            n_heads=2,
            rope=SimpleNamespace(dims=2, traditional=False, base=10000.0),
        )
        self.assertTrue(self._build_kernel(attn).enabled)

    def test_gpt_oss_yarn_rope_is_rejected(self):
        # YarnRoPE has no ``base`` (it is baked into ``_freqs``) and applies
        # mscale outside mx.fast.rope; the kernel would silently compute
        # vanilla RoPE with base=10000.
        attn = _tiny_gpt_oss_model().model.layers[0].self_attn
        kernel = self._build_kernel(attn, head_dim=attn.head_dim, n_kv_heads=2)
        self.assertFalse(kernel.enabled)

    def test_missing_base_is_rejected(self):
        attn = SimpleNamespace(
            n_heads=2, rope=SimpleNamespace(dims=2, traditional=False)
        )
        self.assertFalse(self._build_kernel(attn).enabled)

    def test_precomputed_freqs_are_rejected(self):
        attn = SimpleNamespace(
            n_heads=2,
            rope=SimpleNamespace(
                dims=2, traditional=False, base=10000.0, _freqs=mx.ones(1)
            ),
        )
        self.assertFalse(self._build_kernel(attn).enabled)

    def test_nontrivial_mscale_is_rejected(self):
        attn = SimpleNamespace(
            n_heads=2,
            rope=SimpleNamespace(dims=2, traditional=False, base=10000.0, mscale=1.5),
        )
        self.assertFalse(self._build_kernel(attn).enabled)

    def test_linear_scale_is_rejected(self):
        # rope_scaling type "linear" yields nn.RoPE(..., scale=1/factor); the
        # kernel computes unscaled positions and must fall back, while the
        # nn.RoPE default scale of exactly 1.0 stays accepted.
        def attn(scale):
            return SimpleNamespace(
                n_heads=2,
                rope=SimpleNamespace(
                    dims=2, traditional=False, base=10000.0, scale=scale
                ),
            )

        self.assertFalse(self._build_kernel(attn(0.25)).enabled)
        self.assertTrue(self._build_kernel(attn(1.0)).enabled)


if __name__ == "__main__":
    unittest.main()
