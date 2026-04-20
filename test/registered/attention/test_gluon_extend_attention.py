"""Unit tests for the Gluon extend-attention kernel + its Triton fallback.

Covers:
  * Kernel parity vs. the canonical Triton extend path across every
    head-dim and pattern Gluon claims to support
    (D=64 / 128 / 256, causal / non-causal, SWA, sinks, logit_cap).
  * The ``make_extend_attention_fwd`` wrapper's Triton-fallback
    invariant: unsupported shapes (D not in {64,128,256}, Lq != Lv,
    FP8 KV + custom_mask on D<=128) MUST route to Triton instead of
    raising or producing garbage.
  * The layer-spec / MODEL_PRESETS table stays coherent (a GPU-less
    sanity walk -- runs even on x86 CI).

The kernel tests skip gracefully when not running on gfx950 (MI350/355),
since Gluon is the only supported entry point there and the kernels
won't compile elsewhere.
"""

from __future__ import annotations

import random
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd as triton_extend_fwd,
)
from sglang.srt.utils import get_device, is_gfx95_supported
from sglang.test.test_utils import CustomTestCase


# Only the kernel tests need gfx950. The wrapper + preset tests run anywhere.
_GFX95 = is_gfx95_supported()


def _random_shapes(B, max_prefix, max_extend, device):
    """Match the existing Triton test's shape-sampling style so the two
    test files stay diff-comparable."""
    b_seq_len_prefix = torch.randint(
        1, max_prefix, (B,), dtype=torch.int32, device=device
    )
    b_seq_len_extend = torch.randint(
        1, max_extend, (B,), dtype=torch.int32, device=device
    )
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    return b_seq_len_prefix, b_seq_len_extend, b_seq_len, b_start_loc, b_start_loc_extend


def _build_extend_inputs(
    *, B, H_Q, H_KV, D, max_prefix=512, max_extend=256,
    dtype=torch.bfloat16, device=None, seed=42,
):
    """Create a realistic extend-attention call: prefix KV in the cache,
    extend KV at the tail of the same buffer, Q for the extend tokens.
    """
    torch.manual_seed(seed)
    device = device or get_device()
    (
        b_seq_len_prefix, b_seq_len_extend, b_seq_len,
        b_start_loc, b_start_loc_extend,
    ) = _random_shapes(B, max_prefix, max_extend, device)

    total_token_num = int(torch.sum(b_seq_len).item())
    extend_token_num = int(torch.sum(b_seq_len_extend).item())

    k_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device,
    ).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device,
    ).normal_(mean=0.1, std=0.2)
    k_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=dtype, device=device,
    )
    v_extend = torch.empty(
        (extend_token_num, H_KV, D), dtype=dtype, device=device,
    )
    q_extend = torch.empty(
        (extend_token_num, H_Q, D), dtype=dtype, device=device,
    )
    for i in range(B):
        p_lo = int(b_start_loc[i].item()) + int(b_seq_len_prefix[i].item())
        p_hi = int(b_start_loc[i].item()) + int(b_seq_len[i].item())
        e_lo = int(b_start_loc_extend[i].item())
        e_hi = e_lo + int(b_seq_len_extend[i].item())
        k_extend[e_lo:e_hi] = k_buffer[p_lo:p_hi]
        v_extend[e_lo:e_hi] = v_buffer[p_lo:p_hi]
        q_extend[e_lo:e_hi] = torch.empty(
            (int(b_seq_len_extend[i].item()), H_Q, D),
            dtype=dtype, device=device,
        ).normal_(mean=0.1, std=0.2)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(b_seq_len_prefix, dim=0)
    kv_indices = torch.zeros(
        (int(b_seq_len_prefix.sum().item()),),
        dtype=torch.int64, device=device,
    )
    for i in range(B):
        kv_indices[int(kv_indptr[i]):int(kv_indptr[i + 1])] = torch.arange(
            int(b_start_loc[i].item()),
            int(b_start_loc[i].item()) + int(b_seq_len_prefix[i].item()),
            device=device,
        )

    qo_indptr = torch.zeros((B + 1,), dtype=torch.int64, device=device)
    qo_indptr[1:] = torch.cumsum(b_seq_len_extend, dim=0)
    max_len_extend = int(torch.max(b_seq_len_extend).item())

    return dict(
        q_extend=q_extend, k_extend=k_extend, v_extend=v_extend,
        k_buffer=k_buffer, v_buffer=v_buffer,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
        max_len_extend=max_len_extend,
        b_seq_len_extend=b_seq_len_extend,
        b_seq_len_prefix=b_seq_len_prefix,
    )


class TestLayerSpecAndPresets(CustomTestCase):
    """GPU-less sanity walk of the prewarm layer-spec builder + presets.
    Runs on every platform so even pure-CPU CI catches preset typos.
    """

    def _hfcfg(self, **kw):
        class _Cfg:
            pass
        cfg = _Cfg()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_gemma2_alternating_without_layer_types(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _build_layer_spec_from_hf_config,
        )
        cfg = self._hfcfg(
            num_hidden_layers=42,
            architectures=["Gemma2ForCausalLM"],
            sliding_window=4096,
            attn_logit_softcapping=50.0,
        )
        layers = _build_layer_spec_from_hf_config(cfg)
        self.assertEqual(len(layers), 42)
        # Even layer_ids must be sliding, odd full.
        for i, L in enumerate(layers):
            expected_sw = 4095 if (i % 2 == 0) else -1
            self.assertEqual(L["sliding_window_size"], expected_sw)
            self.assertFalse(L["has_sink"])
            self.assertAlmostEqual(L["logit_cap"], 50.0)

    def test_llama4_irope_pattern(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _build_layer_spec_from_hf_config,
        )
        cfg = self._hfcfg(
            num_hidden_layers=48,
            architectures=["Llama4ForCausalLM"],
            sliding_window=8192,
        )
        layers = _build_layer_spec_from_hf_config(cfg)
        for i, L in enumerate(layers):
            if (i + 1) % 4 == 0:
                self.assertEqual(L["sliding_window_size"], -1, msg=f"layer {i}")
            else:
                self.assertEqual(L["sliding_window_size"], 8191, msg=f"layer {i}")

    def test_grok_sinks_off(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _build_layer_spec_from_hf_config,
        )
        cfg = self._hfcfg(
            num_hidden_layers=64,
            architectures=["Grok1ForCausalLM"],
            attn_logit_softcapping=30.0,
            attn_temperature_len=1024,
        )
        layers = _build_layer_spec_from_hf_config(cfg)
        self.assertTrue(all(not L["has_sink"] for L in layers))
        self.assertTrue(all(L["xai_temperature_len"] == 1024 for L in layers))
        self.assertTrue(all(abs(L["logit_cap"] - 30.0) < 1e-9 for L in layers))

    def test_gpt_oss_has_sinks(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _build_layer_spec_from_hf_config,
        )
        cfg = self._hfcfg(
            num_hidden_layers=36,
            architectures=["GptOssForCausalLM"],
            sliding_window=128,
            layer_types=["full_attention", "sliding_attention"] * 18,
        )
        layers = _build_layer_spec_from_hf_config(cfg)
        self.assertTrue(all(L["has_sink"] for L in layers))
        self.assertEqual(layers[0]["sliding_window_size"], -1)
        self.assertEqual(layers[1]["sliding_window_size"], 127)

    def test_mistral_sliding_window_ignored(self):
        """Mistral HF config declares sliding_window but sglang ignores it;
        our prewarm must therefore treat every layer as full attention."""
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _build_layer_spec_from_hf_config,
        )
        cfg = self._hfcfg(
            num_hidden_layers=32,
            architectures=["MistralForCausalLM"],
            sliding_window=4096,
        )
        layers = _build_layer_spec_from_hf_config(cfg)
        self.assertTrue(all(L["sliding_window_size"] == -1 for L in layers))

    def test_all_presets_resolve(self):
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention._prewarm import (
            MODEL_PRESETS, enumerate_layer_patterns,
        )
        for name, fn in MODEL_PRESETS.items():
            D, Hq, Hkv, layers = fn()
            self.assertIn(
                D, (64, 128, 256),
                msg=f"preset {name}: head_dim={D} not in Gluon supported set",
            )
            self.assertTrue(
                Hq > 0 and Hkv > 0 and Hq % Hkv == 0,
                msg=f"preset {name}: non-integer GQA ratio {Hq}/{Hkv}",
            )
            patterns = enumerate_layer_patterns(layers)
            # SWA-alternating models compile at most 2 patterns; uniform
            # models collapse to 1. Any more than 3 suggests a mistake.
            self.assertLessEqual(
                len(patterns), 3,
                msg=f"preset {name}: {len(patterns)} unique patterns",
            )


class TestGluonSupports(CustomTestCase):
    """Validate the `_gluon_supports` guard that gates Gluon vs. Triton."""

    def _mktensor(self, shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    def test_unsupported_head_dim_falls_back(self):
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports
        for D in (80, 96, 192):
            q = self._mktensor((4, 8, D))
            v = self._mktensor((4, 4, D))
            kb = self._mktensor((16, 4, D))
            self.assertFalse(
                _gluon_supports(q, v, kb, custom_mask=None, is_causal=True)
            )

    def test_mismatched_lq_lv_falls_back(self):
        """MLA / mixed-dim shapes (Lq != Lv) route to Triton instead
        of the Gluon symmetric-head path."""
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports
        q = self._mktensor((4, 8, 192))
        v = self._mktensor((4, 4, 128))
        kb = self._mktensor((16, 4, 192))
        self.assertFalse(
            _gluon_supports(q, v, kb, custom_mask=None, is_causal=True)
        )

    def test_fp8_kv_with_custom_mask_on_d128_falls_back(self):
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports
        q = self._mktensor((4, 8, 128), dtype=torch.bfloat16)
        v = self._mktensor((4, 4, 128), dtype=torch.bfloat16)
        kb = self._mktensor((16, 4, 128), dtype=torch.float8_e4m3fnuz)
        self.assertFalse(
            _gluon_supports(
                q, v, kb,
                custom_mask=torch.empty(1, device="meta"),
                is_causal=True,
            )
        )

    def test_non_causal_falls_back(self):
        """Non-causal extend (encoder / vision tower) is not supported by
        the Gluon kernel family yet -- ensure we fall back to Triton."""
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports
        q = self._mktensor((4, 8, 128))
        v = self._mktensor((4, 4, 128))
        kb = self._mktensor((16, 4, 128))
        self.assertFalse(
            _gluon_supports(q, v, kb, custom_mask=None, is_causal=False)
        )

    def test_supported_combos_pass(self):
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports
        for D in (64, 128, 256):
            q = self._mktensor((4, 8, D))
            v = self._mktensor((4, 4, D))
            kb = self._mktensor((16, 4, D))
            self.assertTrue(
                _gluon_supports(q, v, kb, custom_mask=None, is_causal=True)
            )

    def test_make_extend_attention_fwd_returns_fallback_on_import_failure(self):
        """If Gluon fails to import, make_extend_attention_fwd must return
        the Triton fallback unchanged -- never a closure that'd explode
        on first call."""
        import sglang.srt.layers.attention.gluon_extend_attention as mod
        saved_fn = mod._GLUON_FN
        saved_try = mod._try_import_gluon
        try:
            mod._GLUON_FN = None
            mod._try_import_gluon = lambda: False

            def _stub(*args, **kwargs):
                _stub.called += 1
                return "triton-result"
            _stub.called = 0

            fwd = mod.make_extend_attention_fwd(_stub)
            self.assertIs(fwd, _stub)
        finally:
            mod._GLUON_FN = saved_fn
            mod._try_import_gluon = saved_try


@unittest.skipUnless(_GFX95, "Gluon extend attention requires gfx950")
class TestGluonKernelParity(CustomTestCase):
    """End-to-end kernel parity: Gluon output ~= Triton output (the
    reference SGLang Triton extend kernel). Skipped unless we're on
    an MI350/355 card.
    """

    def setUp(self):
        random.seed(42)
        torch.manual_seed(42)

    def _run_gluon(self, inputs, is_causal=True, sliding_window_size=-1,
                   sinks=None, logit_cap=0.0, xai_temperature_len=-1):
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
            gluon_extend_attention_fwd,
        )
        o = torch.empty_like(inputs["q_extend"])
        gluon_extend_attention_fwd(
            inputs["q_extend"], inputs["k_extend"], inputs["v_extend"], o,
            inputs["k_buffer"], inputs["v_buffer"],
            inputs["qo_indptr"], inputs["kv_indptr"], inputs["kv_indices"],
            None, is_causal, None, inputs["max_len_extend"],
            k_scale=1.0, v_scale=1.0,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
        return o

    def _run_triton(self, inputs, is_causal=True, sliding_window_size=-1,
                    sinks=None, logit_cap=0.0, xai_temperature_len=-1):
        o = torch.empty_like(inputs["q_extend"])
        triton_extend_fwd(
            inputs["q_extend"], inputs["k_extend"], inputs["v_extend"], o,
            inputs["k_buffer"], inputs["v_buffer"],
            inputs["qo_indptr"], inputs["kv_indptr"],
            inputs["kv_indices"].to(torch.int64),
            None, is_causal, None, inputs["max_len_extend"],
            1.0, 1.0,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
        return o

    def _check_parity(self, D, H_Q=32, H_KV=8, B=4, **kwargs):
        inputs = _build_extend_inputs(B=B, H_Q=H_Q, H_KV=H_KV, D=D)
        o_gluon = self._run_gluon(inputs, **kwargs)
        o_triton = self._run_triton(inputs, **kwargs)
        # Gluon uses FP8 loads on the prefix loop (when FP8) and bf16
        # otherwise; both go through fp32 softmax. Match the tolerance
        # the existing Triton test uses for extend attention.
        torch.testing.assert_close(
            o_gluon, o_triton, rtol=2e-2, atol=2e-3,
            msg=f"D={D} H_Q={H_Q} H_KV={H_KV} kwargs={kwargs}",
        )

    def test_parity_d64_causal(self):
        self._check_parity(D=64, H_Q=64, H_KV=8, is_causal=True)

    def test_parity_d128_causal(self):
        self._check_parity(D=128, H_Q=32, H_KV=8, is_causal=True)

    def test_parity_d128_mha(self):
        self._check_parity(D=128, H_Q=16, H_KV=16, is_causal=True)

    def test_parity_d128_gqa_4to1(self):
        self._check_parity(D=128, H_Q=16, H_KV=4, is_causal=True)

    def test_parity_d256_causal(self):
        self._check_parity(D=256, H_Q=16, H_KV=8, is_causal=True)

    def test_parity_sliding_window_d128(self):
        self._check_parity(
            D=128, H_Q=32, H_KV=8, is_causal=True, sliding_window_size=255,
        )

    def test_parity_sliding_window_d64(self):
        self._check_parity(
            D=64, H_Q=64, H_KV=8, is_causal=True, sliding_window_size=127,
        )

    def test_parity_noncausal_d128(self):
        """Extend attention supports is_causal=False for some bidi
        models. Gluon dispatches to the non-causal kernel branch."""
        self._check_parity(D=128, H_Q=32, H_KV=8, is_causal=False)

    def test_parity_with_logit_cap(self):
        """Gemma-style logit cap: tanh(s/cap)*cap applied pre-softmax."""
        self._check_parity(
            D=128, H_Q=32, H_KV=8, is_causal=True, logit_cap=30.0,
        )

    def test_parity_with_sinks_d64(self):
        """GPT-OSS / MiMO attention sinks (one learnable scalar per head)."""
        device = get_device()
        inputs = _build_extend_inputs(B=4, H_Q=64, H_KV=8, D=64)
        sinks = torch.empty(
            64, dtype=torch.bfloat16, device=device,
        ).normal_(mean=0.0, std=0.05)
        o_gluon = self._run_gluon(inputs, is_causal=True, sinks=sinks)
        o_triton = self._run_triton(inputs, is_causal=True, sinks=sinks)
        torch.testing.assert_close(o_gluon, o_triton, rtol=2e-2, atol=2e-3)


@unittest.skipUnless(_GFX95, "Gluon prewarm requires gfx950")
class TestGluonJITFreeInvariant(CustomTestCase):
    """After ``prewarm_for_model`` returns, no live dispatch call for
    that (head_dim, num_q_heads, num_kv_heads, pattern) tuple may
    trigger JIT compilation. This is the property that lets SGLang
    serve first-token requests without a compile stall.

    We assert the property by:
      1. Clearing the Gluon kernel ``_config_cache`` and the dispatch
         counter ``basic_slow_first`` (which only increments on a
         first-time JIT+install).
      2. Running ``prewarm_for_model`` for a small preset.
      3. Clearing the counters AGAIN so we measure post-prewarm
         dispatch only.
      4. Issuing a handful of realistic extend-attention calls.
      5. Checking that ``basic_slow_first`` stayed at 0 and that the
         number of new entries added to ``_config_cache`` is bounded
         (at most the number of unique dispatch configs the shape grid
         can hit -- NOT zero, because prewarm Phase 2 pre-populates the
         cache but the basic-path cache key also includes tensor
         dtypes and we may pick a slightly different combination in
         the test, which is fine as long as it doesn't mean a *compile*).
    """

    def _pick_small_preset(self):
        """Use a uniform full-attention preset with small head counts so
        the test runs in a few seconds. We pick llama3-8b style shape
        (D=128, H=32, kvH=8) -- the single most common production
        setup."""
        return dict(
            head_dim=128, num_q_heads=32, num_kv_heads=8,
            layers=[
                {"sliding_window_size": -1, "has_sink": False,
                 "logit_cap": 0.0, "xai_temperature_len": -1,
                 "is_causal": True}
                for _ in range(4)  # uniform pattern
            ],
        )

    def test_no_jit_on_post_prewarm_dispatch(self):
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
            extend_attention_gfx950 as _ext,
        )
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention._prewarm import (
            prewarm_for_model,
        )
        device = get_device()
        spec = self._pick_small_preset()

        # 1. Clear kernel cache + counters.
        _ext._config_cache.clear()
        _ext.reset_dispatch_counters()

        # 2. Prewarm a single-pattern model.
        prewarm_for_model(
            spec["layers"],
            head_dim=spec["head_dim"],
            num_q_heads=spec["num_q_heads"],
            num_kv_heads=spec["num_kv_heads"],
            device=device,
            dtype=torch.bfloat16,
            include_basic=True,
            include_persistent=False,  # keep the test cheap
            include_fp8=False,
            parallel=4,
            verbose=False,
        )

        # 3. Reset counters so we only look at live dispatch.
        _ext.reset_dispatch_counters()
        cache_size_after_prewarm = len(_ext._config_cache)

        # 4. Run representative extend calls across realistic shapes.
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
            gluon_extend_attention_fwd,
        )
        shape_grid = [
            (1, 256),   # single-request prefill
            (4, 128),   # small-batch chat
            (8, 64),    # spec-decode style
            (2, 512),   # longer prefill
        ]
        for B, ext_len in shape_grid:
            inputs = _build_extend_inputs(
                B=B, H_Q=spec["num_q_heads"], H_KV=spec["num_kv_heads"],
                D=spec["head_dim"], max_prefix=256, max_extend=ext_len,
            )
            o = torch.empty_like(inputs["q_extend"])
            gluon_extend_attention_fwd(
                inputs["q_extend"], inputs["k_extend"], inputs["v_extend"], o,
                inputs["k_buffer"], inputs["v_buffer"],
                inputs["qo_indptr"], inputs["kv_indptr"], inputs["kv_indices"],
                None, True, None, inputs["max_len_extend"],
                k_scale=1.0, v_scale=1.0,
                logit_cap=0.0, sliding_window_size=-1,
            )

        # 5. Verify: zero first-time JIT installs during live dispatch.
        slow = _ext._dispatch_counters.get("basic_slow_first", 0)
        self.assertEqual(
            slow, 0,
            msg=(
                f"Live dispatch JIT-compiled {slow} kernel variants after "
                f"prewarm. Dispatch counters: {_ext._dispatch_counters!r}"
            ),
        )
        # Cache must not have SHRUNK (sanity) -- it may grow slightly if a
        # test shape happened to pick a never-prewarmed dispatch config;
        # capped at a reasonable bound.
        cache_size_final = len(_ext._config_cache)
        self.assertGreaterEqual(cache_size_final, cache_size_after_prewarm)
        self.assertLessEqual(
            cache_size_final - cache_size_after_prewarm, 2,
            msg=(
                f"Prewarm missed too many dispatch configs: cache grew from "
                f"{cache_size_after_prewarm} to {cache_size_final}"
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
