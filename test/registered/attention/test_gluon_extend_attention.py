"""Tests for Gluon extend attention and its Triton fallback."""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd as triton_extend_fwd,
)
from sglang.srt.utils import get_device, is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

# Kernel tests need gfx950; wrapper tests run anywhere.
_GFX95 = is_gfx95_supported()


def _random_shapes(B, max_prefix, max_extend, device):
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

    return (
        b_seq_len_prefix,
        b_seq_len_extend,
        b_seq_len,
        b_start_loc,
        b_start_loc_extend,
    )


def _build_extend_inputs(
    *,
    B,
    H_Q,
    H_KV,
    D,
    max_prefix=512,
    max_extend=256,
    dtype=torch.bfloat16,
    device=None,
    seed=42,
):
    torch.manual_seed(seed)
    device = device or get_device()
    (
        b_seq_len_prefix,
        b_seq_len_extend,
        b_seq_len,
        b_start_loc,
        b_start_loc_extend,
    ) = _random_shapes(B, max_prefix, max_extend, device)

    total_token_num = int(torch.sum(b_seq_len).item())
    extend_token_num = int(torch.sum(b_seq_len_extend).item())

    k_buffer = torch.empty(
        (total_token_num, H_KV, D),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.1, std=0.2)
    k_extend = torch.empty(
        (extend_token_num, H_KV, D),
        dtype=dtype,
        device=device,
    )
    v_extend = torch.empty(
        (extend_token_num, H_KV, D),
        dtype=dtype,
        device=device,
    )
    q_extend = torch.empty(
        (extend_token_num, H_Q, D),
        dtype=dtype,
        device=device,
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
            dtype=dtype,
            device=device,
        ).normal_(mean=0.1, std=0.2)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(b_seq_len_prefix, dim=0)
    kv_indices = torch.zeros(
        (int(b_seq_len_prefix.sum().item()),),
        dtype=torch.int64,
        device=device,
    )
    for i in range(B):
        kv_indices[int(kv_indptr[i]) : int(kv_indptr[i + 1])] = torch.arange(
            int(b_start_loc[i].item()),
            int(b_start_loc[i].item()) + int(b_seq_len_prefix[i].item()),
            device=device,
        )

    qo_indptr = torch.zeros((B + 1,), dtype=torch.int64, device=device)
    qo_indptr[1:] = torch.cumsum(b_seq_len_extend, dim=0)
    max_len_extend = int(torch.max(b_seq_len_extend).item())

    return dict(
        q_extend=q_extend,
        k_extend=k_extend,
        v_extend=v_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        max_len_extend=max_len_extend,
        b_seq_len_extend=b_seq_len_extend,
        b_seq_len_prefix=b_seq_len_prefix,
    )


class TestGluonSupports(CustomTestCase):

    def _mktensor(self, shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    def test_unsupported_head_dim_falls_back(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _gluon_supports,
            _gluon_unsupported_reason,
        )

        for D in (80, 96, 192):
            q = self._mktensor((4, 8, D))
            v = self._mktensor((4, 4, D))
            kb = self._mktensor((16, 4, D))
            self.assertFalse(
                _gluon_supports(q, v, kb, custom_mask=None, is_causal=True)
            )
            self.assertEqual(
                _gluon_unsupported_reason(
                    q,
                    v,
                    kb,
                    custom_mask=None,
                    is_causal=True,
                ),
                f"unsupported_head_dim_{D}",
            )

    def test_mismatched_lq_lv_falls_back(self):
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports

        q = self._mktensor((4, 8, 192))
        v = self._mktensor((4, 4, 128))
        kb = self._mktensor((16, 4, 192))
        self.assertFalse(_gluon_supports(q, v, kb, custom_mask=None, is_causal=True))

    def test_fp8_kv_with_custom_mask_on_d128_falls_back(self):
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports

        q = self._mktensor((4, 8, 128), dtype=torch.bfloat16)
        v = self._mktensor((4, 4, 128), dtype=torch.bfloat16)
        kb = self._mktensor((16, 4, 128), dtype=torch.float8_e4m3fn)
        self.assertFalse(
            _gluon_supports(
                q,
                v,
                kb,
                custom_mask=torch.empty(1, device="meta"),
                is_causal=True,
            )
        )

    def test_fnuz_fp8_kv_falls_back(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _gluon_supports,
            _gluon_unsupported_reason,
        )

        q = self._mktensor((4, 8, 128), dtype=torch.bfloat16)
        v = self._mktensor((4, 4, 128), dtype=torch.bfloat16)
        kb = self._mktensor((16, 4, 128), dtype=torch.float8_e4m3fnuz)
        self.assertFalse(_gluon_supports(q, v, kb, custom_mask=None, is_causal=True))
        self.assertEqual(
            _gluon_unsupported_reason(q, v, kb, custom_mask=None, is_causal=True),
            "unsupported_fp8_dtype_torch.float8_e4m3fnuz",
        )

    def test_fp8_d256_falls_back(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            _gluon_supports,
            _gluon_unsupported_reason,
        )

        q = self._mktensor((4, 8, 256), dtype=torch.bfloat16)
        v = self._mktensor((4, 4, 256), dtype=torch.bfloat16)
        kb = self._mktensor((16, 4, 256), dtype=torch.float8_e4m3fn)
        self.assertFalse(_gluon_supports(q, v, kb, custom_mask=None, is_causal=True))
        self.assertEqual(
            _gluon_unsupported_reason(q, v, kb, custom_mask=None, is_causal=True),
            "unsupported_fp8_d256",
        )

    def test_non_causal_supported(self):
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports

        q = self._mktensor((4, 8, 128))
        v = self._mktensor((4, 4, 128))
        kb = self._mktensor((16, 4, 128))
        self.assertTrue(_gluon_supports(q, v, kb, custom_mask=None, is_causal=False))

    def test_supported_combos_pass(self):
        from sglang.srt.layers.attention.gluon_extend_attention import _gluon_supports

        for D in (64, 128, 256):
            q = self._mktensor((4, 8, D))
            v = self._mktensor((4, 4, D))
            kb = self._mktensor((16, 4, D))
            self.assertTrue(_gluon_supports(q, v, kb, custom_mask=None, is_causal=True))

    def test_make_extend_attention_fwd_returns_fallback_on_import_failure(self):
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

    def test_get_extend_attention_hints_from_cpu_metadata(self):
        from sglang.srt.layers.attention.gluon_extend_attention import (
            get_extend_attention_hints,
        )

        forward_batch = SimpleNamespace(
            extend_prefix_lens_cpu=[8, 16, 32],
            extend_seq_lens_cpu=[1, 4, 2],
        )

        self.assertEqual(
            get_extend_attention_hints(forward_batch),
            {
                "total_prefix_len": 56,
                "total_extend_len": 7,
                "min_len_extend": 1,
            },
        )

    def _meta_wrapper_args(self, D=128, kv_dtype=torch.bfloat16):
        q = self._mktensor((4, 8, D), dtype=torch.bfloat16)
        k = self._mktensor((4, 4, D), dtype=torch.bfloat16)
        v = self._mktensor((4, 4, D), dtype=torch.bfloat16)
        o = self._mktensor((4, 8, D), dtype=torch.bfloat16)
        kb = self._mktensor((16, 4, D), dtype=kv_dtype)
        vb = self._mktensor((16, 4, D), dtype=kv_dtype)
        qo = torch.empty((2,), dtype=torch.int32, device="meta")
        kvp = torch.empty((2,), dtype=torch.int32, device="meta")
        kvi = torch.empty((0,), dtype=torch.int64, device="meta")
        return q, k, v, o, kb, vb, qo, kvp, kvi

    def test_wrapper_unsupported_shape_uses_fallback(self):
        import sglang.srt.layers.attention.gluon_extend_attention as mod

        saved_fn = mod._GLUON_FN
        saved_try = mod._try_import_gluon
        try:

            def _should_not_run(*args, **kwargs):
                raise AssertionError("unsupported shape should not call Gluon")

            mod._GLUON_FN = _should_not_run
            mod._try_import_gluon = lambda: True

            def _fallback(*args, **kwargs):
                return "triton-result"

            fwd = mod.make_extend_attention_fwd(_fallback)
            result = fwd(
                *self._meta_wrapper_args(D=80),
                None,
                True,
                None,
                1,
                total_prefix_len=0,
                total_extend_len=4,
                min_len_extend=1,
            )
            self.assertEqual(result, "triton-result")
        finally:
            mod._GLUON_FN = saved_fn
            mod._try_import_gluon = saved_try

    def test_wrapper_exception_uses_fallback(self):
        import sglang.srt.layers.attention.gluon_extend_attention as mod

        saved_fn = mod._GLUON_FN
        saved_try = mod._try_import_gluon
        try:

            def _raise(*args, **kwargs):
                raise RuntimeError("boom")

            mod._GLUON_FN = _raise
            mod._try_import_gluon = lambda: True

            def _fallback(*args, **kwargs):
                return "triton-result"

            fwd = mod.make_extend_attention_fwd(_fallback)
            result = fwd(
                *self._meta_wrapper_args(D=128),
                None,
                True,
                None,
                1,
                total_prefix_len=0,
                total_extend_len=4,
                min_len_extend=1,
            )
            self.assertEqual(result, "triton-result")
        finally:
            mod._GLUON_FN = saved_fn
            mod._try_import_gluon = saved_try


@unittest.skipUnless(_GFX95, "Gluon extend attention requires gfx950")
class TestGluonKernelParity(CustomTestCase):

    def setUp(self):
        random.seed(42)
        torch.manual_seed(42)

    def _run_gluon(
        self,
        inputs,
        is_causal=True,
        sliding_window_size=-1,
        sinks=None,
        logit_cap=0.0,
        xai_temperature_len=-1,
    ):
        from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention import (
            gluon_extend_attention_fwd,
        )

        o = torch.empty_like(inputs["q_extend"])
        gluon_extend_attention_fwd(
            inputs["q_extend"],
            inputs["k_extend"],
            inputs["v_extend"],
            o,
            inputs["k_buffer"],
            inputs["v_buffer"],
            inputs["qo_indptr"],
            inputs["kv_indptr"],
            inputs["kv_indices"],
            None,
            is_causal,
            None,
            inputs["max_len_extend"],
            k_scale=1.0,
            v_scale=1.0,
            logit_cap=logit_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
        return o

    def _run_triton(
        self,
        inputs,
        is_causal=True,
        sliding_window_size=-1,
        sinks=None,
        logit_cap=0.0,
        xai_temperature_len=-1,
    ):
        o = torch.empty_like(inputs["q_extend"])
        triton_extend_fwd(
            inputs["q_extend"],
            inputs["k_extend"],
            inputs["v_extend"],
            o,
            inputs["k_buffer"],
            inputs["v_buffer"],
            inputs["qo_indptr"],
            inputs["kv_indptr"],
            inputs["kv_indices"].to(torch.int64),
            None,
            is_causal,
            None,
            inputs["max_len_extend"],
            1.0,
            1.0,
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
        torch.testing.assert_close(
            o_gluon,
            o_triton,
            rtol=2e-2,
            atol=2e-3,
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

    def test_parity_d128_fp8_kv(self):
        inputs = _build_extend_inputs(B=2, H_Q=16, H_KV=4, D=128)
        inputs["k_buffer"] = inputs["k_buffer"].to(torch.float8_e4m3fn)
        inputs["v_buffer"] = inputs["v_buffer"].to(torch.float8_e4m3fn)
        o_gluon = self._run_gluon(inputs, is_causal=True)
        o_triton = self._run_triton(inputs, is_causal=True)
        torch.testing.assert_close(o_gluon, o_triton, rtol=5e-2, atol=5e-2)

    def test_parity_d256_causal(self):
        self._check_parity(D=256, H_Q=16, H_KV=8, is_causal=True)

    def test_parity_sliding_window_d128(self):
        self._check_parity(
            D=128,
            H_Q=32,
            H_KV=8,
            is_causal=True,
            sliding_window_size=255,
        )

    def test_parity_sliding_window_d64(self):
        self._check_parity(
            D=64,
            H_Q=64,
            H_KV=8,
            is_causal=True,
            sliding_window_size=127,
        )

    def test_parity_noncausal_d128(self):
        self._check_parity(D=128, H_Q=32, H_KV=8, is_causal=False)

    def test_parity_with_logit_cap(self):
        self._check_parity(
            D=128,
            H_Q=32,
            H_KV=8,
            is_causal=True,
            logit_cap=30.0,
        )

    def test_parity_with_sinks_d64(self):
        device = get_device()
        inputs = _build_extend_inputs(B=4, H_Q=64, H_KV=8, D=64)
        sinks = torch.empty(
            64,
            dtype=torch.bfloat16,
            device=device,
        ).normal_(mean=0.0, std=0.05)
        o_gluon = self._run_gluon(inputs, is_causal=True, sinks=sinks)
        o_triton = self._run_triton(inputs, is_causal=True, sinks=sinks)
        torch.testing.assert_close(o_gluon, o_triton, rtol=2e-2, atol=2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
