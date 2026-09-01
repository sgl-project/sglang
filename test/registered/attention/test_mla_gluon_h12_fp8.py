"""Unit tests for h12 + FP8 Gluon routing and zero-pad fallback dispatch.

CPU-only mocks — no aiter/Triton/GPU required.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestMlaGluonCapability(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention.aiter_mla_gluon import (
            reset_mla_gluon_state_for_test,
        )

        reset_mla_gluon_state_for_test()

    def tearDown(self):
        from sglang.srt.layers.attention.aiter_mla_gluon import (
            reset_mla_gluon_state_for_test,
        )

        reset_mla_gluon_state_for_test()

    def test_env_disable_not_ready(self):
        from sglang.srt.layers.attention import aiter_mla_gluon as mod

        with envs.SGLANG_AITER_MLA_GLUON.override(False):
            mod.reset_mla_gluon_state_for_test()
            cap = mod.probe_mla_gluon_capability(force_refresh=True)
        self.assertFalse(cap.ready)
        self.assertIn("SGLANG_AITER_MLA_GLUON=0", cap.missing_for_ready())

    @mock.patch(
        "sglang.srt.layers.attention.aiter_mla_gluon._triton_cga_layout_ok",
        return_value=True,
    )
    @mock.patch(
        "sglang.srt.layers.attention.aiter_mla_gluon._triton_version",
        return_value="3.7.0",
    )
    def test_ready_when_import_and_cga_ok(self, _ver, _cga):
        from sglang.srt.layers.attention import aiter_mla_gluon as mod

        fake_fn = mock.Mock()
        with mock.patch.dict(
            "sys.modules",
            {
                "aiter": mock.MagicMock(),
                "aiter.ops": mock.MagicMock(),
                "aiter.ops.triton": mock.MagicMock(),
                "aiter.ops.triton.gluon": mock.MagicMock(),
                "aiter.ops.triton.gluon.mla_gluon": mock.MagicMock(mla_gluon=fake_fn),
            },
        ):
            mod.reset_mla_gluon_state_for_test()
            cap = mod.probe_mla_gluon_capability(force_refresh=True)
        self.assertTrue(cap.ready)
        self.assertIn("3.7.0", cap.summary)

    @mock.patch(
        "sglang.srt.layers.attention.aiter_mla_gluon._triton_cga_layout_ok",
        return_value=False,
    )
    @mock.patch(
        "sglang.srt.layers.attention.aiter_mla_gluon.mla_gluon_available",
        return_value=True,
    )
    def test_prefer_false_when_cga_missing(self, _avail, _cga):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.attention.aiter_mla_gluon import prefer_mla_gluon_decode

        self.assertFalse(
            prefer_mla_gluon_decode(
                head_pad_mode="zero", num_head=12, kv_cache_dtype=fp8_dtype
            )
        )

    @mock.patch(
        "sglang.srt.layers.attention.aiter_mla_gluon._gluon_runtime_ok",
        return_value=True,
    )
    def test_prefer_false_when_zero_pad_but_not_h12(self, _ok):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.attention.aiter_mla_gluon import prefer_mla_gluon_decode

        self.assertFalse(
            prefer_mla_gluon_decode(
                head_pad_mode="zero", num_head=10, kv_cache_dtype=fp8_dtype
            )
        )


class TestMlaGluonDecodeFallback(CustomTestCase):
    """Verify _forward_mla_decode uses zero-pad path when Gluon is off or fails."""

    def _make_backend(self):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        be = AiterAttnBackend.__new__(AiterAttnBackend)
        be.num_head = 12
        be.kv_cache_dtype = fp8_dtype
        be.head_pad_mode = "zero"
        be.num_head_padded = 16
        be.forward_metadata = mock.Mock(
            max_q_len=1,
            kv_indices=torch.zeros(4, dtype=torch.int32),
            kv_indptr=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            kv_last_page_len=torch.ones(4, dtype=torch.int32),
            qo_indptr=torch.arange(5, dtype=torch.int32),
            work_metadata=None,
            work_indptr=None,
            work_info_set=None,
            reduce_indptr=None,
            reduce_final_map=None,
            reduce_partial_map=None,
            num_kv_splits=None,
        )
        be.token_to_kv_pool = mock.Mock(
            get_key_buffer=lambda _lid: torch.zeros(8, 576, dtype=fp8_dtype)
        )
        be._resolve_fp8_kv_scale_float = mock.Mock(return_value=1.0)
        be._resolve_mla_gluon_min_kv_seq_len = mock.Mock(return_value=128)
        be._mla_decode_fwd_with_head_pad = mock.Mock(
            return_value=torch.zeros(4, 12, 512)
        )
        return be

    def _make_layer(self):
        layer = mock.Mock()
        layer.tp_q_head_num = 12
        layer.qk_head_dim = 576
        layer.v_head_dim = 512
        layer.scaling = 0.125
        layer.logit_cap = 0.0
        layer.layer_id = 0
        return layer

    @mock.patch(
        "sglang.srt.layers.attention.aiter_backend.prefer_mla_gluon_decode",
        return_value=False,
    )
    @mock.patch("sglang.srt.layers.attention.aiter_backend.mla_gluon_decode")
    def test_skips_gluon_when_disabled(self, mock_gluon, _prefer):
        be = self._make_backend()
        layer = self._make_layer()
        q = torch.zeros(4, 12, 576, dtype=torch.bfloat16)
        fb = mock.Mock(seq_lens=torch.tensor([128, 128, 128, 128]))

        out = be._forward_mla_decode(q, layer, fb, k_descale=1.0)

        mock_gluon.assert_not_called()
        be._mla_decode_fwd_with_head_pad.assert_called_once()
        self.assertIs(out, be._mla_decode_fwd_with_head_pad.return_value)

    @mock.patch(
        "sglang.srt.layers.attention.aiter_backend.prefer_mla_gluon_decode",
        return_value=True,
    )
    @mock.patch(
        "sglang.srt.layers.attention.aiter_backend.mla_gluon_decode",
        return_value=None,
    )
    def test_falls_back_when_gluon_returns_none(self, mock_gluon, _prefer):
        be = self._make_backend()
        layer = self._make_layer()
        q = torch.zeros(4, 12, 576, dtype=torch.bfloat16)
        fb = mock.Mock(seq_lens=torch.tensor([128, 128, 128, 128]))

        be._forward_mla_decode(q, layer, fb, k_descale=1.0)

        mock_gluon.assert_called_once()
        be._mla_decode_fwd_with_head_pad.assert_called_once()

    @mock.patch(
        "sglang.srt.layers.attention.aiter_backend.prefer_mla_gluon_decode",
        return_value=True,
    )
    @mock.patch("sglang.srt.layers.attention.aiter_backend.mla_gluon_decode")
    def test_uses_gluon_output_when_ok(self, mock_gluon, _prefer):
        gluon_out = torch.ones(4, 12, 512)
        mock_gluon.return_value = gluon_out
        be = self._make_backend()
        layer = self._make_layer()
        q = torch.zeros(4, 12, 576, dtype=torch.bfloat16)
        fb = mock.Mock(seq_lens=torch.tensor([128, 128, 128, 128]))

        out = be._forward_mla_decode(q, layer, fb, k_descale=1.0)

        mock_gluon.assert_called_once()
        be._mla_decode_fwd_with_head_pad.assert_not_called()
        self.assertIs(out, gluon_out)


if __name__ == "__main__":
    unittest.main()
