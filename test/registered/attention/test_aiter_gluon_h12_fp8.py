"""Unit tests for the aiter Gluon MLA path: h12 + FP8 routing, MTP (qlen>1)
shaping, and dispatch against the zero-pad ``mla_decode_fwd`` fallback.

Mocked throughout — no real aiter/Triton kernel is invoked.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs
from sglang.srt.layers.attention import aiter_mla_gluon as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_GLUON_FN = "sglang.srt.layers.attention.aiter_mla_gluon._gluon_fn"


def _fake_gluon_modules(*, cga_layout: bool):
    """sys.modules entries that make _gluon_fn() see a usable (or too-old) Triton.

    ``inspect.signature`` is called on ``gl.PaddedSharedLayout``, so that one has
    to be a real callable rather than a Mock — the parameter list is exactly what
    the Triton >= 3.7 probe reads.
    """
    if cga_layout:

        def padded_shared_layout(*, cga_layout=None):
            pass

    else:

        def padded_shared_layout():
            pass

    gl = mock.MagicMock()
    gl.PaddedSharedLayout = padded_shared_layout

    # `import a.b.c as x` binds via getattr on the parent module, so the parents
    # have to point at these exact objects -- a bare MagicMock parent would
    # auto-create a different child and shadow them.
    triton_gluon = mock.MagicMock(language=gl)
    triton_experimental = mock.MagicMock(gluon=triton_gluon)
    triton = mock.MagicMock(__version__="3.7.0", experimental=triton_experimental)

    fake_fn = mock.Mock(name="mla_gluon")
    aiter_mla = mock.MagicMock(mla_gluon=fake_fn)
    aiter_gluon = mock.MagicMock(mla_gluon=aiter_mla)
    aiter_triton = mock.MagicMock(gluon=aiter_gluon)
    aiter_ops = mock.MagicMock(triton=aiter_triton)
    aiter = mock.MagicMock(ops=aiter_ops)

    return fake_fn, {
        "triton": triton,
        "triton.experimental": triton_experimental,
        "triton.experimental.gluon": triton_gluon,
        "triton.experimental.gluon.language": gl,
        "aiter": aiter,
        "aiter.ops": aiter_ops,
        "aiter.ops.triton": aiter_triton,
        "aiter.ops.triton.gluon": aiter_gluon,
        "aiter.ops.triton.gluon.mla_gluon": aiter_mla,
    }


class TestGluonAvailability(CustomTestCase):
    """_gluon_fn() resolves the kernel, or None with the reason logged once."""

    def setUp(self):
        mod._gluon_fn.cache_clear()

    def tearDown(self):
        mod._gluon_fn.cache_clear()

    def test_none_when_env_disabled(self):
        with envs.SGLANG_AITER_MLA_GLUON.override(False):
            self.assertIsNone(mod._gluon_fn())

    def test_none_when_import_fails(self):
        # A None entry in sys.modules makes `import ...` raise ImportError.
        with mock.patch.dict("sys.modules", {"aiter.ops.triton.gluon.mla_gluon": None}):
            self.assertIsNone(mod._gluon_fn())

    def test_none_when_triton_lacks_cga_layout(self):
        _fn, modules = _fake_gluon_modules(cga_layout=False)
        with mock.patch.dict("sys.modules", modules):
            self.assertIsNone(mod._gluon_fn())

    def test_returns_kernel_when_ready(self):
        fake_fn, modules = _fake_gluon_modules(cga_layout=True)
        with mock.patch.dict("sys.modules", modules):
            self.assertIs(mod._gluon_fn(), fake_fn)

    def test_result_is_cached(self):
        fake_fn, modules = _fake_gluon_modules(cga_layout=True)
        with mock.patch.dict("sys.modules", modules):
            first = mod._gluon_fn()
        # Second call must not re-probe: the fake modules are gone by now, so a
        # re-probe would return None instead of the cached kernel.
        self.assertIs(mod._gluon_fn(), first)


class TestPreferMlaGluonDecode(CustomTestCase):
    """Only the validated h12 + zero-pad + FP8 topology may route to Gluon."""

    def _prefer(self, **kwargs):
        args = dict(head_pad_mode="zero", num_head=12, kv_cache_dtype=fp8_dtype)
        args.update(kwargs)
        return mod.prefer_mla_gluon_decode(**args)

    def test_true_for_h12_zero_pad_fp8(self):
        with mock.patch(_GLUON_FN, return_value=mock.Mock()):
            self.assertTrue(self._prefer())

    def test_false_when_gluon_unavailable(self):
        with mock.patch(_GLUON_FN, return_value=None):
            self.assertFalse(self._prefer())

    def test_false_for_other_head_counts(self):
        with mock.patch(_GLUON_FN, return_value=mock.Mock()):
            self.assertFalse(self._prefer(num_head=10))
            self.assertFalse(self._prefer(num_head=16))

    def test_false_for_non_zero_pad_topology(self):
        with mock.patch(_GLUON_FN, return_value=mock.Mock()):
            self.assertFalse(self._prefer(head_pad_mode="repeat"))
            self.assertFalse(self._prefer(head_pad_mode="none"))

    def test_false_for_non_fp8_kv(self):
        with mock.patch(_GLUON_FN, return_value=mock.Mock()):
            self.assertFalse(self._prefer(kv_cache_dtype=torch.bfloat16))


def _layer(num_head=12, qk_head_dim=576, v_head_dim=512):
    layer = mock.Mock()
    layer.tp_q_head_num = num_head
    layer.qk_head_dim = qk_head_dim
    layer.v_head_dim = v_head_dim
    layer.scaling = 0.125
    layer.logit_cap = 0.0
    layer.layer_id = 0
    return layer


class TestMlaGluonDecodeShapes(CustomTestCase):
    """Plain decode stays 3-D; target-verify (qlen>1) goes in as 4-D MTP."""

    def _call(self, *, num_tokens, qlen):
        layer = _layer()
        q = torch.zeros(num_tokens, 12, 576, dtype=torch.bfloat16)
        captured = {}

        def fake_kernel(q_nope, q_pe, kv_c, o, *args, **kwargs):
            captured["q_nope"] = q_nope.shape
            captured["q_pe"] = q_pe.shape
            captured["o"] = o.shape

        with mock.patch(_GLUON_FN, return_value=fake_kernel):
            out = mod.mla_gluon_decode(
                q=q,
                k_buffer=torch.zeros(64, 576, dtype=torch.bfloat16),
                layer=layer,
                kv_indices=torch.zeros(64, dtype=torch.int32),
                kv_indptr=torch.zeros(5, dtype=torch.int32),
                sm_scale=layer.scaling,
                min_kv_seq_len=128,
                qlen=qlen,
            )
        return out, captured

    def test_plain_decode_uses_3d(self):
        out, cap = self._call(num_tokens=4, qlen=1)
        self.assertEqual(cap["q_nope"], torch.Size([4, 12, 512]))
        self.assertEqual(cap["q_pe"], torch.Size([4, 12, 64]))
        self.assertEqual(cap["o"], torch.Size([4, 12, 512]))
        self.assertEqual(out.shape, torch.Size([4, 12, 512]))

    def test_verify_uses_4d_mtp(self):
        # 4 requests x 8 draft tokens, the DSPARK block-size-7 shape.
        out, cap = self._call(num_tokens=32, qlen=8)
        self.assertEqual(cap["q_nope"], torch.Size([4, 8, 12, 512]))
        self.assertEqual(cap["q_pe"], torch.Size([4, 8, 12, 64]))
        self.assertEqual(cap["o"], torch.Size([4, 8, 12, 512]))
        # The caller's contract is the flat [num_tokens, H, v] layout.
        self.assertEqual(out.shape, torch.Size([32, 12, 512]))

    def test_mtp_views_do_not_copy(self):
        """The 4-D q must stay a view of the caller's tensor, not a copy."""
        layer = _layer()
        q = torch.zeros(32, 12, 576, dtype=torch.bfloat16)
        seen = {}

        def fake_kernel(q_nope, q_pe, *args, **kwargs):
            seen["nope_ptr"] = q_nope.data_ptr()
            seen["pe_ptr"] = q_pe.data_ptr()

        with mock.patch(_GLUON_FN, return_value=fake_kernel):
            mod.mla_gluon_decode(
                q=q,
                k_buffer=torch.zeros(64, 576, dtype=torch.bfloat16),
                layer=layer,
                kv_indices=torch.zeros(64, dtype=torch.int32),
                kv_indptr=torch.zeros(5, dtype=torch.int32),
                sm_scale=layer.scaling,
                min_kv_seq_len=128,
                qlen=8,
            )
        self.assertEqual(seen["nope_ptr"], q.data_ptr())
        self.assertEqual(seen["pe_ptr"], q[..., 512:].data_ptr())

    def test_returns_none_when_gluon_unavailable(self):
        with mock.patch(_GLUON_FN, return_value=None):
            out = mod.mla_gluon_decode(
                q=torch.zeros(4, 12, 576, dtype=torch.bfloat16),
                k_buffer=torch.zeros(64, 576, dtype=torch.bfloat16),
                layer=_layer(),
                kv_indices=torch.zeros(64, dtype=torch.int32),
                kv_indptr=torch.zeros(5, dtype=torch.int32),
                sm_scale=0.125,
                min_kv_seq_len=128,
            )
        self.assertIsNone(out)


class TestForwardMlaDecodeDispatch(CustomTestCase):
    """_forward_mla_decode picks Gluon or the zero-pad ASM path, never both."""

    def _make_backend(self, max_q_len=1):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        be = AiterAttnBackend.__new__(AiterAttnBackend)
        be.num_head = 12
        be.kv_cache_dtype = fp8_dtype
        be.head_pad_mode = "zero"
        be.num_head_padded = 16
        be.forward_metadata = mock.Mock(
            max_q_len=max_q_len,
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

    @mock.patch(
        "sglang.srt.layers.attention.aiter_backend.prefer_mla_gluon_decode",
        return_value=False,
    )
    @mock.patch("sglang.srt.layers.attention.aiter_backend.mla_gluon_decode")
    def test_uses_asm_when_gluon_not_preferred(self, mock_gluon, _prefer):
        be = self._make_backend()
        out = be._forward_mla_decode(
            torch.zeros(4, 12, 576, dtype=torch.bfloat16),
            _layer(),
            mock.Mock(),
            k_descale=1.0,
        )
        mock_gluon.assert_not_called()
        be._mla_decode_fwd_with_head_pad.assert_called_once()
        self.assertIs(out, be._mla_decode_fwd_with_head_pad.return_value)

    @mock.patch(
        "sglang.srt.layers.attention.aiter_backend.prefer_mla_gluon_decode",
        return_value=True,
    )
    @mock.patch("sglang.srt.layers.attention.aiter_backend.mla_gluon_decode")
    def test_uses_gluon_output_when_preferred(self, mock_gluon, _prefer):
        gluon_out = torch.ones(4, 12, 512)
        mock_gluon.return_value = gluon_out
        be = self._make_backend()
        out = be._forward_mla_decode(
            torch.zeros(4, 12, 576, dtype=torch.bfloat16),
            _layer(),
            mock.Mock(),
            k_descale=1.0,
        )
        mock_gluon.assert_called_once()
        be._mla_decode_fwd_with_head_pad.assert_not_called()
        self.assertIs(out, gluon_out)

    @mock.patch(
        "sglang.srt.layers.attention.aiter_backend.prefer_mla_gluon_decode",
        return_value=True,
    )
    @mock.patch("sglang.srt.layers.attention.aiter_backend.mla_gluon_decode")
    def test_passes_max_q_len_as_qlen(self, mock_gluon, _prefer):
        """Target-verify must reach the kernel as qlen, not be silently dropped:
        the ASM fallback cannot serve this topology above qSeqLen 4 at all."""
        mock_gluon.return_value = torch.ones(32, 12, 512)
        be = self._make_backend(max_q_len=8)
        be._forward_mla_decode(
            torch.zeros(32, 12, 576, dtype=torch.bfloat16),
            _layer(),
            mock.Mock(),
            k_descale=1.0,
        )
        self.assertEqual(mock_gluon.call_args.kwargs["qlen"], 8)
        be._mla_decode_fwd_with_head_pad.assert_not_called()


if __name__ == "__main__":
    unittest.main()
