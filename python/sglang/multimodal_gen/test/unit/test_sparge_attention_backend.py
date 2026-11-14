import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_sparge = types.ModuleType("spas_sage_attn")
_sparge.spas_sage2_attn_meansim_topk_cuda = Mock()
_BACKEND_MODULE_NAME = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sparge_attn"
)
with patch.dict(sys.modules, {"spas_sage_attn": _sparge}):
    _backend_module = importlib.import_module(_BACKEND_MODULE_NAME)
sys.modules.pop(_BACKEND_MODULE_NAME, None)
SpargeAttentionBackend = _backend_module.SpargeAttentionBackend
SpargeAttentionImpl = _backend_module.SpargeAttentionImpl
_validate_topk = _backend_module._validate_topk


class TestSpargeAttentionBackend(unittest.TestCase):
    def _make_impl(self, *, head_size: int = 128, topk: float = 0.5):
        op = Mock()
        with (
            patch.object(
                _backend_module,
                "get_global_server_args",
                return_value=SimpleNamespace(attention_backend_config={"topk": topk}),
            ),
            patch.object(
                _backend_module,
                "spas_sage2_attn_meansim_topk_cuda",
                op,
            ),
        ):
            impl = SpargeAttentionImpl(
                num_heads=8,
                head_size=head_size,
                softmax_scale=head_size**-0.5,
            )
        return impl, op

    def test_enum_and_generic_dit_capability(self):
        self.assertEqual(str(AttentionBackendEnum.SPARGE_ATTN), "sparge_attn")
        self.assertTrue(AttentionBackendEnum.SPARGE_ATTN.is_sparse)
        self.assertIn(
            AttentionBackendEnum.SPARGE_ATTN,
            BaseDiT._supported_attention_backends,
        )
        self.assertIs(
            SpargeAttentionBackend.get_enum(), AttentionBackendEnum.SPARGE_ATTN
        )

    def test_ltx2_self_attention_head_dims_are_supported(self):
        arch = LTX2ArchConfig()

        self.assertIn(
            arch.attention_head_dim,
            SpargeAttentionBackend.get_supported_head_sizes(),
        )
        self.assertIn(
            arch.audio_attention_head_dim,
            SpargeAttentionBackend.get_supported_head_sizes(),
        )

    def test_topk_validation(self):
        self.assertEqual(_validate_topk(0.5), 0.5)
        self.assertEqual(_validate_topk(0.75), 0.75)
        for invalid in (True, "0.5", 0.0, -0.1, 1.1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "topk"):
                    _validate_topk(invalid)

    def test_head_size_validation(self):
        with self.assertRaisesRegex(ValueError, "head sizes 64 and 128"):
            self._make_impl(head_size=96)

    def test_grouped_query_attention_is_rejected(self):
        with patch.object(
            _backend_module,
            "get_global_server_args",
            return_value=SimpleNamespace(attention_backend_config={}),
        ):
            with self.assertRaisesRegex(ValueError, "grouped-query"):
                SpargeAttentionImpl(
                    num_heads=8,
                    num_kv_heads=4,
                    head_size=128,
                    softmax_scale=128**-0.5,
                )

    def test_forward_uses_nhd_layout_and_configured_topk(self):
        impl, op = self._make_impl(topk=0.75)
        qkv = SimpleNamespace(
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            shape=(1, 128, 8, 128),
            ndim=4,
        )

        expected = op.return_value.contiguous.return_value
        self.assertIs(impl.forward(qkv, qkv, qkv, None), expected)
        op.assert_called_once_with(
            qkv,
            qkv,
            qkv,
            is_causal=False,
            scale=128**-0.5,
            tensor_layout="NHD",
            topk=0.75,
        )
        op.return_value.contiguous.assert_called_once_with()

    def test_short_sequence_falls_back_to_dense_attention(self):
        impl, op = self._make_impl()
        qkv = SimpleNamespace(
            device=torch.device("cuda"),
            dtype=torch.float16,
            shape=(1, 127, 8, 128),
            ndim=4,
        )
        dense_output = object()

        with patch.object(
            _backend_module, "_dense_attention", return_value=dense_output
        ) as dense_attention:
            self.assertIs(impl.forward(qkv, qkv, qkv, None), dense_output)

        dense_attention.assert_called_once_with(
            qkv,
            qkv,
            qkv,
            softmax_scale=128**-0.5,
            causal=False,
        )
        op.assert_not_called()

    def test_asymmetric_sequence_falls_back_to_dense_attention(self):
        impl, op = self._make_impl(head_size=64)
        query = SimpleNamespace(
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            shape=(1, 256, 8, 64),
            ndim=4,
        )
        key_value = SimpleNamespace(
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            shape=(1, 512, 8, 64),
            ndim=4,
        )
        dense_output = object()

        with patch.object(
            _backend_module, "_dense_attention", return_value=dense_output
        ) as dense_attention:
            self.assertIs(impl.forward(query, key_value, key_value, None), dense_output)

        dense_attention.assert_called_once_with(
            query,
            key_value,
            key_value,
            softmax_scale=64**-0.5,
            causal=False,
        )
        op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
