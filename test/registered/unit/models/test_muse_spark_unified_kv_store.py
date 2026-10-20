"""Muse unified KV pools use the attention backend's strided cache writer."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import muse_spark
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMuseSparkUnifiedKVStore(CustomTestCase):
    def test_decode_and_verify_use_prefill_cache_write_path(self):
        self.addCleanup(reset_context)
        for unified in (False, True):
            publish(
                ServerArgs(model_path="dummy", enable_unified_memory=unified),
                role="tokenizer",
            )
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.EXTEND,
            ):
                for rope in (False, True):
                    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                        with self.subTest(
                            unified=unified, mode=mode, rope=rope, dtype=dtype
                        ):
                            self._check_forward(unified, mode, rope, dtype)

    def _check_forward(self, unified, mode, rope, dtype):
        tokens, width = 3, 4
        qkv = torch.randn(tokens, 4 * width)
        q, k, _, _ = qkv.split(width, dim=-1)
        positions = torch.arange(tokens)
        batch = SimpleNamespace(forward_mode=mode, dcp_kv_mask=None)
        locations = torch.tensor([0, 3, 7])
        # Unified pools interleave layers in each token row.
        backing = torch.empty(9, 2, 16, width, dtype=dtype)
        key, value = backing[:, 0, 5], backing[:, 1, 5]
        if not unified:
            key, value = key.contiguous(), value.contiguous()
        pool = SimpleNamespace(
            dtype=dtype,
            get_key_buffer=Mock(return_value=key),
            get_value_buffer=Mock(return_value=value),
        )
        backend = muse_spark.TRTLLMHAAttnBackend.__new__(muse_spark.TRTLLMHAAttnBackend)
        backend.data_type = dtype
        backend.is_xqa_impl = False
        backend._get_layer_cache_loc = Mock(return_value=locations)
        attention = Mock(k_scale=None, v_scale=None, return_value=q)
        layer = SimpleNamespace(
            layer_idx=5,
            q_size=width,
            kv_size=width,
            use_o_gate=True,
            qkv_proj=Mock(return_value=(qkv, None)),
            rotary_emb=(
                SimpleNamespace(cos_sin_cache=torch.empty(8, width)) if rope else None
            ),
            _fused_qk_norm=Mock(return_value=(q, k)),
            attn=attention,
            _fused_o_norm_gate=Mock(return_value=q),
            o_proj=Mock(return_value=(q, None)),
        )
        layer._fused_kv_store = muse_spark.MuseSpark1xAttention._fused_kv_store.__get__(
            layer
        )
        with (
            patch.object(muse_spark, "get_attn_backend", return_value=backend),
            patch.object(muse_spark, "get_token_to_kv_pool", return_value=pool),
            patch.object(muse_spark, "is_cp_active", return_value=False),
        ):
            result = muse_spark.MuseSpark1xAttention.forward(layer, positions, q, batch)

        self.assertIs(result, q)
        store = layer._fused_qk_norm.call_args.kwargs["kv_store"]
        fallback = unified or mode == ForwardMode.EXTEND
        if fallback:
            self.assertIsNone(store)
        else:
            self.assertEqual(store[0].data_ptr(), key.data_ptr())
            self.assertEqual(store[1].data_ptr(), value.data_ptr())
            self.assertIs(store[2], locations)
        self.assertEqual(attention.call_args.kwargs["save_kv_cache"], fallback)
        self.assertIs(attention.call_args.args[3], batch)
        self.assertIs(batch.forward_mode, mode)
        if rope:
            self.assertIs(layer._fused_qk_norm.call_args.args[1], positions)


if __name__ == "__main__":
    unittest.main()
