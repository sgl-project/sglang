import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV4CPKVStore(unittest.TestCase):
    def test_unified_cp_gathers_current_chunk_for_two_source_attention(self):
        layer = MQALayer.__new__(MQALayer)
        layer.fuse_wqa_wkv = False
        layer.wq_a = mock.Mock(return_value=(torch.ones(2, 2), None))
        layer.q_norm = mock.Mock(side_effect=lambda value: value)
        layer._normalize_q_lora = mock.Mock(side_effect=lambda value: (value, value))
        q = torch.ones(2, 1, 2)
        layer._compute_q_b = mock.Mock(return_value=q)
        local_kv = torch.arange(8, dtype=torch.float32).view(2, 4)
        layer._compute_kv_bf16 = mock.Mock(return_value=local_kv)
        layer.indexer = None
        layer.compressor = None
        layer.compress_ratio = 4

        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: False,
                is_target_verify=lambda: False,
            )
        )
        global_kv = torch.arange(12, dtype=torch.float32).view(3, 4)

        with (
            mock.patch.object(deepseek_v4, "_is_hip", True),
            mock.patch.object(deepseek_v4, "_is_npu", False),
            mock.patch.object(deepseek_v4, "is_cp_active", return_value=True),
            mock.patch.object(
                deepseek_v4, "get_token_to_kv_pool", return_value=object()
            ),
            mock.patch.object(
                deepseek_v4,
                "cp_materialize_global_token_order",
                return_value=global_kv,
            ) as materialize,
            mock.patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=True,
            ),
            mock.patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_fp8",
                return_value=False,
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=object()),
        ):
            returned_q, returned_kv = layer._forward_prepare(
                torch.zeros(2, 4),
                torch.arange(2),
                forward_batch,
                object(),
            )

        self.assertIs(returned_q, q)
        self.assertIs(returned_kv, global_kv)
        torch.testing.assert_close(materialize.call_args.args[0], local_kv)
        self.assertIs(materialize.call_args.args[1], forward_batch)

    def test_fused_cp_store_reuses_transformed_kv(self):
        layer = MQALayer.__new__(MQALayer)
        layer.fuse_wqa_wkv = True
        layer.q_lora_rank = 2
        layer.dsa_enable_prefill_cp = True
        layer.use_fused_qk_norm_rope = True
        layer.layer_id = 3
        layer.eps = 1e-6
        layer.qk_rope_head_dim = 2
        layer.cos_cache = torch.ones(1)
        layer.sin_cache = torch.zeros(1)
        layer.indexer = None
        layer.compressor = None
        layer.compress_ratio = 4

        qkv_a = torch.arange(24, dtype=torch.float32).view(4, 6)
        expected_kv = qkv_a[:, 2:].clone() + 100
        layer.wqkv_a = mock.Mock(return_value=(qkv_a, None))
        layer.q_norm = mock.Mock(side_effect=lambda value: value)
        layer.q_norm.weight = torch.ones(2)
        layer.q_norm.variance_epsilon = layer.eps
        layer.kv_norm = SimpleNamespace(weight=torch.ones(4))
        layer.wq_b = mock.Mock(side_effect=lambda value: (value, None))
        layer._compute_kv_bf16 = mock.Mock(
            side_effect=AssertionError("fused KV must not be transformed twice")
        )

        token_to_kv_pool = SimpleNamespace(
            get_swa_raw_buffer=mock.Mock(return_value=object()),
            swa_page_size=256,
        )
        attn_backend = SimpleNamespace(
            get_swa_out_cache_loc=mock.Mock(return_value=torch.arange(4)),
            store_cache=mock.Mock(),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: False,
                is_target_verify=lambda: False,
            )
        )

        def fused_store(*, kv, q, **_):
            kv.add_(100)
            return q

        gathered_kv = object()
        with (
            mock.patch.object(deepseek_v4, "_is_gfx95_supported", False),
            mock.patch.object(deepseek_v4, "is_cp_active", return_value=True),
            mock.patch.object(
                deepseek_v4, "get_token_to_kv_pool", return_value=token_to_kv_pool
            ),
            mock.patch.object(
                deepseek_v4,
                "cp_materialize_global_token_order",
                return_value=gathered_kv,
            ) as materialize,
            mock.patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
            mock.patch(
                "sglang.kernels.ops.attention.fused_qk_norm_rope_store.fused_qk_norm_rope_swa_store",
                side_effect=fused_store,
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=object()),
        ):
            _, returned_kv = layer._forward_prepare(
                torch.zeros(4, 6),
                torch.arange(4),
                forward_batch,
                attn_backend,
            )

        transformed_kv = materialize.call_args.args[0]
        torch.testing.assert_close(transformed_kv, expected_kv)
        layer._compute_kv_bf16.assert_not_called()
        attn_backend.store_cache.assert_called_once_with(
            layer_id=layer.layer_id,
            swa_k=gathered_kv,
            forward_batch=forward_batch,
        )
        self.assertIs(returned_kv, gathered_kv)


if __name__ == "__main__":
    unittest.main()
