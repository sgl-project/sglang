import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    _quantize_fp8_qkv,
    _quantize_fp8_query,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _FakeForwardBatch:
    def __init__(self, has_prefix: bool):
        self.extend_prefix_lens_cpu = [1 if has_prefix else 0]
        self.num_prefix_chunks = 1 if has_prefix else None
        self.mha_return_lse = False
        self.attn_attend_prefix_cache: bool | None = None

    def set_attn_attend_prefix_cache(self, value: bool) -> None:
        self.attn_attend_prefix_cache = value


class _FakeAttention:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, q, k, v, forward_batch, save_kv_cache=False):
        self.owner.suffix_q = q
        output = torch.zeros(
            (q.shape[0], 1, self.owner.v_head_dim), dtype=torch.bfloat16
        )
        if forward_batch.mha_return_lse:
            return output, torch.zeros((q.shape[0], 1), dtype=torch.float32)
        return output


class _FakeOutputProjection:
    def __call__(self, value):
        return value, None


class _FakeMLA:
    num_local_heads = 1
    v_head_dim = 2

    def __init__(self):
        self.attn_mha = _FakeAttention(self)
        self.o_proj = _FakeOutputProjection()
        self.suffix_q = None
        self.prefix_q = None
        self.prefix_kv_dtype = None

    def _chunked_prefix_attn_mha(
        self,
        q,
        accum_output,
        accum_lse,
        forward_batch,
        kv_a_dtype=None,
    ):
        self.prefix_q = q
        self.prefix_kv_dtype = kv_a_dtype
        return accum_output


class TestDeepseekMHAPrefillQueryReuse(CustomTestCase):
    def test_prequantized_query_matches_qkv_baseline_and_is_reused(self):
        q = torch.randn((4, 1, 3), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((4, 1, 3), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((4, 1, 2), dtype=torch.bfloat16, device="cuda")
        layer = mock.Mock(k_scale_float=1.0, v_scale_float=1.0)

        q_baseline, k_baseline, v_baseline, _, _ = _quantize_fp8_qkv(q, k, v, layer)
        q_prepared = _quantize_fp8_query(q)
        q_out, k_out, v_out, k_scale, v_scale = _quantize_fp8_qkv(
            q_prepared, k, v, layer
        )

        self.assertIs(q_out, q_prepared)
        self.assertTrue(torch.equal(q_out, q_baseline))
        self.assertTrue(torch.equal(k_out, k_baseline))
        self.assertTrue(torch.equal(v_out, v_baseline))
        self.assertEqual(k_scale, 1.0)
        self.assertEqual(v_scale, 1.0)

    def test_hybrid_backend_delegates_query_preparation_to_full_attention(self):
        q = torch.zeros((4, 1, 3), dtype=torch.bfloat16)
        q_attn = q.clone()
        full_attn_backend = mock.Mock()
        full_attn_backend.prepare_prefill_query.return_value = q_attn
        backend = object.__new__(HybridLinearAttnBackend)
        backend.full_attn_backend = full_attn_backend

        output = backend.prepare_prefill_query(q)

        full_attn_backend.prepare_prefill_query.assert_called_once_with(q)
        self.assertIs(output, q_attn)

    def test_chunked_prefill_prepares_query_once_for_suffix_and_prefix(self):
        model = _FakeMLA()
        forward_batch = _FakeForwardBatch(has_prefix=True)
        q = torch.zeros((4, 1, 3), dtype=torch.bfloat16)
        q_attn = torch.empty((4, 1, 3), dtype=torch.float8_e4m3fn)
        k = torch.zeros((4, 1, 3), dtype=torch.bfloat16)
        v = torch.zeros((4, 1, 2), dtype=torch.bfloat16)
        backend = mock.Mock()
        backend.prepare_prefill_query.return_value = q_attn

        with mock.patch.object(
            forward_mha,
            "_resolve_attn_backend",
            return_value=backend,
        ):
            output = forward_mha.DeepseekMHAForwardMixin.forward_normal_chunked_kv_core(
                model, q, k, v, forward_batch
            )

        backend.prepare_prefill_query.assert_called_once_with(q)
        self.assertIs(model.suffix_q, q_attn)
        self.assertIs(model.prefix_q, q_attn)
        self.assertEqual(model.prefix_kv_dtype, torch.bfloat16)
        self.assertEqual(output.shape, (4, 2))

    def test_no_prefix_keeps_model_query_and_skips_prepare_hook(self):
        model = _FakeMLA()
        forward_batch = _FakeForwardBatch(has_prefix=False)
        q = torch.zeros((4, 1, 3), dtype=torch.bfloat16)
        k = torch.zeros((4, 1, 3), dtype=torch.bfloat16)
        v = torch.zeros((4, 1, 2), dtype=torch.bfloat16)

        with mock.patch.object(forward_mha, "_resolve_attn_backend") as resolve_backend:
            output = forward_mha.DeepseekMHAForwardMixin.forward_normal_chunked_kv_core(
                model, q, k, v, forward_batch
            )

        resolve_backend.assert_not_called()
        self.assertIs(model.suffix_q, q)
        self.assertIsNone(model.prefix_q)
        self.assertEqual(output.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
