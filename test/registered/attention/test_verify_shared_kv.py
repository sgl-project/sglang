"""Correctness tests for grouped-head target-verify attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
from sglang.srt.layers.attention.triton_backend import (
    _should_use_verify_shared_kv,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

BF16_ATOL = 2e-2
BF16_RTOL = 1e-2
FP8_ATOL = 8e-2
FP8_RTOL = 2e-2


def _build_inputs(
    prefix_lens,
    l_ext,
    h_q,
    head_dim,
    v_head_dim,
    cache_dtype=torch.bfloat16,
):
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(0)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    total_prefix = sum(prefix_lens)
    batch_size = len(prefix_lens)
    num_extend_tokens = batch_size * l_ext

    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

    q = randn(num_extend_tokens, h_q, head_dim)
    k = randn(num_extend_tokens, 1, head_dim)
    v = randn(num_extend_tokens, 1, v_head_dim)
    k_buffer = randn(total_prefix, 1, head_dim).to(cache_dtype)
    v_buffer = randn(total_prefix, 1, v_head_dim).to(cache_dtype)
    qo_indptr = torch.arange(
        0, num_extend_tokens + 1, l_ext, dtype=torch.int32, device=device
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(prefix_lens_t, dim=0)
    kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=device)
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestVerifySharedKV(CustomTestCase):
    def _run_parity(
        self,
        head_dim,
        v_head_dim,
        h_q=4,
        cache_dtype=torch.bfloat16,
        k_scale=1.0,
        v_scale=1.0,
        l_ext=4,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    ):
        inputs = _build_inputs(
            prefix_lens=[512, 2048],
            l_ext=l_ext,
            h_q=h_q,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            cache_dtype=cache_dtype,
        )
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = inputs
        output_shape = (q.shape[0], q.shape[1], v_head_dim)
        reference = torch.empty(output_shape, dtype=q.dtype, device=q.device)
        actual = torch.empty_like(reference)
        scale = head_dim**-0.5

        extend_attention_fwd(
            q,
            k,
            v,
            reference,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            None,
            l_ext,
            k_scale,
            v_scale,
            sm_scale=scale,
        )
        ran = verify_shared_kv_fwd(
            q,
            k,
            v,
            actual,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            None,
            l_ext,
            k_scale,
            v_scale,
            sm_scale=scale,
            max_bs=len(kv_indptr) - 1,
        )

        self.assertTrue(ran)
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)

    def test_qwen3_5_tp_shapes(self):
        # Qwen3.5 has 32 global query heads. TP8, TP4, and InferenceX's TP2
        # configurations expose 4, 8, and 16 local query heads respectively,
        # all sharing one TP-local KV head.
        for h_q in (4, 8, 16):
            with self.subTest(h_q=h_q):
                self._run_parity(head_dim=256, v_head_dim=256, h_q=h_q)

    def test_qwen3_5_short_verify_widths(self):
        for l_ext in (1, 2, 3):
            with self.subTest(l_ext=l_ext):
                self._run_parity(head_dim=256, v_head_dim=256, l_ext=l_ext)

    def test_qwen3_5_fp8_kv_cache(self):
        self._run_parity(
            head_dim=256,
            v_head_dim=256,
            h_q=8,
            cache_dtype=torch.float8_e4m3fn,
            k_scale=0.5,
            v_scale=0.25,
            atol=FP8_ATOL,
            rtol=FP8_RTOL,
        )

    def test_kimi_k3_absorbed_mla_shape(self):
        self._run_parity(head_dim=576, v_head_dim=512)

    def test_rejects_multiple_local_kv_heads(self):
        inputs = list(
            _build_inputs(
                prefix_lens=[512],
                l_ext=4,
                h_q=4,
                head_dim=256,
                v_head_dim=256,
            )
        )
        for index in (1, 2, 3, 4):
            inputs[index] = inputs[index].expand(-1, 2, -1).contiguous()
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = inputs
        output = torch.empty_like(q)
        self.assertFalse(
            verify_shared_kv_fwd(
                q,
                k,
                v,
                output,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                True,
                None,
                4,
                1.0,
                1.0,
            )
        )

    @patch(
        "sglang.srt.layers.attention.triton_backend.is_gfx95_supported",
        return_value=True,
    )
    @patch("sglang.srt.layers.attention.triton_backend.get_parallel")
    def test_backend_dispatch_gate(self, get_parallel_mock, _is_gfx95_mock):
        get_parallel_mock.return_value = SimpleNamespace(
            attn_tp_size=8, attn_dcp_size=1
        )

        def model_config(architecture, local_kv_heads=1):
            return SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture]),
                get_num_kv_heads=lambda _tp, _dcp: local_kv_heads,
            )

        qwen = model_config("Qwen3_5MoeForCausalLM")
        self.assertTrue(_should_use_verify_shared_kv(qwen, 1, False, True))
        self.assertFalse(_should_use_verify_shared_kv(qwen, 2, False, True))
        self.assertFalse(_should_use_verify_shared_kv(qwen, 1, False, False))
        self.assertFalse(
            _should_use_verify_shared_kv(
                model_config("Qwen3_5MoeForCausalLM", local_kv_heads=2),
                1,
                False,
                True,
            )
        )
        self.assertFalse(
            _should_use_verify_shared_kv(
                model_config("LlamaForCausalLM"), 1, False, True
            )
        )
        self.assertTrue(
            _should_use_verify_shared_kv(
                model_config("KimiK3ForConditionalGeneration"), 1, True, False
            )
        )
        with patch(
            "sglang.srt.layers.attention.triton_backend.is_gfx95_supported",
            return_value=False,
        ):
            self.assertFalse(_should_use_verify_shared_kv(qwen, 1, False, True))


if __name__ == "__main__":
    unittest.main()
