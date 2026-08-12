"""Correctness tests for grouped-head target-verify attention."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla import verify_mla_fwd
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

ATOL = 8e-2
RTOL = 2e-2


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
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    total_prefix = sum(prefix_lens)
    batch_size = len(prefix_lens)
    num_extend_tokens = batch_size * l_ext

    q = torch.randn(num_extend_tokens, h_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_extend_tokens, 1, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_extend_tokens, 1, v_head_dim, dtype=dtype, device=device)
    k_buffer = torch.randn(total_prefix, 1, head_dim, dtype=dtype, device=device).to(
        cache_dtype
    )
    v_buffer = torch.randn(total_prefix, 1, v_head_dim, dtype=dtype, device=device).to(
        cache_dtype
    )
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
        ran = verify_mla_fwd(
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
        torch.testing.assert_close(actual, reference, atol=ATOL, rtol=RTOL)

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
            verify_mla_fwd(
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


if __name__ == "__main__":
    unittest.main()
