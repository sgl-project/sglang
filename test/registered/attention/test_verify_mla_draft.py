"""The DFLASH MLA draft's verify kernel (verify_mla_draft) must match extend_attention_fwd."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla_draft import verify_shared_kv_fwd
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
    h_kv=1,
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
    k = randn(num_extend_tokens, h_kv, head_dim)
    v = randn(num_extend_tokens, h_kv, v_head_dim)
    k_buffer = randn(total_prefix, h_kv, head_dim).to(cache_dtype)
    v_buffer = randn(total_prefix, h_kv, v_head_dim).to(cache_dtype)
    qo_indptr = torch.arange(
        0, num_extend_tokens + 1, l_ext, dtype=torch.int32, device=device
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(prefix_lens_t, dim=0)
    kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=device)
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestVerifyMlaDraft(CustomTestCase):
    def _run_parity(
        self,
        head_dim,
        v_head_dim,
        h_q=4,
        h_kv=1,
        is_causal=True,
        cache_dtype=torch.bfloat16,
        k_scale=1.0,
        v_scale=1.0,
        l_ext=4,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        prefix_lens=(512, 2048),
        sliding_window_size=-1,
        bf16_dot=False,
    ):
        inputs = _build_inputs(
            prefix_lens=list(prefix_lens),
            l_ext=l_ext,
            h_q=h_q,
            h_kv=h_kv,
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
            is_causal,
            None,
            l_ext,
            k_scale,
            v_scale,
            sm_scale=scale,
            sliding_window_size=sliding_window_size,
            bf16_dot=bf16_dot,
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
            is_causal,
            None,
            l_ext,
            k_scale,
            v_scale,
            sm_scale=scale,
            max_bs=len(kv_indptr) - 1,
            sliding_window_size=sliding_window_size,
            bf16_dot=bf16_dot,
        )

        self.assertTrue(ran)
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)

    def test_kimi_k31_dflash_mla_draft_shape(self):
        # The DFlash2 MLA draft at TP8 (12 local heads, an 8-token block, fp8 KV) over a long
        # and a short prefix: many splits a request and the combine's row blocks.
        self._run_parity(
            head_dim=576,
            v_head_dim=512,
            h_q=12,
            l_ext=8,
            prefix_lens=(65536, 3000),
            cache_dtype=torch.float8_e4m3fn,
            atol=FP8_ATOL,
            rtol=FP8_RTOL,
        )

    def test_kimi_k31_dflash_mla_draft_sliding_window(self):
        # The MLA5 draft's sliding layers (window 1024) read the window rail; rows near
        # its start are masked as extend_attention_fwd masks them. Rails shorter than,
        # equal to and past the window, 6- and 8-token blocks, bf16 dots on and off.
        for l_ext in (6, 8):
            for bf16_dot in (False, True):
                with self.subTest(l_ext=l_ext, bf16_dot=bf16_dot):
                    self._run_parity(
                        head_dim=576,
                        v_head_dim=512,
                        h_q=12,
                        l_ext=l_ext,
                        prefix_lens=(600, 1024, 1100),
                        cache_dtype=torch.float8_e4m3fn,
                        atol=FP8_ATOL,
                        rtol=FP8_RTOL,
                        sliding_window_size=1024,
                        bf16_dot=bf16_dot,
                    )

    def test_rejects_window_narrower_than_the_block(self):
        inputs = _build_inputs(
            prefix_lens=[256], l_ext=8, h_q=12, head_dim=576, v_head_dim=512
        )
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = inputs
        out = torch.empty(q.shape[0], 12, 512, dtype=q.dtype, device=q.device)
        self.assertFalse(
            verify_shared_kv_fwd(
                q, k, v, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
                None, True, None, 8, 1.0, 1.0, sliding_window_size=4,
            )
        )  # fmt: skip


if __name__ == "__main__":
    unittest.main()
