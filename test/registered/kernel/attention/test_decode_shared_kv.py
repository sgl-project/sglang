"""Decode through the grouped-head verify kernel must match `decode_attention_fwd`."""

import unittest

import torch

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="jit-kernel-unit-test-amd")

H_Q, H_KV, D = 16, 1, 128


def _run_case(seq_lens, cache_dtype, k_scale=1.0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(1)
    bs = len(seq_lens)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    total = int(lens.sum())

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device, generator=g)

    q = randn(bs, H_Q, D)
    k_new = randn(bs, H_KV, D)
    v_new = randn(bs, H_KV, D)
    k_buf = randn(total, H_KV, D)
    v_buf = randn(total, H_KV, D)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(lens, 0)
    # the last page-table slot per request is the token being generated, so kv_len_adjust=-1
    last = (kv_indptr[1:] - 1).long()
    k_buf[last] = k_new / k_scale
    v_buf[last] = v_new / k_scale
    k_buf = k_buf.to(cache_dtype)
    v_buf = v_buf.to(cache_dtype)
    kv_indices = torch.arange(total, dtype=torch.int64, device=device)
    sm_scale = D**-0.5

    o_ref = torch.empty(bs, H_Q, D, dtype=torch.bfloat16, device=device)
    max_splits = 32
    attn_logits = torch.empty(
        bs, H_Q, max_splits, D, dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(bs, H_Q, max_splits, dtype=torch.float32, device=device)
    num_kv_splits = torch.full((bs,), max_splits, dtype=torch.int32, device=device)
    decode_attention_fwd(
        q,
        k_buf,
        v_buf,
        o_ref,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_splits,
        sm_scale,
        k_scale,
        k_scale,
    )

    o = torch.empty_like(o_ref)
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=device)
    ran = verify_shared_kv_fwd(
        q,
        k_new,
        v_new,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        1,
        k_scale,
        k_scale,
        sm_scale,
        max_bs=bs,
        kv_len_adjust=-1,
    )
    return ran, o, o_ref


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestDecodeSharedKV(CustomTestCase):
    def _check(self, seq_lens, cache_dtype, atol, rtol, k_scale=1.0):
        ran, o, o_ref = _run_case(seq_lens, cache_dtype, k_scale)
        self.assertTrue(ran)
        torch.testing.assert_close(o.float(), o_ref.float(), atol=atol, rtol=rtol)

    def test_bf16_cache(self):
        self._check([1, 5, 2048, 20001], torch.bfloat16, 2e-2, 1e-2)

    def test_fp8_cache_many_requests(self):
        self._check(
            [3000 + 37 * i for i in range(24)], torch.float8_e4m3fnuz, 8e-2, 2e-2
        )

    def test_fp8_cache_scaled_long(self):
        self._check(
            [100_000, 8, 65_537], torch.float8_e4m3fnuz, 8e-2, 2e-2, k_scale=2.0
        )


if __name__ == "__main__":
    unittest.main()
