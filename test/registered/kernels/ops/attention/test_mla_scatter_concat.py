"""MLA scatter and query concatenation parity."""

import unittest

import torch

from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q,
    can_use_set_mla_kv_concat_q_fp8,
    set_mla_kv_concat_q,
    set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


NOPE_DIM = 512

ROPE_DIM = 64

MLA_DIM = NOPE_DIM + ROPE_DIM

MLA_PAGES = 256


def _make_mla_inputs(batch_size, num_heads, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return (
            torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)
            .mul(0.1)
            .to(torch.bfloat16)
        )

    pool = randn(MLA_PAGES, MLA_DIM)
    latent = randn(batch_size, MLA_DIM)
    query = randn(batch_size, num_heads, MLA_DIM)
    loc = (
        torch.randperm(MLA_PAGES - 1, generator=generator, device="cuda")[:batch_size]
        + 1
    ).to(torch.int64)
    return (
        pool,
        loc,
        latent[:, :NOPE_DIM],
        latent[:, NOPE_DIM:],
        query[..., :NOPE_DIM],
        query[..., NOPE_DIM:],
    )


class TestMLAScatterConcat(CustomTestCase):
    def test_strided_absorbed_q_and_graph(self):
        if not can_use_set_mla_kv_concat_q_fp8():
            self.skipTest("fused FP8 MLA scatter+concat requires SM90+")
        for batch_size in (1, 7, 64):
            with self.subTest(batch_size=batch_size):
                pool, loc, k_nope, k_rope, _, q_rope = _make_mla_inputs(
                    batch_size, 4, seed=1
                )
                q = torch.randn(batch_size, 4, 128, device="cuda", dtype=torch.bfloat16)
                w_kc = torch.randn(
                    4, 128, NOPE_DIM, device="cuda", dtype=torch.bfloat16
                )
                absorbed = torch.bmm(q.transpose(0, 1), w_kc)
                q_nope = absorbed.transpose(0, 1)
                self.assertEqual(q_nope.data_ptr(), absorbed.data_ptr())
                if batch_size > 1:
                    self.assertFalse(q_nope.is_contiguous())
                expected = torch.cat((q_nope, q_rope), dim=-1)
                torch.testing.assert_close(
                    concat_mla_absorb_q(q_nope, q_rope), expected
                )
                result = set_mla_kv_concat_q(pool, loc, k_nope, k_rope, q_nope, q_rope)
                torch.testing.assert_close(result, expected)
                fp8_pool = torch.zeros_like(pool, dtype=torch.float8_e4m3fn)

                def concat():
                    return set_mla_kv_concat_q_fp8(
                        fp8_pool,
                        loc,
                        k_nope,
                        k_rope,
                        q_nope,
                        q_rope,
                        dcp_world_size=2,
                        dcp_rank=1,
                    )

                concat()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = concat()
                q_nope.mul_(0.5)
                graph.replay()
                expected = torch.cat((q_nope, q_rope), dim=-1).to(torch.float8_e4m3fn)
                self.assertTrue(
                    torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
                )
                owned = loc % 2 == 1
                expected_pool = torch.zeros_like(fp8_pool)
                expected_pool[loc[owned] // 2] = torch.cat(
                    (k_nope[owned], k_rope[owned]), dim=-1
                ).to(torch.float8_e4m3fn)
                self.assertTrue(
                    torch.equal(
                        fp8_pool.view(torch.uint8), expected_pool.view(torch.uint8)
                    )
                )

    def test_mla_scatter_concat_bf16_and_fp8(self):
        batch_size, num_heads = 64, 8
        pool, loc, k_nope, k_rope, q_nope, q_rope = _make_mla_inputs(
            batch_size, num_heads, seed=0
        )

        if not can_use_set_mla_kv_concat_q(NOPE_DIM * 2, ROPE_DIM * 2):
            self.skipTest("fused MLA scatter+concat requires SM90+")
        pool_ref = pool.clone()
        query = set_mla_kv_concat_q(pool, loc, k_nope, k_rope, q_nope, q_rope)
        set_mla_kv_buffer(pool_ref, loc, k_nope, k_rope)
        query_ref = concat_mla_absorb_q(q_nope, q_rope)
        self.assertTrue(torch.equal(pool, pool_ref))
        self.assertTrue(torch.equal(query, query_ref))

        if not can_use_set_mla_kv_concat_q_fp8():
            self.skipTest("fused FP8 MLA scatter+concat requires SM90+")
        fp8_pool = torch.zeros(
            MLA_PAGES, MLA_DIM, device="cuda", dtype=torch.float8_e4m3fn
        )
        fp8_ref = fp8_pool.clone()
        fp8_query = set_mla_kv_concat_q_fp8(
            fp8_pool, loc, k_nope, k_rope, q_nope, q_rope
        )
        row = torch.cat([k_nope, k_rope], dim=-1).to(torch.float8_e4m3fn)
        fp8_ref[loc] = row
        fp8_query_ref = concat_mla_absorb_q_general(q_nope, q_rope).to(
            torch.float8_e4m3fn
        )
        self.assertTrue(
            torch.equal(fp8_pool.view(torch.uint8), fp8_ref.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(
                fp8_query.view(torch.uint8),
                fp8_query_ref.view(torch.uint8),
            )
        )


if __name__ == "__main__":
    unittest.main()
