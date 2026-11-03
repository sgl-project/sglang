# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for native SM90 SubBlock Sage FP8 attention."""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _native_sm90_sage_available() -> bool:
    try:
        import spas_sage_attn._qattn as qattn
    except (ImportError, OSError):
        return False
    return hasattr(
        qattn,
        "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90",
    )


requires_native_sm90_sage = unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (9, 0)
    and _native_sm90_sage_available(),
    "requires SM90 and a compiled SpargeAttention installation",
)


def _masked_reference(q, k, v, index, counts, scale, key_block_size=128):
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            for qb in range(index.shape[2]):
                q_slice = slice(qb * 64, min((qb + 1) * 64, q.shape[1]))
                for slot in range(int(counts[b, h, qb])):
                    kb = int(index[b, h, qb, slot])
                    k_slice = slice(
                        kb * key_block_size,
                        min((kb + 1) * key_block_size, k.shape[1]),
                    )
                    mask[b, h, q_slice, k_slice] = True
    logits.masked_fill_(~mask, -float("inf"))
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, v.float()).to(torch.bfloat16)


def _cosine(a, b):
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        )
    )


@requires_native_sm90_sage
class TestSubBlockSageFp8NativeSm90(CustomTestCase):
    def test_full_budget_ragged_tail_reproduces_dense(self):
        from sglang.kernels.ops.attention.subblock_sage_fp8_sm90 import (
            subblock_sage_fp8_sm90_attention,
        )

        torch.manual_seed(19)
        seq_len = 1024 + 37
        heads = 4
        shape = (1, seq_len, heads, 128)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scale = 1.0 / math.sqrt(128)

        query_blocks = math.ceil(seq_len / 64)
        key_blocks = math.ceil(seq_len / 128)
        index = (
            torch.arange(key_blocks, device="cuda", dtype=torch.int32)
            .view(1, 1, 1, key_blocks)
            .expand(1, heads, query_blocks, key_blocks)
            .contiguous()
        )
        output = subblock_sage_fp8_sm90_attention(q, k, v, index, key_blocks, scale)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=scale,
        ).transpose(1, 2)

        self.assertTrue(torch.isfinite(output.float()).all())
        self.assertGreater(_cosine(output, reference), 0.998)

    def test_sparse_k128_plan_and_variable_counts(self):
        from sglang.kernels.ops.attention.subblock_sage_fp8_sm90 import (
            subblock_sage_fp8_sm90_attention,
        )

        torch.manual_seed(23)
        seq_len = 1024 + 37
        heads = 4
        shape = (1, seq_len, heads, 128)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scale = 1.0 / math.sqrt(128)

        query_blocks = math.ceil(seq_len / 64)
        key_blocks = math.ceil(seq_len / 128)
        width = 4
        index = torch.stack(
            [
                torch.roll(torch.arange(key_blocks), shifts=query_block)[:width]
                for query_block in range(query_blocks)
            ]
        )
        index = (
            index.to(device="cuda", dtype=torch.int32)
            .view(1, 1, query_blocks, width)
            .expand(1, heads, query_blocks, width)
            .contiguous()
        )
        counts = (
            ((torch.arange(query_blocks, device="cuda", dtype=torch.int32) % width) + 1)
            .view(1, 1, query_blocks)
            .expand(1, heads, query_blocks)
            .contiguous()
        )

        output = subblock_sage_fp8_sm90_attention(q, k, v, index, width, scale, counts)
        reference = _masked_reference(q, k, v, index, counts, scale)

        self.assertTrue(torch.isfinite(output.float()).all())
        self.assertGreater(_cosine(output, reference), 0.997)


if __name__ == "__main__":
    unittest.main(verbosity=3)
