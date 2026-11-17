"""Hopper FA3/DCP partial attention against an unsharded FP32 reference.

A wrong log base or nonzero empty-shard output corrupts the normalized
weights even when the result remains finite. These checks pin natural-log
LSE and rank-local ownership for both split settings 0 and 1.
"""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HEADS = 16
_NOPE_DIM = 512
_ROPE_DIM = 64
_CAPACITY = 32
_DCP_SIZE = 2
_SCALE = 1.0 / math.sqrt(_NOPE_DIM + _ROPE_DIM)

# Every valid entry is packed at the front of its row. Row 0 has no KV on
# rank 1, row 1 gives both ranks one KV, and row 2 has uneven ownership.
_GLOBAL_TABLE = torch.tensor(
    [
        [0, -1, -1, -1, -1],
        [0, 1, -1, -1, -1],
        [0, 3, 2, 9, 8],
    ],
    dtype=torch.int32,
)
_LOCAL_TABLES = torch.tensor(
    [
        [
            [0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
            [0, 1, 4, -1, -1],
        ],
        [
            [-1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
            [1, 4, -1, -1, -1],
        ],
    ],
    dtype=torch.int32,
)
_LOCAL_LENGTHS = [
    [1, 1, 3],
    [0, 1, 2],
]


def _cu_seqlens(lengths):
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=lengths.device),
            lengths.cumsum(dim=0, dtype=torch.int32),
        ]
    )


def _forward_fa3(backend, q_nope, q_rope, kv, page_table, lengths, num_splits):
    backend.num_splits = num_splits
    return backend._forward_fa3(
        q_nope=q_nope,
        q_rope=q_rope,
        kv_cache=kv,
        v_head_dim=_NOPE_DIM,
        page_table=page_table,
        cache_seqlens=lengths,
        cu_seqlens_q=torch.arange(
            lengths.numel() + 1, dtype=torch.int32, device="cuda"
        ),
        cu_seqlens_k=_cu_seqlens(lengths),
        max_seqlen_q=1,
        sm_scale=_SCALE,
        logit_cap=0.0,
        page_size=1,
        return_lse=True,
    )


def _reference(q_nope, q_rope, kv):
    q = torch.cat((q_nope, q_rope), dim=-1).float()
    kv = kv.flatten(end_dim=-2).float()
    outputs, lses = [], []
    for row, page_table in enumerate(_GLOBAL_TABLE):
        keys = kv[page_table[page_table >= 0].long()]
        scores = q[row] @ keys.T * _SCALE
        outputs.append(torch.softmax(scores, dim=-1) @ keys[:, :_NOPE_DIM])
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(outputs), torch.stack(lses)


class TestFa3DcpMerge(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability(0)[0] != 9:
            raise unittest.SkipTest("The GLM-5 DCP Hopper path requires SM90")
        cls.backend = object.__new__(DeepseekSparseAttnBackend)
        cls.backend.dsa_index_kpool = 1

    def test_fa3_partials_merge_into_full_attention(self):
        generator = torch.Generator().manual_seed(0)
        q_nope = (torch.randn(3, _HEADS, _NOPE_DIM, generator=generator) * 0.2).to(
            torch.bfloat16
        )
        q_rope = (torch.randn(3, _HEADS, _ROPE_DIM, generator=generator) * 0.2).to(
            torch.bfloat16
        )
        kv = (torch.randn(_CAPACITY, 576, generator=generator) * 0.1).to(torch.bfloat16)
        expected, expected_lse = _reference(q_nope, q_rope, kv)

        q_nope, q_rope = q_nope.cuda(), q_rope.cuda()
        kv = kv.cuda().view(_CAPACITY, 1, 576)
        global_table = _GLOBAL_TABLE.cuda()
        lengths = global_table.ge(0).sum(dim=1, dtype=torch.int32)
        torch.testing.assert_close(
            lengths.cpu(), torch.tensor([1, 2, 5], dtype=torch.int32)
        )

        for num_splits in (0, 1):
            with self.subTest(num_splits=num_splits):
                baseline, baseline_lse = _forward_fa3(
                    self.backend,
                    q_nope,
                    q_rope,
                    kv,
                    global_table,
                    lengths,
                    num_splits,
                )
                torch.testing.assert_close(
                    baseline.float().cpu(), expected, atol=0.003, rtol=0.02
                )
                torch.testing.assert_close(
                    baseline_lse.cpu(), expected_lse, atol=0.001, rtol=0.001
                )

                outputs, lses = [], []
                for rank in range(_DCP_SIZE):
                    local_table, local_lengths = transform_index_page_table_dcp(
                        global_table, _DCP_SIZE, rank
                    )
                    torch.testing.assert_close(
                        local_table.cpu(), _LOCAL_TABLES[rank], atol=0, rtol=0
                    )
                    self.assertEqual(local_lengths.cpu().tolist(), _LOCAL_LENGTHS[rank])
                    output, lse = _forward_fa3(
                        self.backend,
                        q_nope,
                        q_rope,
                        kv[rank::_DCP_SIZE].contiguous(),
                        local_table,
                        local_lengths,
                        num_splits,
                    )
                    outputs.append(output)
                    lses.append(lse)

                # rank 1 owns no KV for row 0; its empty partial must not
                # contribute NaNs or weight to the result from rank 0.
                torch.testing.assert_close(
                    outputs[1][0], torch.zeros_like(outputs[1][0]), atol=0, rtol=0
                )
                self.assertTrue(torch.isneginf(lses[1][0]).all())

                merged, merged_lse = dcp_lse_combine_triton(
                    torch.stack(outputs),
                    torch.stack(lses),
                    is_lse_base_on_e=True,
                    return_lse=True,
                )
                torch.testing.assert_close(
                    merged.float().cpu(), baseline.float().cpu(), atol=0.003, rtol=0.02
                )
                torch.testing.assert_close(
                    merged.float().cpu(), expected, atol=0.003, rtol=0.02
                )
                torch.testing.assert_close(
                    merged_lse.cpu(), baseline_lse.cpu(), atol=0.001, rtol=0.001
                )
                torch.testing.assert_close(
                    merged_lse.cpu(), expected_lse, atol=0.001, rtol=0.001
                )
                self.assertTrue(torch.isfinite(merged_lse).all())

                # Both ranks own one KV in row 1. Interpreting their natural-log
                # LSEs as base-2 therefore has to produce a visible error.
                _, wrong_lse = dcp_lse_combine_triton(
                    torch.stack(outputs),
                    torch.stack(lses),
                    is_lse_base_on_e=False,
                    return_lse=True,
                )
                correct_error = (merged_lse - baseline_lse).abs().max().item()
                wrong_error = (wrong_lse.cpu() - expected_lse)[1].abs().max().item()
                self.assertGreater(wrong_error, max(0.05, 4 * correct_error))


if __name__ == "__main__":
    unittest.main()
