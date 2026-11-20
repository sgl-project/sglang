"""Hopper DCP oracle: expanded pooled picks must partition without losing tails.

This tests the sparse attention/localization/merge math, independently of the
model's indexer and cache allocator. Model integration needs separate coverage.
"""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)
from sglang.srt.layers.attention.dsa.dsa_dcp import (
    DcpPageTableCache,
    localize_dcp_page_table,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference(q, kv, table, scale, value_dim):
    # Float64 reference uses the original token picks, with no DCP remapping or
    # split-K reduction. An empty attention row has output zero and LSE -inf.
    output = torch.zeros(
        (*q.shape[:2], value_dim), device=q.device, dtype=torch.float64
    )
    lse = torch.full(q.shape[:2], -torch.inf, device=q.device, dtype=torch.float64)
    for row in range(q.shape[0]):
        slots = table[row][table[row] >= 0].long()
        if slots.numel():
            keys = kv[slots, 0].double()
            scores = q[row].double() @ keys.T * scale
            output[row] = scores.softmax(-1) @ keys[:, :value_dim]
            lse[row] = scores.logsumexp(-1) / math.log(2)
    return output.float(), lse.float()


def _attention(q, kv, table, counts, splits):
    return triton_sparse_mla_decode_splitk(
        q[:, :, :512],
        q[:, :, 512:],
        kv,
        table.unsqueeze(1),
        sm_scale=256**-0.5,
        d_v=512,
        kv_splits=splits,
        return_lse=True,
        lengths=counts,
        workspace=[],
    )


@unittest.skipUnless(
    torch.cuda.is_available() and not torch.version.hip, "requires CUDA"
)
class TestDsaDcpTriton(CustomTestCase):
    def _check_lse(self, actual, expected):
        valid = torch.isfinite(expected)
        torch.testing.assert_close(
            actual[valid], expected[valid], atol=0.003, rtol=0.002
        )
        # The existing collective path represents empty shards with a finite
        # floor to avoid -inf - -inf. Its exponential must still be exactly zero.
        self.assertFalse(torch.isnan(actual).any().item())
        self.assertTrue((actual[~valid].exp2() == 0).all().item())

    def test_no_rope_prefill_small_and_large_batches(self):
        """The non-DCP control must also support the original Flash layout."""
        torch.manual_seed(43735)
        q = torch.randn(3, 8, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
        table = torch.full((3, 2051), -1, device="cuda", dtype=torch.int32)
        table[1, 0] = 4095
        table[2] = torch.arange(2051, device="cuda")
        expected, _ = _reference(q, kv, table, 256**-0.5, 512)
        # The small batch selects split-K; the large batch selects the
        # single-pass prefill kernel on H200 (132 SMs).
        for repeats in (1, 128):
            query = q.repeat(repeats, 1, 1)
            actual = triton_sparse_mla_fwd(
                query,
                query[:, :, :0],
                kv,
                table.repeat(repeats, 1).unsqueeze(1),
                sm_scale=256**-0.5,
            ).squeeze(0)
            torch.testing.assert_close(
                actual.float(), expected.repeat(repeats, 1, 1), atol=0.015, rtol=0.02
            )

    def test_graph_replay_updates_pooled_picks_and_owned_counts(self):
        """Captured localization must not freeze counts or read stale workspace."""
        torch.manual_seed(43735)
        rows, heads, capacity, dcp = 3, 16, 4096, 2
        q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(capacity, 1, 512, device="cuda", dtype=torch.bfloat16)
        local_kv = [kv[rank::dcp].contiguous() for rank in range(dcp)]
        table = torch.full((rows, 2051), -1, device="cuda", dtype=torch.int32)
        workspaces = [[] for _ in range(dcp)]
        caches = [DcpPageTableCache() for _ in range(dcp)]
        metadata = object()

        def forward(query):
            outputs, lses = [], []
            for rank in range(dcp):
                local, counts = localize_dcp_page_table(
                    caches[rank],
                    metadata=metadata,
                    layer_id=3,
                    topk_indices=table,
                    page_table=table,
                    dcp_size=dcp,
                    dcp_rank=rank,
                )
                out, lse = triton_sparse_mla_decode_splitk(
                    query[:, :, :512],
                    query[:, :, 512:],
                    local_kv[rank],
                    local.unsqueeze(1),
                    sm_scale=256**-0.5,
                    d_v=512,
                    kv_splits=4,
                    lengths=counts,
                    return_lse=True,
                    workspace=workspaces[rank],
                )
                outputs.append(out.squeeze(0))
                lses.append(lse)
            return dcp_lse_combine_triton(
                torch.stack(outputs),
                torch.stack(lses),
                is_lse_base_on_e=False,
                return_lse=True,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward(q)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual, actual_lse = forward(q)

        # Grow the same backend-owned workspaces after capture. Previously
        # captured storage must remain alive while the larger variant runs.
        forward(q.repeat(1, 4, 1))
        for step in range(4):
            table.fill_(-1)
            if step == 1:
                table[0, :2051] = torch.arange(2051, device="cuda")
                table[1, :127] = torch.arange(127, device="cuda") * 2
                table[2, 0] = capacity - 1
            elif step == 2:
                table[0, :129] = torch.arange(129, device="cuda") * 2 + 1
                table[2, :2048] = torch.randperm(capacity, device="cuda")[:2048]
            q.normal_()
            kv.normal_()
            for rank in range(dcp):
                local_kv[rank].copy_(kv[rank::dcp])
            graph.replay()
            expected, expected_lse = _reference(q, kv, table, 256**-0.5, 512)
            with self.subTest(step=step):
                torch.testing.assert_close(
                    actual.float(), expected, atol=0.015, rtol=0.02
                )
                self._check_lse(actual_lse, expected_lse)

    def test_pooled_picks_and_empty_shards_match_reference(self):
        """Lose a tail, use local ids globally, or merge natural LSE: this fails."""
        generator = torch.Generator().manual_seed(43735)
        capacity, rows = 4096, 4
        for kpool, heads, rope in ((1, 8, 64), (2, 16, 0), (4, 32, 0), (8, 64, 0)):
            width = 2048 + kpool - 1
            table = torch.full((rows, width), -1, dtype=torch.int32)
            table[1, 0] = capacity - 1
            # One owner only, including a partial final kernel tile.
            table[2, :127] = torch.arange(127) * 8
            groups = torch.randperm(capacity // kpool - 1, generator=generator)[
                : 2048 // kpool
            ]
            table[3, :2048] = (groups[:, None] * kpool + torch.arange(kpool)).flatten()
            table[3, 2048:] = torch.arange(capacity - kpool, capacity - 1)
            table = table.cuda()
            q = (
                torch.randn(rows, heads, 512 + rope, generator=generator)
                .cuda()
                .bfloat16()
            )
            kv = (
                torch.randn(capacity, 1, 512 + rope, generator=generator)
                .cuda()
                .bfloat16()
            )
            expected, expected_lse = _reference(q, kv, table, 256**-0.5, 512)
            for splits in (1, 4):
                whole, whole_lse = _attention(q, kv, table, None, splits)
                torch.testing.assert_close(
                    whole.squeeze(0).float(), expected, atol=0.015, rtol=0.02
                )
                self._check_lse(whole_lse, expected_lse)
                for dcp in (2, 4, 8):
                    with self.subTest(
                        kpool=kpool, heads=heads, rope=rope, splits=splits, dcp=dcp
                    ):
                        outputs, lses = [], []
                        total_counts = torch.zeros(
                            rows, device="cuda", dtype=torch.int32
                        )
                        for rank in range(dcp):
                            local, counts = transform_index_page_table_dcp(
                                table, dcp, rank
                            )
                            total_counts += counts
                            out, lse = _attention(
                                q, kv[rank::dcp].contiguous(), local, counts, splits
                            )
                            self.assertTrue(torch.isfinite(out).all().item())
                            self.assertFalse(torch.isnan(lse).any().item())
                            outputs.append(out.squeeze(0))
                            lses.append(lse)
                        torch.testing.assert_close(
                            total_counts, (table >= 0).sum(-1).int()
                        )
                        merged, merged_lse = dcp_lse_combine_triton(
                            torch.stack(outputs),
                            torch.stack(lses),
                            is_lse_base_on_e=False,
                            return_lse=True,
                        )
                        torch.testing.assert_close(
                            merged.float(), expected, atol=0.015, rtol=0.02
                        )
                        self._check_lse(merged_lse, expected_lse)


if __name__ == "__main__":
    unittest.main()
