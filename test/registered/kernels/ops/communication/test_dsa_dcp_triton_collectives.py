"""Sparse Hopper attention plus real DCP collectives against unsharded math."""

import os
import unittest

import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _broadcast_indexer_topk_from_rank0,
)
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)
from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mla, dcp_a2a_lse_reduce
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.multigpu import multigpu_launch
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=180, stage="extra-b", runner_config="8-gpu-h200")


class TestDsaDcpTritonCollectives(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        ps.set_custom_all_reduce(False)
        ps.init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            timeout=120,
        )
        publish_build_topology(tp_size=world, world_rank=rank)
        ps.initialize_model_parallel()
        cls.group = ps.get_tp_group()

    @classmethod
    def tearDownClass(cls):
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()

    def test_pooled_ties_share_one_selection_including_graph_replay(self):
        group = self.group
        rows, cols = 4, 1024
        scores = torch.zeros(rows, cols, device="cuda", dtype=torch.float32)
        lengths = torch.full((rows,), cols, device="cuda", dtype=torch.int32)
        picks = topk_from_pooled_history_logits(
            scores, lengths, kpool=4, topk=2048, seq_lens=lengths * 4 + 3
        )
        reference = (
            group.all_gather(picks, dim=0).view(group.world_size, rows, -1)[0].clone()
        )
        _broadcast_indexer_topk_from_rank0(picks, force=True)
        torch.testing.assert_close(picks, reference, atol=0, rtol=0)

        with group.graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                _broadcast_indexer_topk_from_rank0(picks, force=True)
        # Replace every rank's buffer independently. Graph replay must read
        # the current source rank's selection, rather than its captured data.
        picks.copy_(reference.roll(group.rank_in_group + 1, dims=1))
        graph.replay()
        torch.testing.assert_close(picks, reference.roll(1, dims=1), atol=0, rtol=0)
        local, counts = transform_index_page_table_dcp(
            picks, group.world_size, group.rank_in_group
        )
        all_counts = group.all_gather(counts, dim=0).view(group.world_size, rows)
        torch.testing.assert_close(all_counts.sum(0), (picks >= 0).sum(-1))
        valid = local >= 0
        self.assertTrue(
            (local[valid] < (cols * 4 + group.world_size) // group.world_size + 3)
            .all()
            .item()
        )

    def test_head_gather_sparse_attention_and_lse_merge(self):
        group = self.group
        dcp, rank = group.world_size, group.rank_in_group
        rows, local_heads, capacity = 3, 8, 4096
        generator = torch.Generator().manual_seed(43735)
        q = (
            torch.randn(rows, local_heads * dcp, 512, generator=generator)
            .cuda()
            .bfloat16()
        )
        kv = torch.randn(capacity, 1, 512, generator=generator).cuda().bfloat16()
        local_q = q[:, rank * local_heads : (rank + 1) * local_heads].contiguous()
        table = torch.full((rows, 2051), -1, dtype=torch.int32, device="cuda")
        # Empty row, single-owner row, expanded pooled picks plus three tail tokens.
        table[1, :127] = torch.arange(127, device="cuda") * dcp
        table[2, :2048] = torch.arange(2048, device="cuda")
        table[2, 2048:] = torch.arange(capacity - 3, capacity, device="cuda")
        local_table, counts = transform_index_page_table_dcp(table, dcp, rank)
        local_kv = kv[rank::dcp].contiguous()
        workspace = []

        def run(backend):
            gathered_q = group.all_gather(local_q, dim=1)
            out, lse = triton_sparse_mla_decode_splitk(
                gathered_q,
                gathered_q[:, :, :0],
                local_kv,
                local_table.unsqueeze(1),
                sm_scale=256**-0.5,
                d_v=512,
                kv_splits=4,
                return_lse=True,
                lengths=counts,
                workspace=workspace,
            )
            out = out.squeeze(0)
            if backend == "a2a":
                return dcp_a2a_lse_reduce(out, lse, group, is_lse_base_on_e=False)
            return cp_lse_ag_out_rs_mla(
                out, lse, group, is_lse_base_on_e=False
            ).transpose(0, 1)

        expected = torch.zeros(
            rows, local_heads, 512, device="cuda", dtype=torch.float64
        )
        for row in range(rows):
            keys = kv[table[row][table[row] >= 0].long(), 0].double()
            if len(keys):
                scores = local_q[row].double() @ keys.T * (256**-0.5)
                expected[row] = scores.softmax(-1) @ keys
        for backend in ("a2a", "ag_rs"):
            for use_graph in (False, True):
                with self.subTest(backend=backend, graph=use_graph, dcp=dcp):
                    if use_graph:
                        with group.graph_capture() as capture:
                            run(backend)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=capture.stream):
                                actual = run(backend)
                        graph.replay()
                    else:
                        actual = run(backend)
                    torch.testing.assert_close(
                        actual.double(), expected, atol=0.015, rtol=0.02
                    )


if __name__ == "__main__":
    multigpu_launch(
        __name__,
        __file__,
        num_gpus=(2, 4, 8),
        env_key="_SGLANG_TEST_DSA_DCP_COLLECTIVES",
        inner=unittest.main,
        kind="test",
        timeout=600,
    )
