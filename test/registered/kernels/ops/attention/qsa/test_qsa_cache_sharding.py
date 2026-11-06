import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sglang.srt.layers.attention.qsa.distributed_topk as distributed_topk
from sglang.srt.layers.attention.qsa.cache_sharding import QSACacheShardingStaticBuffers
from sglang.srt.layers.attention.qsa.owner_sparse_attn import owner_sparse_attention
from sglang.srt.mem_cache.qsa_kv_pool import QSARawKVSharding
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import find_available_port

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class _Group:
    def __init__(self, rank, world_size):
        self.rank_in_group, self.world_size = rank, world_size
        self.device_group = dist.group.WORLD

    def all_to_all_single(self, output, value):
        dist.all_to_all_single(output, value)

    def all_gather(self, value, dim=0):
        values = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(values, value)
        return torch.cat(values, dim=dim)


def _capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    return graph, outputs


def _worker(rank, world_size, port):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        SGLANG_QSA_CANDIDATE_SYMM_MEM="1",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        group = _Group(rank, world_size)
        rows, topk, head_dim = 2, 64, 128
        torch.manual_seed(43527)

        def randn(*shape):
            return torch.randn(*shape, dtype=torch.bfloat16, device=device)

        def expand(values):
            return values.expand(rows, -1).contiguous()

        global_tokens = topk * world_size
        logical = expand(
            torch.arange(
                rank, global_tokens, world_size, dtype=torch.int32, device=device
            )
        )
        logits = randn(rows, topk).float()
        local_ids = expand(torch.arange(topk, dtype=torch.int32, device=device))
        row_starts = torch.zeros(rows, dtype=torch.int32, device=device)
        full_q = randn(rows, world_size, head_dim)
        local_q = full_q[:, rank : rank + 1].contiguous()
        full_k = randn(global_tokens, 1, head_dim)
        full_v = randn(global_tokens, 1, head_dim)
        local_k = full_k[rank::world_size].contiguous()
        local_v = full_v[rank::world_size].contiguous()
        buffers = QSACacheShardingStaticBuffers(
            max_rows=rows,
            topk=topk,
            num_heads=world_size,
            head_dim=head_dim,
            world_size=world_size,
            device=device,
            dtype=torch.bfloat16,
            candidate_group=group,
        )
        local_shard = QSARawKVSharding(topk, world_size, rank)
        full_shard = QSARawKVSharding(global_tokens, 1, 0)

        def publish():
            return distributed_topk.fused_publish_and_merge_qsa_topk_candidates(
                logits,
                local_ids,
                row_starts,
                logical,
                group=group,
                topk=topk,
                static_buffers=buffers,
            )

        def run():
            scores, slots = publish()
            output, lse = owner_sparse_attention(
                local_q,
                local_k,
                local_v,
                slots,
                local_shard,
                group=group,
                static_buffers=buffers,
            )
            return scores, slots, output, lse

        def reference():
            scores, slots = distributed_topk.gather_and_merge_qsa_topk_candidates(
                logits, logical, group=group, topk=topk
            )
            output, lse = owner_sparse_attention(
                full_q, full_k, full_v, slots, full_shard
            )
            return scores, slots, output[:, rank : rank + 1], lse[:, rank : rank + 1]

        def check(actual, expected):
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(actual[2], expected[2], rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(actual[3], expected[3], rtol=2e-3, atol=2e-3)

        check(run(), reference())
        dist.barrier(device_ids=[rank])
        graph, outputs = _capture(publish)
        logits.add_(rank + 0.25)
        graph.replay()
        torch.cuda.synchronize()
        expected = distributed_topk.gather_and_merge_qsa_topk_candidates(
            logits, logical, group=group, topk=topk
        )
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_qsa_multimem_owner_pipeline_eager_and_cuda_graph():
    if torch.cuda.device_count() < 4 or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("four SM90 CUDA devices are required")
    mp.spawn(_worker, args=(4, find_available_port(29500)), nprocs=4, join=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
