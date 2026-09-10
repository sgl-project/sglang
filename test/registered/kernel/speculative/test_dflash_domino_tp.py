import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch import nn

from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_worker_v2 import _DominoDraftSampler
from sglang.srt.speculative.domino_utils import domino_greedy_rollout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _check_rollouts(rank):
    torch.manual_seed(42)
    embedding = nn.Embedding(127, 32, device="cuda", dtype=torch.bfloat16)
    prefix_gru = nn.GRU(32, 16, batch_first=True, bias=False).cuda().bfloat16()
    embed_proj = (
        nn.Sequential(
            nn.Linear(48, 16, bias=False), nn.SiLU(), nn.Linear(16, 127, bias=False)
        )
        .cuda()
        .bfloat16()
    )
    full_weight = torch.randn(127, 32, device="cuda", dtype=torch.bfloat16)
    sharded_embedding = VocabParallelEmbedding(
        127, 32, params_dtype=torch.bfloat16
    ).cuda()
    sharded_embedding.weight.weight_loader(sharded_embedding.weight, embedding.weight)
    local_weight = torch.full((64, 32), 1000, device="cuda", dtype=torch.bfloat16)
    num_org = 64 if rank == 0 else 63
    local_weight[:num_org].copy_(full_weight[rank * 64 : rank * 64 + num_org])
    group = get_tp_group()
    for pool_size in (0, 7):
        for bs in (1, 3):
            for shift_label in (False, True):
                hidden = torch.randn(bs, 16, 32, device="cuda", dtype=torch.bfloat16)
                block_ids = torch.zeros(bs, 16, device="cuda", dtype=torch.long)
                sampler = _DominoDraftSampler(
                    target_embedding=sharded_embedding,
                    lm_head_weight=local_weight,
                    prefix_gru=prefix_gru,
                    embed_proj=embed_proj,
                    vocab_size=127,
                    block_size=16,
                    shift_label=shift_label,
                    max_bs=bs,
                    candidate_pool_size=pool_size,
                    tp_group=group,
                    lm_head_org_vocab_start=rank * 64,
                    lm_head_num_org=num_org,
                    lm_head_num_org_padded=64,
                )
                with graph_capture() as capture:
                    for _ in range(3):
                        sampler(hidden.flatten(0, 1), block_ids.flatten())
                    capture.stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture.stream):
                        sampler(hidden.flatten(0, 1), block_ids.flatten())
                torch.cuda.synchronize()
                for replay in range(2):
                    hidden.copy_(torch.randn_like(hidden))
                    block_ids[:, 0].copy_(torch.randint(127, (bs,), device="cuda"))
                    kwargs = dict(
                        draft_hidden=hidden,
                        bonus_tokens=block_ids[:, 0],
                        prefix_gru=prefix_gru,
                        embed_proj=embed_proj,
                        vocab_size=127,
                        shift_label=shift_label,
                        candidate_pool_size=pool_size,
                    )
                    expected = domino_greedy_rollout(
                        **kwargs, target_embedding=embedding, lm_head_weight=full_weight
                    )
                    for compact in (False, True):
                        eager = domino_greedy_rollout(
                            **kwargs,
                            target_embedding=sharded_embedding,
                            lm_head_weight=local_weight,
                            tp_group=group,
                            lm_head_org_vocab_start=rank * 64,
                            lm_head_num_org=num_org,
                            lm_head_num_org_padded=64,
                            prefer_tp_candidate_pool=compact,
                        )
                        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                    graph.replay()
                    torch.cuda.synchronize()
                    actual = sampler.out.view(bs, 15)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    peers = group.all_gather(actual, dim=0).view(2, bs, 15)
                    torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
                    if rank == 0:
                        print(
                            f"K={pool_size}, bs={bs}, shift={shift_label}, replay={replay}: passed",
                            flush=True,
                        )


def _run_tp(rank, init_method):
    torch.cuda.set_device(rank)
    with get_context().override_server_args(tp_size=2):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
            timeout=60,
        )
        try:
            initialize_model_parallel(tensor_model_parallel_size=2)
            with get_parallel().override(tp_rank=rank):
                _check_rollouts(rank)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


class TestDFlashDominoTP(CustomTestCase):
    def test_distributed_rollout_and_graph_replay(self):
        self.assertGreaterEqual(
            torch.cuda.device_count(), 2, "Two CUDA GPUs are required"
        )
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(
                _run_tp,
                args=(Path(directory, "rendezvous").as_uri(),),
                nprocs=2,
                join=False,
            )
            try:
                deadline = time.monotonic() + 180
                while not context.join(timeout=5):
                    if time.monotonic() > deadline:
                        self.fail("Domino TP workers timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
