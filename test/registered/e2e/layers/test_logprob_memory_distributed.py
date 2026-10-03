"""Logprob memory budgets must preserve TP collectives and graph replay safety."""

import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import InputLogprobProcessor
from sglang.srt.model_executor.runner_utils import pool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")


def _run_rank(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    nccl_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(0)
        device = torch.device("cuda", rank)
        rows, vocab = 2048, 202752
        states = torch.randint(-2, 3, (rows, 64), device=device).to(torch.bfloat16)
        weight = torch.randint(-2, 3, (vocab, 64), device=device).to(torch.bfloat16)
        local_weight = weight.chunk(2)[rank].contiguous()
        token_ids = torch.randint(vocab, (rows,), device=device)
        sample_rows = [rows // 2 - 1, rows - 1]
        metadata = SimpleNamespace(
            sample_indices_cpu=sample_rows,
            input_logprob_indices_cpu=list(range(rows)),
            extend_return_top_logprob=False,
            extend_token_ids_logprob=False,
            top_logprobs_nums=None,
            extend_logprob_pruned_lens_cpu=[rows // 2, rows // 2],
            extend_input_logprob_token_ids_gpu=token_ids,
            token_ids_logprobs=None,
        )
        chunk_rows = []
        allocations_borrowed = []
        runs = []

        def get_logits(chunk, *_args, **_kwargs):
            chunk_rows.append(chunk.shape[0])
            local = torch.mm(chunk, local_weight.T)
            gathered = torch.empty(
                (2 * chunk.shape[0], vocab // 2), device=device, dtype=local.dtype
            )
            dist.all_gather_into_tensor(gathered, local, group=nccl_group)
            reshaped = (
                gathered.reshape(2, chunk.shape[0], vocab // 2)
                .movedim(0, 1)
                .reshape(chunk.shape[0], vocab)
            )
            converted = reshaped.float()
            allocations_borrowed.extend(
                any(
                    lo <= tensor.data_ptr()
                    and tensor.data_ptr() + tensor.nbytes <= lo + size
                    for lo, size in runs
                )
                for tensor in (local, gathered, reshaped, converted)
            )
            return converted

        processor = InputLogprobProcessor(vocab, chunking_group=dist.group.WORLD)
        args = dict(
            pruned_states=states,
            sample_indices=torch.tensor(sample_rows, device=device),
            input_logprob_indices=torch.arange(rows, device=device),
            token_to_seq_idx=[0] * (rows // 2) + [1] * (rows // 2),
            lm_head=None,
            get_logits_fn=get_logits,
            logits_metadata=metadata,
        )
        for fast in (False, True):
            processor.enable_fast_input_logprobs = fast
            processor.enable_logprobs_chunk = False
            reference, reference_sampled = processor.forward(**args)
            reference_logprobs = reference.token_logprobs.cpu()
            processor.enable_logprobs_chunk = True
            for chunk_size, disabled_rank, borrowing in (
                (128, None, True),
                (256, None, False),
                (128, 1, False),
            ):
                processor.logprobs_chunk_size = chunk_size
                state = pool.GraphPoolBorrowState()
                handle = torch.cuda.graph_pool_handle()
                graph = torch.cuda.CUDAGraph()
                seed = torch.zeros(8, device=device)
                stream = torch.cuda.Stream()
                with (
                    torch.cuda.stream(stream),
                    torch.cuda.graph(graph, pool=handle, stream=stream),
                ):
                    transient = torch.empty(
                        (512 + rank * 256) << 20, dtype=torch.uint8, device=device
                    )
                    transient.fill_(7)
                    keep = seed + 1
                    del transient
                torch.cuda.synchronize()
                state.disabled_reason = (
                    "test fallback" if rank == disabled_rank else None
                )
                with (
                    pool.get_resources().override(graph_pool_borrow=state),
                    envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(True),
                    patch.object(
                        pool, "get_global_graph_memory_pool", return_value=handle
                    ),
                    patch.object(
                        torch.cuda,
                        "mem_get_info",
                        side_effect=AssertionError(
                            "Chunk sizing must not query heap headroom"
                        ),
                    ),
                ):
                    chunk_rows.clear()
                    allocations_borrowed.clear()
                    runs = pool.find_free_graph_pool_runs(handle)
                    result, sampled = processor.forward(**args)
                    with pool.graph_pool_replay_scope():
                        graph.replay()
                    if result.input_copy_done is not None:
                        result.input_copy_done.synchronize()
                    assert torch.equal(result.token_logprobs.cpu(), reference_logprobs)
                    assert torch.equal(sampled, reference_sampled)
                    assert (result.input_copy_done is not None) == borrowing
                    assert all(
                        borrowed == borrowing for borrowed in allocations_borrowed
                    )
                    all_chunk_rows = [None, None]
                    dist.all_gather_object(all_chunk_rows, chunk_rows)
                    assert all_chunk_rows[0] == all_chunk_rows[1]
                    assert chunk_rows == [chunk_size] * (rows // chunk_size)
                    pool._teardown_borrow_pool()
                del graph, keep
    finally:
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_logprob_chunks_share_tp_budget_and_survive_replay(tmp_path):
    mp.spawn(_run_rank, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
