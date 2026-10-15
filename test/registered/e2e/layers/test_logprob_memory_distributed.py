"""Logprob memory budgets must preserve TP collectives and graph replay safety."""

import dataclasses
import sys
import traceback
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.layers import logprob_processor as logprob_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    DistributedLogprobContext,
    InputLogprobProcessor,
)
from sglang.srt.model_executor.runner_utils import pool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")


class _NcclGroup:
    def __init__(self, group):
        self.device_group = group
        self.world_size = dist.get_world_size(group)

    def all_reduce(self, value):
        dist.all_reduce(value, group=self.device_group)
        return value

    def all_gather(self, value, dim):
        outputs = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(outputs, value, group=self.device_group)
        return torch.cat(outputs, dim=dim)


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
            extend_return_top_logprob=True,
            extend_token_ids_logprob=True,
            top_logprobs_nums=[20, 0],
            extend_logprob_pruned_lens_cpu=[rows // 2, rows // 2],
            extend_input_logprob_token_ids_gpu=token_ids,
            token_ids_logprobs=[[0, vocab // 2, vocab - 1], []],
        )
        chunk_rows = []
        allocations_borrowed = []
        runs = []

        def is_borrowed(tensor):
            return any(
                lo <= tensor.data_ptr()
                and tensor.data_ptr() + tensor.nbytes <= lo + size
                for lo, size in runs
            )

        def get_gathered_logits(chunk, *_args, **_kwargs):
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
                is_borrowed(tensor) for tensor in (local, gathered, reshaped, converted)
            )
            return converted

        def get_local_logits(chunk, *_args, **_kwargs):
            chunk_rows.append(chunk.shape[0])
            local = torch.mm(chunk, local_weight.T)
            allocations_borrowed.append(is_borrowed(local))
            return local

        tp_group = _NcclGroup(nccl_group)
        distributed_context = DistributedLogprobContext(
            tp_group=tp_group,
            vocab_start_index=rank * (vocab // 2),
            vocab_end_index=(rank + 1) * (vocab // 2),
            vocab_size=vocab,
            get_local_logits_fn=get_local_logits,
            gather_sampled_logits_fn=lambda value: tp_group.all_gather(value, 1),
            local_vocab_size=vocab // 2,
            tp_size=2,
        )

        processor = InputLogprobProcessor(vocab, chunking_group=dist.group.WORLD)
        common_args = dict(
            pruned_states=states,
            sample_indices=torch.tensor(sample_rows, device=device),
            input_logprob_indices=torch.arange(rows, device=device),
            token_to_seq_idx=[0] * (rows // 2) + [1] * (rows // 2),
            lm_head=None,
            logits_metadata=metadata,
        )
        for path, fast in (
            ("gathered", False),
            ("gathered", True),
            ("distributed", True),
        ):
            processor.enable_fast_input_logprobs = fast
            processor.enable_logprobs_chunk = False
            path_args = dict(
                common_args,
                get_logits_fn=get_gathered_logits,
                distributed_context=(
                    distributed_context if path == "distributed" else None
                ),
            )
            reference, reference_sampled = processor.forward(**path_args)
            reference_output = LogitsProcessorOutput(
                next_token_logits=reference_sampled
            )
            reference.write_input_to(reference_output)
            reference_output.finalize_input_logprobs()
            reference_logprobs = reference.token_logprobs.cpu()
            reference_top_values = reference_output.input_top_logprobs_val
            reference_top_indices = reference_output.input_top_logprobs_idx
            reference_probe_values = reference_output.input_token_ids_logprobs_val
            reference_probe_indices = reference_output.input_token_ids_logprobs_idx
            processor.enable_logprobs_chunk = True
            cases = (
                (
                    (128, None, None, True),
                    (256, None, None, False),
                    (128, 1, None, False),
                )
                if path == "gathered"
                else (
                    (512, None, None, True),
                    (1024, None, None, False),
                    (512, 1, None, False),
                    (512, None, 1, False),
                )
            )
            for chunk_size, disabled_rank, missing_width_rank, borrowing in cases:
                processor.logprobs_chunk_size = chunk_size
                case_args = path_args
                if path == "distributed" and rank == missing_width_rank:
                    case_args = dict(
                        path_args,
                        distributed_context=dataclasses.replace(
                            distributed_context, local_vocab_size=None
                        ),
                    )
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
                    original_topk = logprob_module.get_distributed_topk
                    original_compact = logprob_module.get_distributed_input_state

                    def tracked_topk(local_logits, *args, **kwargs):
                        allocations_borrowed.append(is_borrowed(local_logits))
                        return original_topk(local_logits, *args, **kwargs)

                    def tracked_compact(local_logits, *args, **kwargs):
                        allocations_borrowed.append(is_borrowed(local_logits))
                        return original_compact(local_logits, *args, **kwargs)

                    with (
                        patch.object(
                            logprob_module,
                            "get_distributed_topk",
                            # A Mock call history would retain borrowed GPU tensors,
                            # creating an artificial leak/OOM before graph replay.
                            new=tracked_topk,
                        ),
                        patch.object(
                            logprob_module,
                            "get_distributed_input_state",
                            new=tracked_compact,
                        ),
                    ):
                        result, sampled = processor.forward(**case_args)
                        output = LogitsProcessorOutput(next_token_logits=sampled)
                        result.write_input_to(output)
                        # Replay before host finalization: stream ordering must
                        # protect every borrowed source and staged D2H.
                        with pool.graph_pool_replay_scope():
                            graph.replay()
                        output.finalize_input_logprobs()
                    assert torch.equal(
                        result.token_logprobs.cpu(), reference_logprobs
                    ), (
                        path,
                        chunk_size,
                        borrowing,
                        rank,
                        (result.token_logprobs.cpu() - reference_logprobs)
                        .abs()
                        .max()
                        .item(),
                    )
                    assert torch.equal(sampled, reference_sampled)
                    assert output.input_top_logprobs_val == reference_top_values
                    assert [
                        [len(row) for row in request_rows]
                        for request_rows in output.input_top_logprobs_idx
                    ] == [
                        [len(row) for row in request_rows]
                        for request_rows in reference_top_indices
                    ]
                    assert all(
                        len(set(row)) == len(row)
                        and all(0 <= token_id < vocab for token_id in row)
                        for request_rows in output.input_top_logprobs_idx
                        for row in request_rows
                    )
                    assert output.input_token_ids_logprobs_val == reference_probe_values
                    assert (
                        output.input_token_ids_logprobs_idx == reference_probe_indices
                    )
                    assert result.token_logprobs.is_pinned() == borrowing
                    if path == "distributed":
                        assert result.input_copy_done is not None
                    else:
                        assert (result.input_copy_done is not None) == borrowing
                    assert allocations_borrowed
                    assert all(
                        borrowed == borrowing for borrowed in allocations_borrowed
                    )
                    all_chunk_rows = [None, None]
                    dist.all_gather_object(all_chunk_rows, chunk_rows)
                    assert all_chunk_rows[0] == all_chunk_rows[1]
                    assert chunk_rows == [chunk_size] * (rows // chunk_size)
                    pool._teardown_borrow_pool()
                del graph, keep
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_logprob_chunks_share_tp_budget_and_survive_replay(tmp_path):
    mp.spawn(_run_rank, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
