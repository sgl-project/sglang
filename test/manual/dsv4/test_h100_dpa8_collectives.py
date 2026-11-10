"""Eight-rank H100 component checks, outside any serving measurement.

Run with torchrun --standalone --nproc-per-node=8 and the candidate PYTHONPATH.
DSV41_TEST_MODEL points at the unmodified checkpoint; DSV41_TEST_OUTPUT is a
private output directory. These are component checks, not quality equivalence.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.pop("SGLANG_RAGGED_VERIFY_MODE", None)

import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers import dp_attention as dp
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish
from sglang.srt.server_args import ServerArgs


def check_close(actual, expected):
    difference = actual.float() - expected.float()
    result = {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (
            difference.norm() / expected.float().norm().clamp_min(1e-20)
        ).item(),
    }
    # BF16 output can change GEMM tactic when all ranks' rows are coalesced.
    # Keep this explicit bound; do not weaken it after observing a failure.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-6, atol=2**-4)
    return result


def main():
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (9, 0)
    model = Path(os.environ["DSV41_TEST_MODEL"])
    output = Path(os.environ["DSV41_TEST_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    args = ServerArgs(
        model_path=str(model),
        trust_remote_code=True,
        device="cuda",
        tp_size=8,
        attn_dp_size=8,
        ep_size=8,
        attn_cp_size=1,
        attention_backend="dsv4",
        moe_runner_backend="flashinfer_mxfp4",
        flashinfer_mxfp4_moe_precision="fp8",
        speculative_algorithm="DSPARK",
        enable_dp_lm_head=True,
    )
    publish(args, role="test", ranks=SpawnRanks(world_rank=rank, gpu_id=rank))
    ps.init_distributed_environment(
        world_size=8, rank=rank, local_rank=rank, backend="nccl", timeout=180
    )
    ps.initialize_model_parallel()
    dp.initialize_dp_attention(args)
    dp.set_is_extend_in_batch(False)
    group = get_parallel().tp_group
    results = []

    def record(item):
        results.append(item)
        with (output / f"rank-{rank}-journal.jsonl").open("a") as journal:
            journal.write(json.dumps(item) + "\n")

    # Copy and reduce against an independent native NCCL reference, across
    # workspace reuse and the largest supported short verification buffer.
    for rows in (1, 5, 6, 24, 48):
        local = torch.empty((rows, 5120), device="cuda", dtype=torch.bfloat16)
        gathered = torch.empty((8 * rows, 5120), device="cuda", dtype=torch.bfloat16)
        reference = torch.empty_like(gathered)
        texture = (
            torch.arange(rows, device="cuda")[:, None] % 4
            + torch.arange(5120, device="cuda")[None, :] % 4
        )
        for turn in range(3):
            local.copy_(texture + rank * 3 + turn)
            dist.all_gather_into_tensor(reference, local, group=group.device_group)
            assert dp._try_hopper_dp_all_gather(gathered, local), (
                "eligible H100 gather declined"
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(gathered, reference, rtol=0, atol=0)
        with group.graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                assert dp._try_hopper_dp_all_gather(gathered, local)
        for turn in range(3):
            local.copy_(texture + rank + turn + 7)
            dist.all_gather_into_tensor(reference, local, group=group.device_group)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(gathered, reference, rtol=0, atol=0)
        reduce_texture = (
            torch.arange(8 * rows, device="cuda")[:, None] % 13
            + torch.arange(5120, device="cuda")[None, :] % 5
        )
        source = (reduce_texture + rank).to(torch.bfloat16)
        reduced, expected = torch.empty_like(local), torch.empty_like(local)
        dist.reduce_scatter_tensor(expected, source.clone(), group=group.device_group)
        with patch.object(
            group, "reduce_scatterv", wraps=group.reduce_scatterv
        ) as routed:
            dp.dp_reduce_scatterv(reduced, source, [rows] * 8)
            assert routed.call_args.kwargs["sizes"] is None
        torch.cuda.synchronize()
        torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
        with group.graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                dp.dp_reduce_scatterv(reduced, source, [rows] * 8)
        for turn in range(3):
            source.copy_(reduce_texture + rank + turn)
            dist.reduce_scatter_tensor(
                expected, source.clone(), group=group.device_group
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
        record({"component": "gather_reduce", "rows_per_rank": rows, "passed": True})

    sizes = [1, 2, 0, 3, 1, 0, 1, 2]
    source = (
        torch.arange(sum(sizes), device="cuda")[:, None] % 13
        + torch.arange(5120, device="cuda")[None, :] % 5
        + rank
    ).to(torch.bfloat16)
    summed = source.clone()
    dist.all_reduce(summed, group=group.device_group)
    reduced = torch.empty((sizes[rank], 5120), device="cuda", dtype=torch.bfloat16)
    with patch.object(group, "reduce_scatterv", wraps=group.reduce_scatterv) as routed:
        dp.dp_reduce_scatterv(reduced, source, sizes)
        assert routed.call_args.kwargs["sizes"] == sizes
    torch.cuda.synchronize()
    torch.testing.assert_close(
        reduced, summed.narrow(0, sum(sizes[:rank]), sizes[rank]), rtol=0, atol=0
    )

    (output / f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "passed": True, "results": results}, indent=2)
    )
    dist.barrier(group=group.cpu_group)
    if rank == 0:
        print(json.dumps({"passed": True, "ranks": 8}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
