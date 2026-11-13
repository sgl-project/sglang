"""Validate CNCL collectives and compiled variable-size Torch fallback."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_mlu  # noqa: F401

from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=90, suite="pr-test-2-mlu")


def _compile_collectives(graph, example_inputs):
    # Keep collectives opaque to Dynamo instead of decomposing them into c10d ops.
    targets = {node.target for node in graph.graph.nodes if node.op == "call_function"}
    for op in (
        torch.ops.sglang.inplace_all_reduce,
        torch.ops.sglang.reg_all_gather_into_tensor,
    ):
        assert op in targets or op.default in targets, targets
    assert not any("c10d" in str(target) for target in targets)
    return torch._inductor.compile(graph, example_inputs)


def _run_mlu_communicator_worker(
    rank: int, world_size: int, init_method: str, variable_sizes: bool = False
):
    from sglang.srt.distributed.parallel_state import init_model_parallel_group

    torch.mlu.set_device(rank)
    dist.init_process_group(
        backend="cncl",
        init_method=init_method,
        timeout=timedelta(seconds=60),
        rank=rank,
        world_size=world_size,
    )
    comm = None
    try:
        comm = init_model_parallel_group(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            backend="cncl",
            use_custom_allreduce=False,
            use_mscclpp=False,
            use_torch_symm_mem_allreduce=False,
        )
        assert comm.pynccl_comm is None

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            if variable_sizes:
                _check_variable_collectives(comm, dtype)
                continue

            for dim in (0, 1, -1):

                def collect(x):
                    return comm.all_reduce(x.clone()), comm.all_gather(x, dim=dim)

                compiled = torch.compile(
                    collect, backend=_compile_collectives, fullgraph=True
                )
                for fn in (collect, compiled):
                    # Exercise replay with new values and a changed leading dimension.
                    for rows, offset in ((2, 0), (2, 3), (4, 1)):
                        base = torch.arange(rows * 3, dtype=dtype).reshape(rows, 3)
                        peers = [base + r + offset for r in range(world_size)]
                        x = peers[rank].to("mlu")
                        reduced, gathered = fn(x)
                        torch.testing.assert_close(
                            reduced.cpu(), sum(peers), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            gathered.cpu(), torch.cat(peers, dim=dim), rtol=0, atol=0
                        )
                        torch.testing.assert_close(x.cpu(), peers[rank], rtol=0, atol=0)
                torch._dynamo.reset()
    finally:
        if comm is not None:
            comm.destroy()
        dist.destroy_process_group()


def _check_variable_collectives(comm, dtype):
    """Rank-distinct values expose wrong offsets and shard sizes."""
    size_cases = (None, [2] * comm.world_size)
    if comm.world_size > 1:
        size_cases += ([1, 3], [0, 2], [2, 0], [0, 0])
    for sizes in size_cases:
        counts = sizes if sizes is not None else [2] * comm.world_size
        rows = sum(counts)
        base = torch.arange(rows * 3, dtype=dtype).reshape(rows, 3)
        peers = [base + r for r in comm.ranks]
        source = peers[comm.rank_in_group].to(comm.device)
        start = sum(counts[: comm.rank_in_group])
        count = counts[comm.rank_in_group]
        expected = sum(peers).narrow(0, start, count)

        def collect(x, local, reduced, gathered):
            return (
                comm.reduce_scatterv(x, output=reduced, sizes=sizes),
                comm.all_gatherv(local, sizes=sizes, output=gathered),
            )

        local = source.narrow(0, start, count).contiguous()
        gathered_expected = torch.cat(
            [p.narrow(0, sum(counts[:i]), counts[i]) for i, p in enumerate(peers)]
        )
        reduced = torch.empty_like(expected, device=comm.device)
        gathered = torch.empty_like(source)
        compiled = torch.compile(collect, backend="inductor", fullgraph=True)
        for fn in (collect, compiled):
            for offset in (0, 7):
                result, results = fn(source + offset, local + offset, reduced, gathered)
                assert result is reduced
                assert len(results) == 1 and results[0] is gathered
                torch.testing.assert_close(
                    result.cpu(), expected + offset * comm.world_size, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    gathered.cpu(), gathered_expected + offset, rtol=0, atol=0
                )
        torch._dynamo.reset()

        allocated = comm.reduce_scatterv(source, sizes=sizes)
        torch.testing.assert_close(allocated.cpu(), expected, rtol=0, atol=0)
        # Preserve the historical list return, including single-tensor input.
        outputs = comm.all_gatherv([local, local + 1], sizes=sizes)
        assert len(outputs) == 2
        for i, result in enumerate(outputs):
            torch.testing.assert_close(
                result.cpu(), gathered_expected + i, rtol=0, atol=0
            )
        torch.testing.assert_close(source.cpu(), peers[comm.rank_in_group])


class TestMLUCommunicator(CustomTestCase):
    def test_variable_collectives_on_cncl(self):
        world_size = 2
        self.assertGreaterEqual(torch.mlu.device_count(), world_size)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _run_mlu_communicator_worker,
                args=(world_size, (Path(directory) / "rendezvous").as_uri(), True),
                nprocs=world_size,
                join=True,
            )

    def test_all_reduce_and_all_gather_on_cncl(self):
        world_size = 2
        self.assertGreaterEqual(torch.mlu.device_count(), world_size)
        with tempfile.TemporaryDirectory() as directory:
            init_method = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(
                _run_mlu_communicator_worker,
                args=(world_size, init_method),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
