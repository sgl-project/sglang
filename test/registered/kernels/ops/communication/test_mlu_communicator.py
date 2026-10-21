"""Validate generic SGLang collectives on CNCL, including compiled execution."""

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

register_mlu_ci(est_time=60, suite="pr-test-2-mlu")


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


def _run_mlu_communicator_worker(rank: int, world_size: int, init_method: str):

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


class TestMLUCommunicator(CustomTestCase):
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
