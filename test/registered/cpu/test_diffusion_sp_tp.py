import os
import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_ring_parallel_world_size,
    get_sp_world_size,
    get_tp_world_size,
    get_ulysses_parallel_world_size,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=40, suite="extra-tp-test-cpu-intel")


def _parallel_worker(
    rank: int,
    server_args: ServerArgs,
    master_port: int,
):
    torch.set_num_threads(1)

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(server_args.num_gpus),
        LOCAL_RANK=str(rank),
        LOCAL_SIZE=str(server_args.num_gpus),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )

    try:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=server_args.tp_size,
            sp_size=server_args.sp_degree,
            ulysses_degree=server_args.ulysses_degree,
            ring_degree=server_args.ring_degree,
            distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        )

        assert get_tp_world_size() == server_args.tp_size
        assert get_sp_world_size() == server_args.sp_degree
        assert get_ulysses_parallel_world_size() == server_args.ulysses_degree
        assert get_ring_parallel_world_size() == server_args.ring_degree

    finally:
        cleanup_dist_env_and_memory()


class TestDiffusionCPUParallelUnit(CustomTestCase):
    def _run_parallel_test(self, tp_size: int, sp_degree: int):
        server_args = ServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            num_gpus=1,
            tp_size=tp_size,
            ulysses_degree=sp_degree,
            ring_degree=1,
        )

        self.assertEqual(server_args.tp_size, tp_size)
        self.assertEqual(server_args.sp_degree, sp_degree)
        self.assertEqual(server_args.ulysses_degree, sp_degree)
        self.assertEqual(server_args.ring_degree, 1)
        self.assertEqual(server_args.num_gpus, tp_size * sp_degree)

        # Initialize distributed groups using the adjusted server args.
        mp.spawn(
            _parallel_worker,
            args=(
                server_args,
                find_available_port(23456),
            ),
            nprocs=server_args.num_gpus,
            join=True,
        )

    def test_cpu_tp_only(self):
        self._run_parallel_test(tp_size=2, sp_degree=1)

    def test_cpu_sp_only(self):
        self._run_parallel_test(tp_size=1, sp_degree=2)

    def test_cpu_tp2_sp2(self):
        self._run_parallel_test(tp_size=2, sp_degree=2)


if __name__ == "__main__":
    unittest.main()
