import os
import unittest

import sgl_kernel  # noqa: F401
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_ring_parallel_world_size,
    get_sp_world_size,
    get_tp_world_size,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=40, suite="extra-tp-test-cpu-intel")


def _parallel_worker(
    rank: int,
    tp_size: int,
    sp_degree: int,
    master_port: int,
):
    world_size = tp_size * sp_degree

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )

    try:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=tp_size,
            sp_size=sp_degree,
            ulysses_degree=sp_degree,
            ring_degree=1,
            distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        )

        assert get_tp_world_size() == tp_size
        assert get_sp_world_size() == sp_degree
        assert get_ring_parallel_world_size() == 1
    finally:
        cleanup_dist_env_and_memory()


class TestDiffusionCPUParallelUnit(CustomTestCase):
    def _run_parallel_test(self, tp_size: int, sp_degree: int):
        mp.spawn(
            _parallel_worker,
            args=(tp_size, sp_degree, find_available_port(23456)),
            nprocs=tp_size * sp_degree,
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
