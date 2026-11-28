"""Global TCPStore master IP when ranks disagree on how to find it (#53700)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import multiprocessing as mp
import os
import unittest
from datetime import timedelta

from sglang.srt.utils.network import get_free_port
from sglang.test.test_utils import CustomTestCase

_WORLD_SIZE = 2


def _create_store(rank, init_method, store_port, dynamic_ranks, env_ranks, results):
    if rank in env_ranks:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    else:
        os.environ.pop("MASTER_ADDR", None)
    os.environ["SGLANG_TCP_STORE_PORT"] = str(store_port)

    import torch.distributed as dist

    from sglang.srt.distributed import parallel_state
    from sglang.srt.distributed.utils import get_global_tcp_store

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=10),
    )
    parallel_state._create_global_tcp_store(
        rank=rank,
        world_size=_WORLD_SIZE,
        dist_init_addr=init_method,
        allow_dynamic_membership=rank in dynamic_ranks,
        required=True,
    )
    results.put((rank, get_global_tcp_store().host))
    dist.destroy_process_group()


class TestGlobalTcpStoreMasterIp(CustomTestCase):
    def _run(self, dynamic_ranks=(), env_ranks=()):
        ctx = mp.get_context("spawn")
        results = ctx.Queue()
        init_method = f"tcp://127.0.0.1:{get_free_port()}"
        args = (init_method, get_free_port(), dynamic_ranks, env_ranks, results)
        procs = [
            ctx.Process(target=_create_store, args=(rank, *args))
            for rank in range(_WORLD_SIZE)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join()

        self.assertEqual([p.exitcode for p in procs], [0] * _WORLD_SIZE)
        hosts = dict(results.get(timeout=5) for _ in range(_WORLD_SIZE))
        self.assertEqual(hosts, {0: "127.0.0.1", 1: "127.0.0.1"})

    def test_only_rank0_allows_dynamic_membership(self):
        self._run(dynamic_ranks={0})

    def test_master_addr_set_on_rank0_only(self):
        self._run(env_ranks={0})

    def test_master_addr_set_on_rank1_only(self):
        self._run(env_ranks={1})


if __name__ == "__main__":
    unittest.main()
