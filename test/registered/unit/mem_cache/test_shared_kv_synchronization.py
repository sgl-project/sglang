import inspect
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.multiprocessing as mp

from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-c", runner_config="4-gpu-b200")

PORT = 29827


def _run_delayed_device_publication(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    publisher = SharedWritePublisher(get_parallel().attn_cp_group)
    if rank == 0:
        torch.cuda._sleep(10_000_000)
    publisher.publish()
    assert publisher._fence.item() == world_size

    destroy_model_parallel()
    destroy_distributed_environment()


class TestSharedWritePublisher(CustomTestCase):
    def test_publish_uses_one_attention_cp_device_collective(self):
        cp_group = MagicMock()
        fence = MagicMock()
        with (
            patch("torch.cuda.current_device", return_value=3),
            patch("torch.ones", return_value=fence) as make_tensor,
        ):
            publisher = SharedWritePublisher(cp_group)
            publisher.publish()

        make_tensor.assert_called_once_with(
            (1,), dtype=torch.int32, device=torch.device("cuda", 3)
        )
        fence.fill_.assert_called_once_with(1)
        cp_group._all_reduce_in_place.assert_called_once_with(fence)
        cp_group.cpu_group.assert_not_called()

    def test_constructor_has_no_runtime_disable_or_strategy_switch(self):
        parameters = inspect.signature(SharedWritePublisher).parameters

        self.assertEqual(list(parameters), ["attention_cp_group"])
        self.assertFalse(
            {"enabled", "async_op", "strategy", "signal"}.intersection(parameters)
        )

    def test_production_source_contains_no_host_collective(self):
        source = inspect.getsource(
            __import__(
                "sglang.srt.mem_cache.shared_kv.synchronization",
                fromlist=["SharedWritePublisher"],
            )
        )

        for forbidden in (
            "all_gather_object",
            "gather_object",
            "cpu_group",
            "SGLANG_",
            "os.environ",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_delayed_writer_is_published_on_the_device_collective(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("device publication test needs at least two GPUs")
        mp.spawn(
            _run_delayed_device_publication,
            args=(2, PORT),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
