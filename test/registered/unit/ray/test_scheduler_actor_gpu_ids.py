"""Unit tests for srt/ray/scheduler_actor: which GPU id each consumer receives."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import CustomTestCase


class TestSchedulerActorGpuIds(CustomTestCase):
    def test_device_ordinal_follows_the_visible_device_view(self):
        """Ray narrows CUDA_VISIBLE_DEVICES to an actor's GPUs, and torch numbers
        that view from 0, so a physical id other than 0 is an invalid ordinal."""
        from sglang.srt.ray.scheduler_actor import resolve_actor_gpu_ids

        cases = [
            # (assigned by Ray, CUDA_VISIBLE_DEVICES, fallback) -> (physical, local)
            (["3"], "3", 7, (3, 0)),
            (["0"], "0", 7, (0, 0)),
            # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES keeps the whole node visible.
            (["3"], "0,1,2,3", 7, (3, 3)),
            (["3"], None, 7, (3, 3)),
            (["2", "3"], "2,3", 7, (2, 0)),
            # No Ray assignment: the launcher's per-node index is used as-is.
            ([], "0,1", 1, (1, 1)),
        ]
        for assigned, visible, fallback, expected in cases:
            with self.subTest(assigned=assigned, visible=visible):
                self.assertEqual(
                    resolve_actor_gpu_ids(
                        assigned_gpus=assigned,
                        fallback_gpu_id=fallback,
                        cuda_visible_devices=visible,
                    ),
                    expected,
                )

    def test_actor_publishes_ordinal_and_binds_topology_to_physical_gpu(self):
        """The published gpu_id drives torch.cuda.set_device; CPU affinity and NUMA
        binding look the GPU up in the host topology and need its physical id."""
        from sglang.srt.environ import envs
        from sglang.srt.managers import scheduler as scheduler_module
        from sglang.srt.ray import scheduler_actor as actor_module
        from sglang.srt.utils import numa_utils

        published = []
        runtime_context = MagicMock()
        runtime_context.get_accelerator_ids.return_value = {"GPU": ["3"]}
        configure = MagicMock()
        numa_lookup = MagicMock(return_value=None)
        server_args = SimpleNamespace()

        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3"}),
            envs.SGLANG_NUMA_BIND_V2.override(True),
            patch.object(numa_utils, "get_numa_node_if_available", numa_lookup),
            patch.object(
                actor_module.ray, "get_runtime_context", return_value=runtime_context
            ),
            patch.object(
                actor_module,
                "publish",
                side_effect=lambda *a, **kw: published.append(kw["ranks"]),
            ),
            patch.object(actor_module, "spawn_world_rank", return_value=0),
            patch.object(scheduler_module, "resolve_spawn_dp_rank", return_value=0),
            patch.object(scheduler_module, "configure_scheduler_process", configure),
            patch.object(scheduler_module, "Scheduler"),
        ):
            actor_module.SchedulerActor.__ray_actor_class__(
                server_args=server_args,
                port_args=SimpleNamespace(),
                gpu_id=0,
                tp_rank=0,
                pp_rank=0,
                dp_rank=None,
            )

        self.assertEqual([ranks.gpu_id for ranks in published], [0])
        configure.assert_called_once_with(server_args, 3)
        numa_lookup.assert_called_once_with(server_args, 3)


if __name__ == "__main__":
    unittest.main()
