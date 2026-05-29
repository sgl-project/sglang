"""Unit tests for Ray scheduler actor plugin loading."""

import importlib.abc
import importlib.machinery
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch

    try:
        torch.xpu.get_device_name = lambda device=None: ""
    except Exception:
        pass
except Exception:
    pass


def _maybe_stub_sgl_kernel():
    try:
        import sgl_kernel  # noqa: F401

        return
    except (ImportError, OSError):
        pass

    stub = types.ModuleType("sgl_kernel")
    stub.__getattr__ = lambda name: MagicMock(name=f"sgl_kernel.{name}")
    sys.modules["sgl_kernel"] = stub


_maybe_stub_sgl_kernel()

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerActorPlugins(CustomTestCase):
    def _install_fake_ray(self):
        ray_module_names = [
            "ray",
            "ray.util",
            "ray.util.placement_group",
            "ray.util.scheduling_strategies",
        ]
        old_modules = {name: sys.modules.get(name) for name in ray_module_names}
        for name in ray_module_names:
            sys.modules.pop(name, None)

        ray_module = types.ModuleType("ray")
        ray_module.__path__ = []
        util_module = types.ModuleType("ray.util")
        util_module.__path__ = []
        placement_group_module = types.ModuleType("ray.util.placement_group")
        scheduling_strategies_module = types.ModuleType(
            "ray.util.scheduling_strategies"
        )

        def remote(obj=None, **kwargs):
            def decorate(cls):
                return SimpleNamespace(
                    __ray_metadata__=SimpleNamespace(modified_class=cls)
                )

            if obj is None:
                return decorate
            return decorate(obj)

        class PlacementGroup:
            pass

        class PlacementGroupSchedulingStrategy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        ray_module.remote = remote
        ray_module.get = MagicMock()
        ray_module.kill = MagicMock()
        ray_module.get_runtime_context = MagicMock()
        ray_module.exceptions = SimpleNamespace(RayActorError=Exception)
        ray_module.util = util_module
        util_module.get_node_ip_address = lambda: "127.0.0.1"
        util_module.get_current_placement_group = lambda: None
        placement_group_module.PlacementGroup = PlacementGroup
        placement_group_module.placement_group = MagicMock()
        scheduling_strategies_module.PlacementGroupSchedulingStrategy = (
            PlacementGroupSchedulingStrategy
        )

        sys.modules["ray"] = ray_module
        sys.modules["ray.util"] = util_module
        sys.modules["ray.util.placement_group"] = placement_group_module
        sys.modules["ray.util.scheduling_strategies"] = scheduling_strategies_module
        return old_modules

    def _restore_modules(self, old_modules):
        for name in old_modules:
            sys.modules.pop(name, None)
        for name, module in old_modules.items():
            if module is not None:
                sys.modules[name] = module

    def test_ray_scheduler_actor_loads_plugins_before_scheduler_import(self):
        old_ray_modules = self._install_fake_ray()
        fake_ray = sys.modules["ray"]
        old_sglang_ray_modules = {
            name: sys.modules.get(name)
            for name in [
                "sglang.srt.ray",
                "sglang.srt.ray.engine",
                "sglang.srt.ray.scheduler_actor",
            ]
        }
        for name in old_sglang_ray_modules:
            sys.modules.pop(name, None)

        try:
            from sglang.srt.ray.scheduler_actor import SchedulerActor
        finally:
            for name in old_sglang_ray_modules:
                sys.modules.pop(name, None)
            for name, module in old_sglang_ray_modules.items():
                if module is not None:
                    sys.modules[name] = module
            self._restore_modules(old_ray_modules)

        events = []
        scheduler_module_name = "sglang.srt.managers.scheduler"
        old_scheduler_module = sys.modules.pop(scheduler_module_name, None)

        scheduler_module = types.ModuleType(scheduler_module_name)
        scheduler_module.__package__ = "sglang.srt.managers"

        def configure_scheduler_process(*args, **kwargs):
            events.append("configure_scheduler_process")
            return args[-1]

        class DummyScheduler:
            def __init__(self, **kwargs):
                events.append("scheduler_init")
                self.ps = SimpleNamespace(gpu_id=kwargs["gpu_id"])

            def get_init_info(self):
                return {}

        scheduler_module.configure_scheduler_process = configure_scheduler_process
        scheduler_module.Scheduler = DummyScheduler

        class SchedulerFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path, target=None):
                if fullname == scheduler_module_name:
                    return importlib.machinery.ModuleSpec(fullname, self)
                return None

            def create_module(self, spec):
                return scheduler_module

            def exec_module(self, module):
                events.append("scheduler_import")

        finder = SchedulerFinder()
        actor_cls = SchedulerActor.__ray_metadata__.modified_class

        with patch(
            "sglang.srt.plugins.load_plugins",
            side_effect=lambda: events.append("load_plugins"),
        ):
            fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {
                "GPU": ["3"]
            }
            sys.modules["ray"] = fake_ray
            sys.meta_path.insert(0, finder)
            try:
                actor_cls(
                    SimpleNamespace(numa_node=None),
                    SimpleNamespace(),
                    gpu_id=0,
                    tp_rank=0,
                    attn_cp_rank=0,
                    moe_dp_rank=0,
                    moe_ep_rank=0,
                    pp_rank=0,
                    dp_rank=None,
                )
            finally:
                sys.meta_path.remove(finder)
                sys.modules.pop(scheduler_module_name, None)
                sys.modules.pop("ray", None)
                if old_scheduler_module is not None:
                    sys.modules[scheduler_module_name] = old_scheduler_module

        self.assertEqual(
            events,
            [
                "load_plugins",
                "scheduler_import",
                "configure_scheduler_process",
                "scheduler_init",
            ],
        )


if __name__ == "__main__":
    unittest.main()
