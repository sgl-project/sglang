# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen import plugins
from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.srt.plugins.hook_registry import HookRegistry as SrtHookRegistry


def _entry_point(name, distribution):
    entry_point = MagicMock(name=f"entry_point_{name}")
    entry_point.name = name
    entry_point.value = f"test_plugin:{name}"
    entry_point.dist = SimpleNamespace(name=distribution)
    entry_point.load.return_value = MagicMock(name=f"plugin_{name}")
    return entry_point


class TestDiffusionPlugins(unittest.TestCase):
    def test_loads_and_applies_callbacks_once(self):
        register = MagicMock()

        with (
            patch.object(plugins, "_plugins_loaded", False),
            patch.object(
                plugins,
                "load_plugins_by_group",
                return_value={"test": (register, "test-package")},
            ) as load_plugins_by_group,
            patch.object(plugins, "_get_excluded_dists", return_value=set()),
            patch.object(plugins.HookRegistry, "apply_hooks") as apply_hooks,
        ):
            plugins.load_plugins()
            plugins.load_plugins()

        load_plugins_by_group.assert_called_once_with(
            plugins.GENERAL_PLUGINS_GROUP,
            excluded_dists=set(),
            exclusion_reason="diffusion platform selection",
        )
        register.assert_called_once_with()
        apply_hooks.assert_called_once_with()

    def test_plugin_hook_uses_the_diffusion_registry(self):
        target = "test_diffusion_plugins.target"

        def hook():
            pass

        self.addCleanup(plugins.HookRegistry._hooks.pop, target, None)
        plugins.plugin_hook(target)(hook)

        self.assertIs(plugins.HookRegistry._hooks[target][0][1], hook)
        self.assertNotIn(target, SrtHookRegistry._hooks)

    def test_selected_platform_excludes_other_platform_packages(self):
        entries = [
            _entry_point("selected", "selected-package"),
            _entry_point("selected_extra", "selected-package"),
            _entry_point("other", "other-package"),
            _entry_point("cpu", "reserved-package"),
        ]
        selections = (
            (
                {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "selected"},
                {"other-package", "reserved-package"},
            ),
            (
                {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu"},
                {"selected-package", "other-package", "reserved-package"},
            ),
        )
        for selection, expected in selections:
            with (
                self.subTest(selection=selection),
                patch.dict(os.environ, selection),
                patch.object(plugins, "entry_points", return_value=entries),
            ):
                self.assertEqual(plugins._get_excluded_dists(), expected)

    def test_scheduler_process_loads_plugins(self):
        with (
            patch(
                "sglang.multimodal_gen.plugins.load_plugins",
                side_effect=RuntimeError("plugins loaded"),
            ),
            self.assertRaisesRegex(RuntimeError, "plugins loaded"),
        ):
            gpu_worker.run_scheduler_process(
                local_rank=0,
                rank=0,
                master_port=0,
                server_args=None,
                pipe_writer=None,
                task_pipe_r=None,
                result_pipe_w=None,
            )


if __name__ == "__main__":
    unittest.main()
