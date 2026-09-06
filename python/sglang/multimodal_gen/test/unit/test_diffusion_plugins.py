# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen import plugins
from sglang.srt.plugins.hook_registry import HookRegistry as SrtHookRegistry
from sglang.srt.plugins.hook_registry import (
    HookSource,
    HookType,
)


def _entry_point(name, distribution):
    entry_point = MagicMock(name=f"entry_point_{name}")
    entry_point.name = name
    entry_point.value = f"test_plugin:{name}"
    entry_point.dist = SimpleNamespace(name=distribution)
    entry_point.load.return_value = MagicMock(name=f"plugin_{name}")
    return entry_point


class TestDiffusionPluginBarrier(unittest.TestCase):
    def setUp(self):
        state = (
            plugins._plugin_registration.state,
            plugins._plugin_registration.error,
            plugins._hook_application.state,
            plugins._hook_application.error,
            plugins._required_dist,
        )
        self.addCleanup(self._restore_lifecycle, state)
        plugins._reset_lifecycle_for_tests()

    @staticmethod
    def _restore_lifecycle(state):
        (
            plugins._plugin_registration.state,
            plugins._plugin_registration.error,
            plugins._hook_application.state,
            plugins._hook_application.error,
            plugins._required_dist,
        ) = state

    def test_body_runs_once_across_repeated_calls(self):
        with patch.object(plugins, "_register_plugins_once") as load_once:
            plugins.load_plugins()
            plugins.load_plugins()

        load_once.assert_called_once_with()

    def test_failure_is_terminal_instead_of_replaying_partial_side_effects(self):
        with patch.object(
            plugins,
            "_register_plugins_once",
            side_effect=RuntimeError("plugin init exploded"),
        ) as load_once:
            with self.assertRaisesRegex(RuntimeError, "exploded"):
                plugins.load_plugins()
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                plugins.load_plugins()

        load_once.assert_called_once_with()

    def test_reentrant_call_returns_instead_of_recursing(self):
        calls = []

        def reentrant():
            calls.append(1)
            plugins.load_plugins()

        with patch.object(plugins, "_register_plugins_once", reentrant):
            plugins.load_plugins()

        self.assertEqual(len(calls), 1)

    def test_hook_application_is_a_separate_once_only_phase(self):
        with (
            patch.object(plugins, "_register_plugins_once", return_value="vendor-pkg"),
            patch.object(plugins.HookRegistry, "apply_hooks") as apply_hooks,
            patch.object(plugins, "_require_hooks_applied") as require_hooks,
        ):
            plugins.load_plugins()
            apply_hooks.assert_not_called()

            plugins.apply_plugin_hooks()
            plugins.apply_plugin_hooks()

        apply_hooks.assert_called_once_with()
        require_hooks.assert_called_once_with("vendor-pkg")

    def test_hook_application_failure_is_terminal(self):
        with (
            patch.object(plugins, "_register_plugins_once", return_value=None),
            patch.object(
                plugins.HookRegistry,
                "apply_hooks",
                side_effect=RuntimeError("hook application exploded"),
            ) as apply_hooks,
        ):
            with self.assertRaisesRegex(RuntimeError, "application exploded"):
                plugins.apply_plugin_hooks()
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                plugins.apply_plugin_hooks()

        apply_hooks.assert_called_once_with()

    def test_hooks_from_a_reentrant_callback_are_applied_by_the_outer_call(self):
        def reentrant():
            plugins.apply_plugin_hooks()

        with (
            patch.object(plugins, "_register_plugins_once", reentrant),
            patch.object(plugins.HookRegistry, "apply_hooks") as apply_hooks,
        ):
            plugins.apply_plugin_hooks()

        apply_hooks.assert_called_once_with()


class TestDiffusionPlugins(unittest.TestCase):
    def test_load_executes_callbacks_without_resolving_hook_targets(self):
        register = MagicMock()

        with (
            patch.object(
                plugins, "_discover", return_value={"test": (register, "test-package")}
            ),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="test-package"
            ),
            patch.object(plugins.HookRegistry, "apply_hooks") as apply_hooks,
        ):
            required_dist = plugins._register_plugins_once()

        register.assert_called_once_with()
        apply_hooks.assert_not_called()
        self.assertEqual(required_dist, "test-package")

    def test_a_failing_callback_does_not_stop_the_others(self):
        healthy = MagicMock()

        with (
            patch.object(
                plugins,
                "_discover",
                return_value={
                    "broken": (MagicMock(side_effect=RuntimeError("boom")), "a"),
                    "healthy": (healthy, "b"),
                },
            ),
        ):
            plugins._register_plugins_once()

        healthy.assert_called_once_with()

    def test_hooks_registered_while_importing_a_plugin_keep_their_source(self):
        target = "test_diffusion_plugins.import_time_target"
        hook = MagicMock()
        entry_point = _entry_point("vendor", "vendor-pkg")

        def load():
            plugins.plugin_hook(target)(hook)
            return MagicMock()

        entry_point.load.side_effect = load
        self.addCleanup(plugins.HookRegistry._hooks.pop, target, None)

        with (
            patch.object(plugins, "entry_points", return_value=[entry_point]),
            patch.object(plugins, "_get_excluded_dists", return_value=set()),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="vendor-pkg"
            ),
        ):
            plugins._discover()

        self.assertEqual(
            plugins.HookRegistry._hooks[target][0][2],
            HookSource("vendor", "vendor-pkg"),
        )

    def test_failing_optional_callback_discards_its_registered_hooks(self):
        target = "test_diffusion_plugins.partial_callback_target"

        def register_then_fail():
            plugins.plugin_hook(target)(lambda result: result)
            raise RuntimeError("boom")

        self.addCleanup(plugins.HookRegistry._hooks.pop, target, None)
        with (
            patch.object(
                plugins,
                "_discover",
                return_value={"broken": (register_then_fail, "optional-pkg")},
            ),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="vendor-pkg"
            ),
        ):
            plugins._register_plugins_once()

        self.assertNotIn(target, plugins.HookRegistry._hooks)

    def test_failing_optional_import_discards_its_registered_hooks(self):
        target = "test_diffusion_plugins.partial_import_target"
        entry_point = _entry_point("broken", "optional-pkg")

        def load_then_fail():
            plugins.plugin_hook(target)(lambda result: result)
            raise RuntimeError("boom")

        entry_point.load.side_effect = load_then_fail
        self.addCleanup(plugins.HookRegistry._hooks.pop, target, None)
        with (
            patch.object(plugins, "entry_points", return_value=[entry_point]),
            patch.object(plugins, "_get_excluded_dists", return_value=set()),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="vendor-pkg"
            ),
        ):
            self.assertEqual(plugins._discover(), {})

        self.assertNotIn(target, plugins.HookRegistry._hooks)

    def test_duplicate_plugin_names_fail_before_import(self):
        entries = [_entry_point("duplicate", "one"), _entry_point("duplicate", "two")]
        with (
            patch.object(plugins, "entry_points", return_value=entries),
            patch.object(plugins, "_get_excluded_dists", return_value=set()),
            patch.object(plugins, "get_selected_platform_dist", return_value=None),
            self.assertRaisesRegex(RuntimeError, "must be unique"),
        ):
            plugins._discover()

        for entry_point in entries:
            entry_point.load.assert_not_called()

    def test_a_failing_load_from_the_selected_platform_aborts_startup(self):
        broken = _entry_point("vendor", "vendor-pkg")
        broken.load.side_effect = RuntimeError("vendor wheel is broken")
        unrelated = _entry_point("other", "other-pkg")
        unrelated.load.side_effect = RuntimeError("third party is broken")

        with (
            patch.object(plugins, "entry_points", return_value=[unrelated]),
            patch.object(plugins, "_get_excluded_dists", return_value=set()),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="vendor-pkg"
            ),
        ):
            self.assertEqual(plugins._discover(), {})

        with (
            patch.object(plugins, "entry_points", return_value=[broken]),
            patch.object(plugins, "_get_excluded_dists", return_value=set()),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="vendor-pkg"
            ),
            self.assertRaisesRegex(RuntimeError, "vendor wheel is broken"),
        ):
            plugins._discover()

    def test_a_failing_callback_from_the_selected_platform_aborts_startup(self):
        boom = MagicMock(side_effect=RuntimeError("vendor hook is broken"))

        with (
            patch.object(plugins, "_discover", return_value={"x": (boom, "other-pkg")}),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="vendor-pkg"
            ),
        ):
            plugins._register_plugins_once()

        with (
            patch.object(
                plugins, "_discover", return_value={"v": (boom, "vendor-pkg")}
            ),
            patch.object(
                plugins, "get_selected_platform_dist", return_value="vendor-pkg"
            ),
            self.assertRaisesRegex(RuntimeError, "vendor hook is broken"),
        ):
            plugins._register_plugins_once()

    def test_an_unapplied_hook_from_the_selected_platform_aborts_startup(self):
        # apply_hooks() logs and moves on, so a required target can go unpatched.
        target = "test_diffusion_plugins.unappliable"
        self.addCleanup(plugins.HookRegistry._hooks.pop, target, None)
        self.addCleanup(plugins.HookRegistry._patched.discard, target)
        plugins.HookRegistry._hooks[target] = [
            (HookType.AFTER, lambda r: r, HookSource("v", "vendor-pkg"))
        ]

        with self.assertRaisesRegex(RuntimeError, "could not apply hooks"):
            plugins._require_hooks_applied("vendor-pkg")

        plugins._require_hooks_applied("other-pkg")

        plugins.HookRegistry._patched.add(target)
        plugins._require_hooks_applied("vendor-pkg")

    def test_plugin_hook_uses_the_diffusion_registry(self):
        target = "test_diffusion_plugins.target"

        def hook():
            pass

        self.addCleanup(plugins.HookRegistry._hooks.pop, target, None)
        plugins.plugin_hook(target)(hook)

        self.assertIs(plugins.HookRegistry._hooks[target][0][1], hook)
        self.assertNotIn(target, SrtHookRegistry._hooks)

    def test_excludes_every_unselected_platform_distribution(self):
        entries = [
            _entry_point("selected", "selected-package"),
            _entry_point("selected_extra", "selected-package"),
            _entry_point("other", "other-package"),
        ]
        # None is a built-in platform: nothing installed is in use.
        cases = (
            ("selected-package", {"other-package"}),
            (None, {"selected-package", "other-package"}),
        )
        for selected_dist, expected in cases:
            with (
                self.subTest(selected_dist=selected_dist),
                patch.object(
                    plugins, "get_selected_platform_dist", return_value=selected_dist
                ),
                patch.object(plugins, "entry_points", return_value=entries),
            ):
                self.assertEqual(plugins._get_excluded_dists(), expected)


if __name__ == "__main__":
    unittest.main()
