"""
Unit tests for the plugin loading flow.

Covers: idempotency, apply_hooks invocation, exception resilience,
SGLANG_PLUGINS whitelist, SGLANG_PLATFORM exclusion logic,
SGLANG_REQUIRED_PLUGINS checks, and _current_plugin_source context var reset.

Run:  python -m pytest test/registered/unit/plugins/test_load_plugins.py -v
"""

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.srt.plugins import (
    _current_plugin_source,
    _get_excluded_dists,
    load_plugins,
    load_plugins_by_group,
)
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_ep(name, dist_name=None, load_fn=None):
    """Create a mock entry point."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"fake_module:{name}"
    ep.dist = MagicMock()
    ep.dist.name = dist_name or f"{name}-dist"
    if load_fn is not None:
        ep.load.return_value = load_fn
    else:
        ep.load.return_value = MagicMock()
    return ep


def _reset_plugins_loaded():
    """Reset the _plugins_loaded flag so load_plugins() can run again."""
    import sglang.srt.plugins as plugins_mod

    plugins_mod._plugins_loaded = False


class TestLoadPlugins(CustomTestCase):
    """Tests for load_plugins() and related helpers."""

    def setUp(self):
        _reset_plugins_loaded()

    def tearDown(self):
        _reset_plugins_loaded()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points", return_value=[])
    def test_load_plugins_idempotent_and_calls_apply(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Second call is a no-op; first call invokes apply_hooks."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        load_plugins()
        self.assertEqual(mock_registry.apply_hooks.call_count, 1)

        load_plugins()  # should be skipped
        self.assertEqual(mock_registry.apply_hooks.call_count, 1)

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_plugin_exception_does_not_crash(self, mock_eps, mock_envs, mock_registry):
        """A failing plugin should not prevent others from loading."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        def bad_plugin():
            raise RuntimeError("boom")

        good_call_log = []

        def good_plugin():
            good_call_log.append("ok")

        eps = [
            _make_ep("bad", load_fn=bad_plugin),
            _make_ep("good", load_fn=good_plugin),
        ]
        mock_eps.return_value = eps

        with self.assertLogs("sglang.srt.plugins", level="ERROR") as cm:
            load_plugins()

        self.assertTrue(any("boom" in msg for msg in cm.output))
        self.assertEqual(good_call_log, ["ok"])
        mock_registry.apply_hooks.assert_called_once()

    @patch("sglang.srt.plugins.entry_points")
    @patch("sglang.srt.plugins.envs")
    def test_sglang_plugins_whitelist(self, mock_envs, mock_eps):
        """Only plugins named in SGLANG_PLUGINS should be loaded."""
        mock_envs.SGLANG_PLUGINS.get.return_value = "alpha,gamma"
        mock_envs.SGLANG_PLATFORM.get.return_value = ""

        alpha_fn = MagicMock()
        beta_fn = MagicMock()
        gamma_fn = MagicMock()

        eps = [
            _make_ep("alpha", load_fn=alpha_fn),
            _make_ep("beta", load_fn=beta_fn),
            _make_ep("gamma", load_fn=gamma_fn),
        ]
        mock_eps.return_value = eps

        result = load_plugins_by_group("test.group")
        self.assertIn("alpha", result)
        self.assertNotIn("beta", result)
        self.assertIn("gamma", result)

    @patch("sglang.srt.plugins.entry_points")
    @patch("sglang.srt.plugins.envs")
    def test_excluded_dists(self, mock_envs, mock_eps):
        """SGLANG_PLATFORM excludes other platform dists; empty when unset."""
        # Case 1: no env set → empty
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        self.assertEqual(_get_excluded_dists(), set())

        # Case 2: env set → exclude other dists
        mock_envs.SGLANG_PLATFORM.get.return_value = "kunlun"
        ep_kunlun = _make_ep("kunlun", dist_name="kunlun-pkg")
        ep_other = _make_ep("other_hw", dist_name="other-pkg")
        mock_eps.return_value = [ep_kunlun, ep_other]

        excluded = _get_excluded_dists()
        self.assertNotIn("kunlun-pkg", excluded)
        self.assertIn("other-pkg", excluded)

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_current_plugin_source_set_during_and_reset_after(
        self, mock_eps, mock_envs, mock_registry
    ):
        """_current_plugin_source is set during plugin execution, reset after."""
        sources_seen = []

        def spy_plugin():
            sources_seen.append(_current_plugin_source.get())

        mock_eps.return_value = [_make_ep("spy", load_fn=spy_plugin)]
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        load_plugins()
        # During execution: source was set (not None)
        self.assertEqual(len(sources_seen), 1)
        self.assertIsNotNone(sources_seen[0])
        self.assertEqual(sources_seen[0].plugin_name, "spy")
        # After execution: source is back to None
        self.assertIsNone(_current_plugin_source.get())

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_current_plugin_source_reset_after_exception(
        self, mock_eps, mock_envs, mock_registry
    ):
        """_current_plugin_source is reset to None even when a plugin raises."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        def bad_plugin():
            raise RuntimeError("boom")

        mock_eps.return_value = [_make_ep("bad", load_fn=bad_plugin)]

        load_plugins()
        self.assertIsNone(_current_plugin_source.get())

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_missing_required_plugins(self, mock_eps, mock_envs, mock_registry):
        """load_plugins must raise when required plugins are not discovered."""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical", "important")
        mock_eps.return_value = [_make_ep("alpha"), _make_ep("important")]
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        with self.assertRaisesRegex(RuntimeError, "critical"):
            load_plugins()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points", return_value=[])
    def test_missing_required_plugins_with_no_plugins_installed(
        self, mock_eps, mock_envs, mock_registry
    ):
        """load_plugins must raise when no general plugin is installed at all."""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        with self.assertRaisesRegex(RuntimeError, "critical"):
            load_plugins()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_all_required_plugins_present(self, mock_eps, mock_envs, mock_registry):
        """load_plugins runs successfully when all required plugins are loaded."""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical", "important")
        mock_eps.return_value = [_make_ep("critical"), _make_ep("important")]
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        load_plugins()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_plugins_exclude_by_sglang_plugins(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Raises when a required plugin is excluded by SGLANG_PLUGINS."""
        mock_envs.SGLANG_PLUGINS.get.return_value = "critical,white_list"
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical", "important")
        mock_eps.return_value = [_make_ep("critical"), _make_ep("important")]
        mock_envs.SGLANG_PLATFORM.get.return_value = ""

        with self.assertRaisesRegex(RuntimeError, "important.*SGLANG_PLUGINS"):
            load_plugins()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_plugins_exclude_by_sglang_platform(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Raises when a required plugin is excluded by SGLANG_PLATFORM."""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_PLATFORM.get.return_value = "kahsieh"
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical", "important")
        platform_eps = [
            _make_ep("kahsieh", dist_name="kahsieh-pkg"),
            _make_ep("other_hw", dist_name="other-pkg"),
        ]
        plugin_eps = [
            _make_ep("critical", dist_name="kahsieh-pkg"),
            _make_ep("important", dist_name="other-pkg"),
        ]
        mock_eps.side_effect = lambda group: (
            platform_eps if group == "sglang.srt.platforms" else plugin_eps
        )

        with self.assertRaisesRegex(RuntimeError, "important.*SGLANG_PLATFORM"):
            load_plugins()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_plugins_fail_to_load(self, mock_eps, mock_envs, mock_registry):
        """load_plugins must raise when required plugins fail to load."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)
        ep = _make_ep("critical")
        mock_eps.return_value = [ep]
        ep.load.side_effect = ImportError("boom!")

        with self.assertRaisesRegex(RuntimeError, "critical"):
            load_plugins()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_plugins_fail_to_run(self, mock_eps, mock_envs, mock_registry):
        """load_plugins must raise when required plugins fail to run."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)

        def bad_fn():
            raise ValueError("boom!")

        mock_eps.return_value = [_make_ep("critical", load_fn=bad_fn)]

        with self.assertRaisesRegex(RuntimeError, "critical"):
            load_plugins()


def bad_hook_plugin():
    HookRegistry.register(
        "sglang.srt.plugins.no_such_function", lambda result, *a, **k: result
    )


# Imported by apply_hooks() while it resolves "lazy_hook_mod.kernel"; registering
# here adds a target after apply_hooks() has listed the targets it will visit.
_LAZY_HOOK_MODULE = """\
from sglang.srt.plugins.hook_registry import HookRegistry


def kernel():
    return "k"


HookRegistry.register(
    "sglang.srt.plugins.no_such_function", lambda result: result, required=True
)
"""


class TestLoadPluginsWithRealHookRegistry(CustomTestCase):
    """Tests for load_plugins() with real `HookRegistry`"""

    def setUp(self):
        _reset_plugins_loaded()
        HookRegistry.reset()

    def tearDown(self):
        _reset_plugins_loaded()
        HookRegistry.reset()

    def _make_target(self):
        """Create a throwaway module with a function hooks can wrap."""
        mod = types.ModuleType("late_hook_target")
        vars(mod)["fn"] = lambda: "orig"
        sys.modules[mod.__name__] = mod
        self.addCleanup(sys.modules.pop, mod.__name__, None)
        return mod, f"{mod.__name__}.fn"

    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_plugins_fail_to_apply(self, mock_eps, mock_envs):
        """A failed hook from a required plugin raises an exception."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)

        mock_eps.return_value = [_make_ep("critical", load_fn=bad_hook_plugin)]

        with self.assertRaisesRegex(RuntimeError, "critical.*no_such_function"):
            load_plugins()

    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_non_required_plugin_fail_to_apply_does_not_raise(
        self, mock_eps, mock_envs
    ):
        """A failed hook from a non-required plugin is only logged."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)

        mock_eps.return_value = [
            _make_ep("critical"),
            _make_ep("trivial", load_fn=bad_hook_plugin),
        ]

        load_plugins()

    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_plugin_import_time_hook_fail_to_apply(self, mock_eps, mock_envs):
        """A hook registered while importing a required plugin is still attributed to it,
        so a failure to apply it raises instead of being skipped as unowned."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)

        def import_module_with_hook():
            bad_hook_plugin()
            return lambda: None

        ep = _make_ep("critical")
        ep.load.side_effect = import_module_with_hook
        mock_eps.return_value = [ep]

        with self.assertRaisesRegex(RuntimeError, "critical.*no_such_function"):
            load_plugins()

    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_late_required_plugins(self, mock_eps, mock_envs):
        """Raise when a required hook registers too late."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)

        _, target = self._make_target()

        def early_apply_plugin():
            HookRegistry.register(target, lambda result: result)
            HookRegistry.apply_hooks()

        def critical_plugin():
            HookRegistry.register(target, lambda result: result)

        mock_eps.return_value = [
            _make_ep("trivial", load_fn=early_apply_plugin),
            _make_ep("critical", load_fn=critical_plugin),
        ]

        with self.assertRaisesRegex(RuntimeError, "critical -> late_hook_target.fn"):
            load_plugins()

        self.assertIn(target, HookRegistry.failed_targets())

    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_hook_applied_by_early_apply_does_not_raise(
        self, mock_eps, mock_envs
    ):
        """A non-required hook that is applied late must not abort startup."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ("critical",)

        mod, target = self._make_target()
        results = []

        def critical_plugin():
            HookRegistry.register(target, lambda result: results.append(result))
            HookRegistry.apply_hooks()

        def late_plugin():
            HookRegistry.register(target, lambda result: result)

        mock_eps.return_value = [
            _make_ep("critical", load_fn=critical_plugin),
            _make_ep("late", load_fn=late_plugin),
        ]

        load_plugins()
        mod.fn()
        self.assertEqual(results, ["orig"])

    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points", return_value=[])
    def test_required_hook_fail_to_apply_without_required_plugins(
        self, mock_eps, mock_envs
    ):
        """Raise when hooks with register(required=True) fail to apply."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ()

        HookRegistry.register(
            "sglang.srt.plugins.no_such_function",
            lambda result: result,
            required=True,
        )

        with self.assertRaisesRegex(
            RuntimeError, "unknown -> sglang.srt.plugins.no_such_function"
        ):
            load_plugins()

    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_required_hook_registered_during_apply_raises(self, mock_eps, mock_envs):
        """A required hook registered while apply_hooks() imports a target module
        lands on a target that call never visits; it must still abort startup."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        mock_envs.SGLANG_REQUIRED_PLUGINS.get.return_value = ()

        module_dir = self.enterContext(tempfile.TemporaryDirectory())
        Path(module_dir, "lazy_hook_mod.py").write_text(_LAZY_HOOK_MODULE)
        self.enterContext(patch.object(sys, "path", [module_dir, *sys.path]))
        self.addCleanup(sys.modules.pop, "lazy_hook_mod", None)

        def offload_plugin():
            HookRegistry.register("lazy_hook_mod.kernel", lambda result: result)

        mock_eps.return_value = [_make_ep("offload", load_fn=offload_plugin)]

        with self.assertRaisesRegex(
            RuntimeError, "unknown -> sglang.srt.plugins.no_such_function"
        ):
            load_plugins()


if __name__ == "__main__":
    import unittest

    unittest.main()
