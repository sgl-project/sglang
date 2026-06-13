"""
Unit tests for the plugin loading flow.

Covers: startup plugin loading, idempotency, apply_hooks invocation,
exception resilience, SGLANG_PLUGINS whitelist, SGLANG_PLATFORM exclusion
logic, and _current_plugin_source context var reset.

Run:  python -m pytest test/registered/unit/plugins/test_load_plugins.py -v
"""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from sglang.srt.plugins import (
    GENERAL_PLUGINS_GROUP,
    STARTUP_PLUGINS_GROUP,
    _current_plugin_source,
    _get_excluded_dists,
    load_plugins,
    load_plugins_by_group,
    load_startup_plugins,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


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
    """Reset plugin load flags so plugin loaders can run again."""
    import sglang.srt.plugins as plugins_mod

    plugins_mod._startup_plugins_loaded = False
    plugins_mod._plugins_loaded = False


class TestLoadPlugins(TestCase):
    """Tests for load_plugins() and related helpers."""

    def setUp(self):
        _reset_plugins_loaded()

    def tearDown(self):
        _reset_plugins_loaded()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points", return_value=[])
    def test_load_startup_plugins_no_entrypoints(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Startup loader is safe when no startup plugins are installed."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        load_startup_plugins()

        mock_eps.assert_called_once_with(group=STARTUP_PLUGINS_GROUP)
        mock_registry.apply_hooks.assert_not_called()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_load_startup_plugins_executes_once(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Startup plugins execute once and do not apply runtime hooks."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        startup_fn = MagicMock()
        mock_eps.return_value = [_make_ep("startup", load_fn=startup_fn)]

        load_startup_plugins()
        load_startup_plugins()

        startup_fn.assert_called_once()
        mock_registry.apply_hooks.assert_not_called()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_load_startup_plugins_filters_by_platform(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Startup plugins from unselected platform dists are not imported."""
        mock_envs.SGLANG_PLATFORM.get.return_value = "kunlun"
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        selected_fn = MagicMock()
        other_fn = MagicMock()
        platform_eps = [
            _make_ep("kunlun", dist_name="kunlun-pkg"),
            _make_ep("other_hw", dist_name="other-pkg"),
        ]
        startup_eps = [
            _make_ep("kunlun_startup", dist_name="kunlun-pkg", load_fn=selected_fn),
            _make_ep("other_startup", dist_name="other-pkg", load_fn=other_fn),
        ]

        def entry_points_side_effect(group):
            if group == "sglang.srt.platforms":
                return platform_eps
            if group == STARTUP_PLUGINS_GROUP:
                return startup_eps
            return []

        mock_eps.side_effect = entry_points_side_effect

        load_startup_plugins()

        selected_fn.assert_called_once()
        other_fn.assert_not_called()
        startup_eps[1].load.assert_not_called()
        mock_registry.apply_hooks.assert_not_called()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_load_startup_plugins_propagates_load_exception(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Startup entry point import failures are surfaced to stop startup."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        bad_ep = _make_ep("bad_startup")
        bad_ep.load.side_effect = RuntimeError("startup import boom")
        mock_eps.return_value = [bad_ep]

        with self.assertRaisesRegex(RuntimeError, "startup import boom"):
            load_startup_plugins()
        mock_registry.apply_hooks.assert_not_called()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_load_startup_plugins_propagates_execute_exception(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Startup plugin execution failures are surfaced to stop startup."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""

        def bad_startup_plugin():
            raise RuntimeError("startup boom")

        mock_eps.return_value = [_make_ep("bad_startup", load_fn=bad_startup_plugin)]

        with self.assertRaisesRegex(RuntimeError, "startup boom"):
            load_startup_plugins()
        mock_registry.apply_hooks.assert_not_called()

    @patch("sglang.srt.plugins.HookRegistry")
    @patch("sglang.srt.plugins.envs")
    @patch("sglang.srt.plugins.entry_points")
    def test_load_startup_plugins_can_retry_after_failure(
        self, mock_eps, mock_envs, mock_registry
    ):
        """Startup loader is not marked loaded until plugins execute successfully."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        calls = []

        def flaky_startup_plugin():
            calls.append("call")
            if len(calls) == 1:
                raise RuntimeError("startup boom")

        mock_eps.return_value = [
            _make_ep("flaky_startup", load_fn=flaky_startup_plugin)
        ]

        with self.assertRaisesRegex(RuntimeError, "startup boom"):
            load_startup_plugins()
        load_startup_plugins()

        self.assertEqual(calls, ["call", "call"])
        mock_registry.apply_hooks.assert_not_called()

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
    def test_load_plugins_loads_startup_before_general(
        self, mock_eps, mock_envs, mock_registry
    ):
        """General plugin loading first executes startup import-time shims."""
        mock_envs.SGLANG_PLATFORM.get.return_value = ""
        mock_envs.SGLANG_PLUGINS.get.return_value = ""
        calls = []

        def startup_plugin():
            calls.append("startup")

        def general_plugin():
            calls.append("general")

        def entry_points_side_effect(group):
            if group == STARTUP_PLUGINS_GROUP:
                return [_make_ep("startup", load_fn=startup_plugin)]
            if group == GENERAL_PLUGINS_GROUP:
                return [_make_ep("general", load_fn=general_plugin)]
            return []

        mock_eps.side_effect = entry_points_side_effect

        load_plugins()

        self.assertEqual(calls, ["startup", "general"])
        mock_registry.apply_hooks.assert_called_once()

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

        def entry_points_side_effect(group):
            if group == GENERAL_PLUGINS_GROUP:
                return eps
            return []

        mock_eps.side_effect = entry_points_side_effect

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

        eps = [_make_ep("spy", load_fn=spy_plugin)]

        def entry_points_side_effect(group):
            if group == GENERAL_PLUGINS_GROUP:
                return eps
            return []

        mock_eps.side_effect = entry_points_side_effect
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

        eps = [_make_ep("bad", load_fn=bad_plugin)]

        def entry_points_side_effect(group):
            if group == GENERAL_PLUGINS_GROUP:
                return eps
            return []

        mock_eps.side_effect = entry_points_side_effect

        load_plugins()
        self.assertIsNone(_current_plugin_source.get())


if __name__ == "__main__":
    import unittest

    unittest.main()
