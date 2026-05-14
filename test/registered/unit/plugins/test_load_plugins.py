"""
Unit tests for the plugin loading flow.

Covers: idempotency, apply_hooks invocation, exception resilience,
SGLANG_PLUGINS whitelist, SGLANG_PLATFORM exclusion logic,
and _current_plugin_source context var reset.

Run:  python -m pytest test/registered/unit/plugins/test_load_plugins.py -v
"""

from unittest.mock import MagicMock, patch

from sglang.srt.plugins import (
    _current_plugin_source,
    _get_excluded_dists,
    load_plugins,
    load_plugins_by_group,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")


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


if __name__ == "__main__":
    import unittest

    unittest.main()
