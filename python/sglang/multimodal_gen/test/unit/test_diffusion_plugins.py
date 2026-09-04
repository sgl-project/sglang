"""
Unit tests for the sglang-diffusion plugin framework.

Run:
  python -m pytest python/sglang/multimodal_gen/test/unit/test_diffusion_plugins.py -v
"""

import ast
import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import sglang
import sglang.multimodal_gen
import sglang.multimodal_gen.plugins as diffusion_plugins
from sglang.multimodal_gen.plugins import (
    _get_excluded_dists,
    discover_diffusion_plugins,
    load_diffusion_plugins,
)
from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import (
    HookRegistry,
    HookType,
    _current_plugin_source,
)

FAKE_TARGET_MODULE = "fake_diffusion_hook_target"

# Discovery lives in sglang.srt.plugins now, so entry_points must be patched
# where it is actually called.
SRT_ENTRY_POINTS = "sglang.srt.plugins.entry_points"


def _make_ep(name, dist_name=None, load_fn=None, load_raises=False):
    """Build a mock importlib.metadata.EntryPoint."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"fake_module:{name}"
    ep.dist = MagicMock()
    ep.dist.name = dist_name or f"{name}-dist"
    if load_raises:
        ep.load.side_effect = RuntimeError("boom on load")
    else:
        ep.load.return_value = load_fn if load_fn is not None else MagicMock()
    return ep


class _DiffusionPluginTestCase(unittest.TestCase):
    """Shared setup: env overrides, plugin-load flag and registry isolation."""

    def setUp(self):
        self._stack = contextlib.ExitStack()
        self.addCleanup(self._stack.close)

        # Save/restore the process-global registry contents instead of
        # resetting it: a real installed plugin may own entries here.
        saved_hooks = {
            target: list(hooks) for target, hooks in HookRegistry._hooks.items()
        }
        saved_patched = set(HookRegistry._patched)
        saved_applied = dict(HookRegistry._applied)
        self.addCleanup(
            self._restore_registry, saved_hooks, saved_patched, saved_applied
        )
        HookRegistry._hooks.clear()
        HookRegistry._patched.clear()
        HookRegistry._applied.clear()

        diffusion_plugins._plugins_loaded = False
        self.addCleanup(setattr, diffusion_plugins, "_plugins_loaded", False)

    @staticmethod
    def _restore_registry(saved_hooks, saved_patched, saved_applied):
        HookRegistry._hooks.clear()
        HookRegistry._hooks.update(saved_hooks)
        HookRegistry._patched.clear()
        HookRegistry._patched.update(saved_patched)
        HookRegistry._applied.clear()
        HookRegistry._applied.update(saved_applied)

    def set_envs(self, platform="", plugins=""):
        """Override the real diffusion env descriptors for this test."""
        self._stack.enter_context(envs.SGLANG_DIFFUSION_PLATFORM.override(platform))
        self._stack.enter_context(envs.SGLANG_DIFFUSION_PLUGINS.override(plugins))


class TestDiffusionPluginLoading(_DiffusionPluginTestCase):
    def tearDown(self):
        sys.modules.pop(FAKE_TARGET_MODULE, None)

    def test_load_is_idempotent_and_applies_hooks(self):
        self.set_envs()
        with patch(SRT_ENTRY_POINTS, return_value=[]), patch(
            "sglang.multimodal_gen.plugins.HookRegistry"
        ) as mock_registry:
            load_diffusion_plugins()
            self.assertEqual(mock_registry.apply_hooks.call_count, 1)
            load_diffusion_plugins()
            self.assertEqual(mock_registry.apply_hooks.call_count, 1)

    def test_plugin_exception_is_isolated(self):
        calls = []

        def bad():
            calls.append("bad")
            raise RuntimeError("plugin blew up")

        def good():
            calls.append("good")

        eps = [_make_ep("bad", load_fn=bad), _make_ep("good", load_fn=good)]
        self.set_envs()
        with patch(SRT_ENTRY_POINTS, return_value=eps), patch(
            "sglang.multimodal_gen.plugins.HookRegistry"
        ):
            load_diffusion_plugins()

        self.assertEqual(calls, ["bad", "good"])

    def test_plugin_source_contextvar_is_reset(self):
        eps = [_make_ep("noop", load_fn=lambda: None)]
        self.set_envs()
        with patch(SRT_ENTRY_POINTS, return_value=eps), patch(
            "sglang.multimodal_gen.plugins.HookRegistry"
        ):
            load_diffusion_plugins()

        self.assertIsNone(_current_plugin_source.get())

    def test_whitelist_filters_by_entry_point_name(self):
        eps = [_make_ep("a"), _make_ep("b")]
        self.set_envs(plugins="b")
        with patch(SRT_ENTRY_POINTS, return_value=eps):
            found = discover_diffusion_plugins("any.group")

        self.assertEqual(list(found), ["b"])

    def test_srt_whitelist_does_not_filter_diffusion_plugins(self):
        """SGLANG_PLUGINS must not gate the diffusion group."""
        eps = [_make_ep("a"), _make_ep("b")]
        self.set_envs()
        self._stack.enter_context(envs.SGLANG_PLUGINS.override("b"))
        with patch(SRT_ENTRY_POINTS, return_value=eps):
            found = discover_diffusion_plugins("any.group")

        self.assertEqual(sorted(found), ["a", "b"])

    def test_excluded_dist_is_never_loaded(self):
        ep_a = _make_ep("a", dist_name="a-dist")
        ep_b = _make_ep("b", dist_name="b-dist")
        self.set_envs()
        with patch(SRT_ENTRY_POINTS, return_value=[ep_a, ep_b]):
            found = discover_diffusion_plugins("any.group", excluded_dists={"a-dist"})

        self.assertEqual(list(found), ["b"])
        ep_a.load.assert_not_called()
        ep_b.load.assert_called_once()

    def test_entry_point_load_failure_is_isolated(self):
        ep_bad = _make_ep("bad", load_raises=True)
        ep_ok = _make_ep("ok")
        self.set_envs()
        with patch(SRT_ENTRY_POINTS, return_value=[ep_bad, ep_ok]):
            found = discover_diffusion_plugins("any.group")

        self.assertEqual(list(found), ["ok"])

    def test_get_excluded_dists_returns_other_platform_dists(self):
        eps = [
            _make_ep("klx", dist_name="klx-dist"),
            _make_ep("other", dist_name="other-dist"),
        ]
        self.set_envs(platform="klx")
        with patch(SRT_ENTRY_POINTS, return_value=eps):
            self.assertEqual(_get_excluded_dists(), {"other-dist"})

    def test_get_excluded_dists_empty_when_platform_unset(self):
        self.set_envs(platform="")
        with patch(SRT_ENTRY_POINTS) as mock_eps:
            self.assertEqual(_get_excluded_dists(), set())
            mock_eps.assert_not_called()

    def test_discovery_failure_does_not_break_import(self):
        """Discovery is called from the package __init__: it must never raise."""
        self.set_envs()
        with patch(
            "sglang.multimodal_gen.plugins.discover_diffusion_plugins",
            side_effect=RuntimeError("entry_points blew up"),
        ), patch("sglang.multimodal_gen.plugins.HookRegistry") as mock_registry:
            with self.assertLogs("sglang.multimodal_gen.plugins", level="ERROR") as cm:
                load_diffusion_plugins()

        self.assertTrue(any("discovery failed" in msg for msg in cm.output))
        self.assertEqual(mock_registry.apply_hooks.call_count, 1)

    def test_registered_hook_is_actually_applied(self):
        target_mod = types.ModuleType(FAKE_TARGET_MODULE)

        def double(x):
            return x * 2

        target_mod.double = double
        sys.modules[FAKE_TARGET_MODULE] = target_mod

        def plugin():
            def around(original_fn, *args, **kwargs):
                return original_fn(*args, **kwargs) + 1

            HookRegistry.register(
                f"{FAKE_TARGET_MODULE}.double", around, HookType.AROUND
            )

        eps = [_make_ep("hooky", load_fn=plugin)]
        self.set_envs()
        with patch(SRT_ENTRY_POINTS, return_value=eps):
            load_diffusion_plugins()

        self.assertEqual(target_mod.double(3), 7)


class TestPackageLoadPoint(unittest.TestCase):
    """The package __init__ must load plugins before importing the runtime."""

    def _package_dir(self) -> pathlib.Path:
        return pathlib.Path(sglang.multimodal_gen.__file__).parent

    def test_load_call_precedes_other_diffusion_imports(self):
        tree = ast.parse((self._package_dir() / "__init__.py").read_text())
        call_index = None
        first_other_import_index = None
        for index, node in enumerate(tree.body):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "load_diffusion_plugins"
            ):
                call_index = index
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (
                    module.startswith("sglang.multimodal_gen")
                    and module != "sglang.multimodal_gen.plugins"
                    and first_other_import_index is None
                ):
                    first_other_import_index = index
        self.assertIsNotNone(call_index, "load_diffusion_plugins() call not found")
        self.assertIsNotNone(
            first_other_import_index, "no other sglang.multimodal_gen import found"
        )
        self.assertLess(call_index, first_other_import_index)

    def test_plugins_module_does_not_import_runtime_at_module_scope(self):
        tree = ast.parse((self._package_dir() / "plugins" / "__init__.py").read_text())
        imported: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        for name in imported:
            self.assertFalse(
                name.startswith("sglang.multimodal_gen.runtime"),
                f"plugins module must not import {name} at module scope",
            )


PLUGIN_MODULE_NAME = "fake_diffusion_plugin_pkg"
PLUGIN_MODULE_TEMPLATE = '''\
"""Throwaway diffusion plugin used by the fresh-process test."""

import pathlib

MARKER = pathlib.Path({marker!r})


def target():
    return "original"


def _after(result, *args, **kwargs):
    return result + "+hooked"


def register():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register("{module}.target", _after, HookType.AFTER)
    MARKER.write_text("register-ran")
'''

CHILD_CODE = """\
import sys

import sglang.multimodal_gen  # noqa: F401  (the only trigger)
import sglang.multimodal_gen.plugins as p

assert p._plugins_loaded, "package __init__ did not load plugins"
assert "{module}" in sys.modules, "plugin module was never imported by the loader"

import {module}

print("HOOK_RESULT:" + {module}.target())
"""


class TestFreshProcessPluginLoading(unittest.TestCase):
    """A brand-new interpreter must discover, run and APPLY plugin hooks.

    The plugin is materialized as a throwaway installed-looking distribution
    (module file + ``*.dist-info`` with ``entry_points.txt``) in a temp dir put
    first on the child's PYTHONPATH, so importlib.metadata discovers it. The
    parent never imports that module (asserted below), so both the marker file
    and the hooked return value can only come from the child.
    """

    def _write_plugin_dist(self, root: pathlib.Path, marker: pathlib.Path) -> None:
        (root / f"{PLUGIN_MODULE_NAME}.py").write_text(
            PLUGIN_MODULE_TEMPLATE.format(marker=str(marker), module=PLUGIN_MODULE_NAME)
        )
        dist_info = root / "fake_diffusion_plugin-0.0.1.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: fake-diffusion-plugin\nVersion: 0.0.1\n"
        )
        (dist_info / "entry_points.txt").write_text(
            f"[{diffusion_plugins.DIFFUSION_GENERAL_PLUGINS_GROUP}]\n"
            f"fake_diffusion = {PLUGIN_MODULE_NAME}:register\n"
        )

    def test_importing_package_loads_and_applies_plugins_in_a_fresh_process(self):
        repo_python = pathlib.Path(sglang.__file__).parents[1]
        self.assertNotIn(
            PLUGIN_MODULE_NAME, sys.modules, "parent must not know the fake plugin"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            marker = tmp_path / "register_ran.txt"
            self._write_plugin_dist(tmp_path, marker)
            self.assertFalse(marker.exists())

            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(tmp_path), str(repo_python), env.get("PYTHONPATH", "")]
            ).strip(os.pathsep)
            # Do not let an inherited whitelist filter the fake plugin out.
            env["SGLANG_DIFFUSION_PLUGINS"] = ""
            env["SGLANG_DIFFUSION_PLATFORM"] = ""
            proc = subprocess.run(
                [sys.executable, "-c", CHILD_CODE.format(module=PLUGIN_MODULE_NAME)],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr[-3000:])
            # register() ran inside the child ...
            self.assertTrue(marker.exists(), msg=proc.stderr[-3000:])
            self.assertEqual(marker.read_text(), "register-ran")
            # ... and its hook was actually APPLIED, not merely registered.
            self.assertIn("HOOK_RESULT:original+hooked", proc.stdout)


if __name__ == "__main__":
    unittest.main()
