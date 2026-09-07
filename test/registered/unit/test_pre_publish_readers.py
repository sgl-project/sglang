"""The launch path runs before `publish`, so nothing on it may ask a config bag.

`publish` is what projects the bags; every accessor fails closed until then
(`config namespace 'observability' not published`). Most readers live deep in a
runtime path and are safely downstream of it, so converting one to a bag is
normally free. The launcher's own reads are not: everything
`_launch_subprocesses` calls before its `publish` runs with no bags at all, and
`multimodal_gen` calls into the same code with a `ServerArgs` of its own that
never publishes them.

This is a class no other test in the tree catches: a converted reader is
exercised everywhere by tests that publish first, so it passes unit CI and then
takes the server down on the first real launch -- which is how `configure_logger`
shipped. So the protected set is *derived from the launch path* rather than
listed here: the callees named in `_launch_subprocesses` above its `publish` are
read out of the source, and each is called against a context where nothing has
been published. A conversion of any of them turns this red without a boot.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import ast
import logging
import pathlib
import unittest

import sglang
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.runtime_context import get_observability, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import configure_logger
from sglang.test.test_utils import CustomTestCase

# Callees this guard can drive directly. `_set_envs_and_config` is here too: it
# is the other one that runs on every launch before anything is published.
_EXERCISED = {
    "configure_logger": configure_logger,
    "_set_envs_and_config": _set_envs_and_config,
}

# Named by the launcher before `publish`, but none of them reads config out of a
# record. Listed so the set above stays a statement about all of the callees.
_NOT_EXERCISED = {
    "load_plugins",
    "resolve_auto_parsers",
    "snapshot_context",
    "resolving_view",
}


def _pre_publish_callees():
    """Functions `_launch_subprocesses` calls before it publishes.

    Read from the source so the set cannot go stale: a call added above the
    `publish(...)` line joins the protected set on its own.
    """
    source = (
        pathlib.Path(next(iter(sglang.__path__))) / "srt" / "entrypoints" / "engine.py"
    ).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    launcher = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_launch_subprocesses"
    )
    publish_line = min(
        node.lineno
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "publish"
    )
    return {
        node.func.id
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.lineno < publish_line
    }


class TestPrePublishReaders(CustomTestCase):
    def setUp(self):
        self._levels = {
            name: logging.getLogger(name).level
            for name in (None, "sglang", "httpx", "httpcore")
        }
        reset_context()
        self.addCleanup(self._restore_levels)
        self.addCleanup(reset_context)

    def _restore_levels(self):
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)

    def test_nothing_is_published_here(self):
        """The premise: this fixture really is a pre-publish context."""
        with self.assertRaises(ValueError) as caught:
            get_observability()
        self.assertIn("not published", str(caught.exception))

    def test_the_protected_set_is_what_the_launcher_calls(self):
        """If the launcher stops calling one of these, or starts calling
        something new before publishing, this file has to be looked at."""
        self.assertEqual(set(_EXERCISED) | _NOT_EXERCISED, _pre_publish_callees())

    def test_none_of_them_asks_a_bag(self):
        """Each is called with nothing published. A bag read raises
        `config namespace ... not published` -- any other failure is the
        callee's own business and does not belong to this guard."""
        server_args = ServerArgs(model_path="dummy", log_level="warning")
        for name, call in sorted(_EXERCISED.items()):
            with self.subTest(callee=name):
                reset_context()
                try:
                    call(server_args)
                except Exception as exc:  # noqa: BLE001 -- see the docstring
                    self.assertNotIn(
                        "not published",
                        str(exc),
                        f"{name} runs before publish and asked a config bag",
                    )

    def test_configure_logger_reads_the_record_it_was_handed(self):
        """The one that shipped broken, pinned by value rather than by not
        raising: `multimodal_gen` hands it a ServerArgs that never publishes."""
        configure_logger(ServerArgs(model_path="dummy", log_level="warning"))
        self.assertEqual(logging.getLogger().level, logging.WARNING)
        configure_logger(ServerArgs(model_path="dummy", log_level="error"))
        self.assertEqual(logging.getLogger().level, logging.ERROR)


if __name__ == "__main__":
    unittest.main()
