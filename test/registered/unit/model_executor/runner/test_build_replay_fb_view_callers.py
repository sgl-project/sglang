import ast
import inspect
import unittest
from pathlib import Path

import sglang.srt
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    build_replay_fb_view,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _calls():
    """Every build_replay_fb_view(...) call written in sglang.srt, as
    (file, line, positional count, keyword names)."""
    root = Path(sglang.srt.__path__[0])
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "build_replay_fb_view(" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_replay_fb_view"
            ):
                yield path, node.lineno, len(node.args), [k.arg for k in node.keywords]


class TestBuildReplayFbViewCallers(CustomTestCase):
    """The CUDA and NPU graph runners both build their replay view with
    build_replay_fb_view. The NPU DFlash replay call only runs on Ascend, so a
    parameter added to the builder can leave it behind and raise a TypeError
    there while every CPU and CUDA test passes. Each call must bind."""

    def test_every_call_binds_to_the_signature(self):
        signature = inspect.signature(build_replay_fb_view)
        calls = list(_calls())
        self.assertGreaterEqual(len(calls), 2)
        for path, line, n_args, keywords in calls:
            with self.subTest(call=f"{path.name}:{line}"):
                self.assertNotIn(None, keywords)
                signature.bind(*([None] * n_args), **dict.fromkeys(keywords))


if __name__ == "__main__":
    unittest.main()
