"""Regression tests for the SGLANG_USE_MODELSCOPE switch in the benchmark clients."""

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from sglang.benchmark import offline_throughput
from sglang.benchmark.utils import get_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

MODEL_ID = "org/model"
MODELSCOPE_PATH = "/modelscope-cache/org/model"


def _fake_modelscope() -> ModuleType:
    module = ModuleType("modelscope")
    module.snapshot_download = lambda *args, **kwargs: MODELSCOPE_PATH
    return module


class _ReachedServerArgs(Exception):
    pass


def _stop_at_server_args(args):
    raise _ReachedServerArgs(args.model_path)


class TestBenchmarkModelScopeSwitch(CustomTestCase):
    """The benchmark clients must read SGLANG_USE_MODELSCOPE as the server does
    (true/1/yes/y); otherwise the client loads its tokenizer from the HF hub while
    the server downloads the model from ModelScope."""

    def setUp(self):
        patcher = patch.dict(sys.modules, {"modelscope": _fake_modelscope()})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_get_model_follows_the_declared_boolean(self):
        cases = [
            ("1", MODELSCOPE_PATH),
            ("true", MODELSCOPE_PATH),
            ("yes", MODELSCOPE_PATH),
            ("0", MODEL_ID),
        ]
        for value, expected in cases:
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"SGLANG_USE_MODELSCOPE": value}),
            ):
                self.assertEqual(get_model(MODEL_ID), expected)

    def test_offline_throughput_downloads_from_modelscope(self):
        argv = ["offline_throughput", "--model-path", MODEL_ID]
        with (
            patch.dict(os.environ, {"SGLANG_USE_MODELSCOPE": "yes"}),
            patch.object(sys, "argv", argv),
            patch.object(
                offline_throughput.ServerArgs, "from_cli_args", _stop_at_server_args
            ),
            self.assertRaises(_ReachedServerArgs) as reached,
        ):
            offline_throughput.cli_main()

        self.assertEqual(str(reached.exception), MODELSCOPE_PATH)


if __name__ == "__main__":
    unittest.main()
