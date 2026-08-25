"""Import smoke for the Bazel-owned, accelerator-free runtime boundary."""

import contextlib
import io
import pathlib
import sys
import unittest
from unittest import mock

import sglang
from sglang.cli.main import main as cli_main
from sglang.kernels.registry import KernelRegistry
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.environ import envs
from sglang.utils import LazyImport


class RuntimeImportTest(unittest.TestCase):
    def test_public_api_and_runtime_boundaries_come_from_runfiles(self):
        self.assertEqual(sys.version_info[:2], (3, 12))
        self.assertIn("sglang", pathlib.Path(sglang.__file__).parts)
        self.assertIsInstance(sglang.Engine, LazyImport)
        self.assertEqual(Engine.__module__, "sglang.srt.entrypoints.engine")
        self.assertIsNotNone(KernelRegistry)
        self.assertIsNotNone(envs)

    def test_version_cli_dispatches_through_public_package(self):
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["sglang", "version"]),
            contextlib.redirect_stdout(output),
        ):
            cli_main()

        self.assertIn("sglang version:", output.getvalue())
        self.assertIn("git revision:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
