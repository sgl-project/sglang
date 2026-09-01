import importlib.util
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _load_bridge():
    path = pathlib.Path("python/sglang/test/ci/diffusion_suite_bridge.py")
    spec = importlib.util.spec_from_file_location("diffusion_suite_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiffusionSuiteBridge(unittest.TestCase):
    def test_forwards_partition_environment_and_ignores_file_runner_flags(self):
        bridge = _load_bridge()
        captured = []
        fake_runner = SimpleNamespace(main=lambda: captured.append(list(sys.argv)))
        environment = {
            "DIFFUSION_PARTITION_ID": "2",
            "DIFFUSION_TOTAL_PARTITIONS": "4",
            "DIFFUSION_PARTITION_PLAN_JSON": '{"suite":"1-gpu"}',
            "DIFFUSION_CONTINUE_ON_ERROR": "true",
        }

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(
                sys.modules,
                {
                    "sglang.multimodal_gen.test.runner.diffusion_suite_runner": fake_runner
                },
            ),
            patch("os.chdir") as change_directory,
            patch.object(sys, "argv", ["bridge.py", "-f"]),
        ):
            bridge.run_diffusion_suite("1-gpu")

        self.assertEqual(
            captured,
            [
                [
                    "bridge.py",
                    "--suite",
                    "1-gpu",
                    "--partition-id",
                    "2",
                    "--total-partitions",
                    "4",
                    "--partition-plan-json",
                    '{"suite":"1-gpu"}',
                    "--continue-on-error",
                ]
            ],
        )
        change_directory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
