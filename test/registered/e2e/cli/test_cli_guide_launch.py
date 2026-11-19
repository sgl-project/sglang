# SPDX-License-Identifier: Apache-2.0
"""Nightly: bare `sglang serve MODEL` prints its Cookbook command and serves.

Launches one small model the way an agent does (no terminal, nothing but the
model name). That every exported command parses and matches the Cookbook is
covered by the CPU unit tests in test/registered/unit/cli.
"""

import subprocess
import sys
import tempfile
import unittest

from sglang.cli import interactive
from sglang.cli.interactive import command_argv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, terminate_and_kill_process_tree
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=300, stage="nightly", runner_config="1-gpu-large")

# The smallest checkpoint (0.5 GB) with a verified 1-GPU H100 Cookbook command.
MODEL = "LiquidAI/LFM2.5-230M"
# Includes downloading the checkpoint.
LAUNCH_TIMEOUT = 900


class TestCliGuideLaunch(CustomTestCase):
    def test_bare_serve_launches_the_cookbook_command(self):
        recipes = interactive.recipes_for_this_machine(MODEL)
        self.assertTrue(recipes, f"No Cookbook command for {MODEL} on this runner.")
        recipe = interactive._pick_mode(recipes=recipes, interactive=False)
        argv = command_argv(recipe["command"])
        port = argv[argv.index("--port") + 1] if "--port" in argv else "30000"
        with tempfile.TemporaryFile("w+") as stdout:
            process = subprocess.Popen(
                [sys.executable, "-m", "sglang.cli.main", "serve", MODEL],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
            )
            try:
                wait_for_http_ready(
                    url=f"http://127.0.0.1:{port}/health_generate",
                    timeout=LAUNCH_TIMEOUT,
                    process=process,
                )
                stdout.seek(0)
                self.assertTrue(
                    stdout.read().startswith(recipe["command"] + "\n"),
                    "The Guide did not print the Cookbook command it launched.",
                )
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main(verbosity=3)
