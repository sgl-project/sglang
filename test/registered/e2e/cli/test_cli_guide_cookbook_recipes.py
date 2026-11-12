# SPDX-License-Identifier: Apache-2.0
"""Nightly: `sglang serve MODEL` runs its Cookbook command at the page's GSM8K.

Each run samples Cookbook commands the CLI Guide picks on this runner's GPUs,
seeded by the date so a failing night can be rerun. It launches each one the
way an agent does (no terminal, nothing but the model name), then checks that
the Guide printed exactly the Cookbook command and that the server scores
within GSM8K_TOLERANCE of the GSM8K the Cookbook page reports for it.
"""

import contextlib
import datetime
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from importlib import resources
from types import SimpleNamespace

from sgl_eval.model_preset import list_model_preset_ids

from sglang.cli import interactive
from sglang.cli.interactive import command_argv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import CustomTestCase, terminate_and_kill_process_tree
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=3000, stage="nightly", runner_config="8-gpu-h200")
register_cuda_ci(est_time=3000, stage="nightly", runner_config="8-gpu-b200")

SAMPLES = 2
GSM8K_EXAMPLES = 200
# 200 questions near 95% carry a ~1.5-point standard error; the page may have
# measured with another harness or more questions.
GSM8K_TOLERANCE = 0.05
# Includes downloading the checkpoint.
LAUNCH_TIMEOUT = 3600
# Arbitrary; for models sgl-eval has no preset for, which it then runs greedy.
FALLBACK_MAX_TOKENS = 32768


def _guide_picks() -> list[tuple[str, dict]]:
    """What the Guide launches here per model and GPU count, if its GSM8K is known."""
    device_name, gpus = interactive._detect_gpus()
    hardware = interactive._hardware_ids(device_name)
    picks = []
    files = (resources.files("sglang.cli") / "_commands").iterdir()
    for path in sorted(files, key=lambda path: path.name):
        model = path.name.removesuffix(".json").replace("--", "/")
        recipes = [r for r in json.loads(path.read_text()) if r["hw"] in hardware]
        for count in sorted({r["gpus"] for r in recipes if r["gpus"] <= gpus}):
            with contextlib.redirect_stderr(io.StringIO()):
                recipe = interactive._pick_mode(
                    recipes=[r for r in recipes if r["gpus"] == count],
                    interactive=False,
                )
            if recipe["verified"] == "verified" and recipe["gsm8k_pct"] is not None:
                picks.append((model, recipe))
    return picks


def _generation(model: str) -> dict:
    """sgl-eval's preset for the model when it has one; greedy and capped otherwise."""
    if model in list_model_preset_ids():
        return {"load_preset_from_model_id": model}
    return {"max_tokens": FALLBACK_MAX_TOKENS}


class TestCliGuideCookbookRecipes(CustomTestCase):
    def test_guide_launches_the_cookbook_command_at_its_accuracy(self):
        picks = _guide_picks()
        self.assertTrue(picks, "No Cookbook command with a measured GSM8K fits here.")
        seed = datetime.date.today().isoformat()
        sample = random.Random(seed).sample(picks, min(SAMPLES, len(picks)))
        print(f"Sampled with seed {seed}: {[recipe['url'] for _, recipe in sample]}")
        for model, recipe in sample:
            with self.subTest(recipe["url"]):
                self._check(model, recipe)

    def _check(self, model: str, recipe: dict) -> None:
        argv = command_argv(recipe["command"])
        port = argv[argv.index("--port") + 1] if "--port" in argv else "30000"
        base_url = f"http://127.0.0.1:{port}"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else map(str, range(recipe["gpus"]))
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(list(devices)[: recipe["gpus"]]),
        }
        print(
            f"Reproduce: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
            f"sglang serve {model}",
            flush=True,
        )
        with tempfile.TemporaryFile("w+") as stdout:
            process = subprocess.Popen(
                [sys.executable, "-m", "sglang.cli.main", "serve", model],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                env=env,
            )
            try:
                wait_for_http_ready(
                    url=f"{base_url}/health_generate",
                    timeout=LAUNCH_TIMEOUT,
                    process=process,
                )
                stdout.seek(0)
                self.assertTrue(
                    stdout.read().startswith(recipe["command"] + "\n"),
                    "The Guide did not print the Cookbook command it launched.",
                )
                metrics = run_sgl_eval(
                    SimpleNamespace(
                        eval_name="gsm8k",
                        base_url=base_url,
                        num_examples=GSM8K_EXAMPLES,
                        num_threads=64,
                        **_generation(model),
                    )
                )
            finally:
                terminate_and_kill_process_tree(process)
        floor = recipe["gsm8k_pct"] / 100 - GSM8K_TOLERANCE
        self.assertGreaterEqual(metrics["score"], floor, recipe["url"])


if __name__ == "__main__":
    unittest.main(verbosity=3)
