"""Breakable CUDA graph must register legacy custom all-reduce inputs under TP.

With ``--tp-size > 1`` every ``RowParallelLinear`` issues a per-block all-reduce
that lands *inside* a captured BCG segment (BCG's break points are the attention
modules, so everything else is captured). Custom all-reduce hands the captured
kernel a rank-data slot whose peer pointers are only filled in by the
``register_graph_buffers()`` that runs when ``CustomAllreduce.capture()`` exits,
and that registration is a host-side IPC exchange -- it cannot happen inside the
captured region. If capture is not wrapped in the TP group's ``graph_capture()``,
the slots stay unwritten and replay dereferences them:
``cudaErrorIllegalAddress``, no image.

The failure is invisible to the single-GPU BCG suite (``--num-gpus 1`` never
takes the custom all-reduce path at all), which is why this file exists. Capture
itself succeeds either way -- the log still prints "captured N segment(s)" -- so
asserting on the capture marker alone is not enough; we also require the
registration to have happened.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_diffusion_bcg_tp2_zimage_turbo.py
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    run_logged_command,
)
from sglang.test.test_utils import CustomTestCase

IMAGE_SIZE = "512x512"
NUM_INFERENCE_STEPS = 4
SEED = 0
TP_SIZE = 2

BCG_CAPTURE_MARKER = "[Diffusion BCG] captured"
# Printed by CustomAllreduce.register_graph_buffers(), i.e. only on exit of the
# capture() context the fix enters. Its absence is the precise signature of the
# regression: capture ran, registration did not.
GRAPH_BUFFER_REGISTRATION_MARKER = "cuda graph addresses"
# CustomAllReduceV2 registers graph inputs only for its zero-copy pull algorithms,
# which this workload does not select, so the marker only applies to v1. V2 is
# detected by its request in the symmetric workspace log.
CUSTOM_ALL_REDUCE_V2_MARKER = "custom_all_reduce_v2: slot = "
ILLEGAL_MEMORY_MARKER = "illegal memory access"

# The regression fails during warmup within ~1 min; the timeout only has to be
# generous enough for weight load plus one capture per bucket.
GENERATE_TIMEOUT_SECONDS = 900


class TestDiffusionBCGTP2ZImageTurbo(CustomTestCase):
    def test_zimage_turbo_bcg_generates_under_tp2(self):
        if torch.cuda.device_count() < TP_SIZE:
            self.skipTest(f"needs {TP_SIZE} GPUs, found {torch.cuda.device_count()}")

        artifact_dir = Path(
            os.environ.get(
                "SGLANG_DIFFUSION_ARTIFACT_DIR",
                tempfile.mkdtemp(prefix="sglang_diffusion_bcg_tp2_"),
            )
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_dir / "zimage_turbo_bcg_tp2.log"

        cmd = [
            "sglang",
            "generate",
            "--backend",
            "sglang",
            "--model-path",
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "--prompt",
            "A red fox in fresh snow",
            "--width",
            IMAGE_SIZE.split("x")[0],
            "--height",
            IMAGE_SIZE.split("x")[1],
            "--seed",
            str(SEED),
            "--num-inference-steps",
            str(NUM_INFERENCE_STEPS),
            "--num-gpus",
            str(TP_SIZE),
            "--tp-size",
            str(TP_SIZE),
            "--warmup-resolutions",
            IMAGE_SIZE,
            "--no-save-output",
            "--guidance-scale",
            "0.0",
            "--enable-breakable-cuda-graph",
            # One bucket keeps the run short; the TP all-reduce path does not
            # depend on how many prompt buckets get captured.
            "--bcg-text-buckets",
            "128",
            "--enable-torch-compile",
            "false",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
        ]

        env = os.environ.copy()
        # this guard exercises legacy graph-buffer registration; v2 uses a
        # symmetric workspace and does not emit the registration marker
        env["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] = "0"

        output = run_logged_command(
            cmd, log_path, timeout=GENERATE_TIMEOUT_SECONDS, env=env
        )
        self.assertNotIn(
            ILLEGAL_MEMORY_MARKER,
            output,
            f"replay hit an unregistered peer address. Log: {log_path}",
        )
        self.assertNotIn("[Diffusion BCG] capture failed", output)
        self.assertIn(
            BCG_CAPTURE_MARKER,
            output,
            f"BCG never captured, so this run did not exercise TP+BCG. Log: {log_path}",
        )
        if CUSTOM_ALL_REDUCE_V2_MARKER not in output:
            self.assertIn(
                GRAPH_BUFFER_REGISTRATION_MARKER,
                output,
                f"custom all-reduce graph buffers were never registered, so capture "
                f"did not go through the TP group's graph_capture(). "
                f"Log: {log_path}",
            )
        self.assertIn("Pixel data generated successfully", output)


if __name__ == "__main__":
    unittest.main(verbosity=3)
