"""Nightly e2e per-op GPU time as benchmark marker records."""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=240, stage="nightly", runner_config="1-gpu-large")

WORKLOADS = "decode:1,decode:8,decode:64,prefill:4096"


class TestE2EOpTraceNightly(CustomTestCase):
    def test_op_records(self):
        sink = envs.SGLANG_JIT_BENCHMARK_OUTPUT.get()
        out_dir = tempfile.mkdtemp(prefix="op_trace_")
        cmd = [
            sys.executable,
            "-m",
            "sglang.srt.observability.op_trace",
            "bench",
            "--model",
            DEFAULT_MODEL_NAME_FOR_TEST,
            "--workloads",
            WORKLOADS,
            "--out",
            out_dir,
        ]
        # The bench exits non-zero when the kernel -> op join is not trustworthy.
        subprocess.run(cmd, check=True, timeout=720)

        records = []
        for path in glob.glob(os.path.join(out_dir, "bench-*.jsonl")):
            with open(path) as f:
                records += [json.loads(line) for line in f if line.strip()]
        steps = [r for r in records if r["line"] == "<step>"]
        self.assertEqual(len(steps), len(WORKLOADS.split(",")))
        for path in glob.glob(os.path.join(out_dir, "*.md")):
            with open(path) as f:
                print(f.read())

        if sink is not None:
            os.makedirs(sink, exist_ok=True)
            for path in glob.glob(os.path.join(out_dir, "bench-*.jsonl")):
                shutil.move(path, os.path.join(sink, os.path.basename(path)))


if __name__ == "__main__":
    unittest.main()
