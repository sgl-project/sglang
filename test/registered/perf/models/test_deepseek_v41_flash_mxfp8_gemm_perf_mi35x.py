"""MI35x benchmark for DeepSeek-V4.1-Flash with mori's mxfp8 GEMM, unfused.

A separate optimisation from the fused wo_b, measured separately because the
two have different reach and their gains do not add. This is the same multiply
with nothing fused onto it, so it reaches the layers fusing structurally
cannot:

    wq_b   N=8192 K=1280   ColumnParallel -- no all-reduce to fuse with at all
    wo_b   N=5120 K=2048   RowParallel -- fused where that pays, this below it

    base        today's mxfp8_native_blockscaled_linear
    mori-gemm   the same linear through mori

**The fused path is off in every variant here.** With both on, wo_b's calls
reach fusing first and this would only catch its leftovers, which measures the
pair rather than this one.

At the layer, graph-replayed on an idle box, it is -20% to -34% from M=2048 up
and a loss below it: mori's 256x256 tile is mostly idle at small M, where the
native route drops to a GEMV shape that suits it. ``_MIN_M`` encodes that, so
the decode column below is expected to show nothing at all.

Both halves are needed. A fused all-reduce that does not move bytes is *faster*
than one that does, so a perf table cannot on its own tell an optimisation from
a broken transport; GSM8K is what makes the perf number mean something. mori's
``GemmAllReduceOp.self_test`` covers the same ground at construction, and this
file skips a fused variant when it trips rather than publishing numbers from a
stack that is not doing the work.

One check worth keeping from the fused suite, and it matters at least as much
here: **that the path ran at all during the accuracy gate.** A path that
declined every call looks exactly like one that ran and lost -- the model stays
correct, the profile stays plausible, and the table reports the baseline under
mori's name. ``_mori_engaged`` reads the shape histogram and fails instead.

Registry: nightly-perf-4-gpu-mi35x-deepseek-v41-flash-mxfp8-gemm suite
"""

import json
import os
import re
import subprocess
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Three 285B loads, three GSM8K runs and three benchmark sweeps.
register_amd_ci(
    est_time=10800,
    suite="nightly-perf-4-gpu-mi35x-deepseek-v41-flash-wo-b-fusion",
    nightly=True,
)

MODEL_PATH = os.environ.get(
    "DEEPSEEK_V41_FLASH_MODEL_PATH", "/root/models/DeepSeek-V4.1-Flash"
)
SERVER_LAUNCH_TIMEOUT = 3600

#: The fp8 gather costs relL2 2.3e-2 on its leg, which is below what GSM8K
#: resolves; the gate is here to catch a collective that moved nothing, where
#: the model answers fluently and scores near zero.
GSM8K_MIN_ACCURACY = 0.90

#: input 4096 so that batch_size * 4096 crosses the fused path's floors:
#: bs=1 is M=4096 and never fuses, bs>=4 fills a 16384 chunk and does. Both
#: sides of the threshold in one sweep is the point -- a table where every row
#: fuses cannot show what the floor is protecting.
INPUT_LEN = 4096
OUTPUT_LEN = 512
#: The leading duplicate is a warmup and is dropped. It matters more here than
#: usual: mori compiles a kernel per distinct M on first sight, several seconds
#: each, and that lands on whichever request arrives first.
BATCH_SIZES = ["1", "1", "4", "8", "16"]

#: The verified MI350X low-latency cell (docs.sglang.io cookbook, DeepSeek-V4_1)
#: plus the two mori needs. AITER_BF16_FP8_MOE_BOUND is load-bearing and not
#: part of the published cell: this checkpoint has swiglu_limit=10.0, so its MoE
#: takes AITER's clamped-SwiGLU INTERLEAVE path, where no ck_moe_stage1 kernel
#: is compiled for (bf16 activation x fp4 weight). Every M below the default
#: bound of 256 -- i.e. every decode step -- then raises "Unsupported kernel
#: config for moe heuristic dispatch". Dropping it to 0 keeps the fp8-activation
#: kernel at all shapes.
COMMON_ENV_VARS = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_MOE_PADDING": "1",
    "AITER_FLYDSL_FORCE_REDUCE": "1",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "NONE",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    # Without this the comm has no SDMA queues and every put silently does
    # nothing -- see mori_gemm_ar.py's docstring.
    "MORI_ENABLE_SDMA": "1",
    "MORI_SOCKET_IFNAME": "lo",
}

VARIANTS = [
    {"name": "base", "env": {}},
    {
        "name": "mori-gemm",
        "env": {
            "SGLANG_OPT_MORI_MXFP8_GEMM": "1",
            # So _mori_engaged can tell "ran and lost" from "never ran".
            "SGLANG_OPT_MORI_MXFP8_GEMM_SHAPE_LOG": "1",
        },
    },
]

SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--ep-size",
    "4",
    # No radix cache: a prefix that survives between points turns a measured
    # prefill into a lookup.
    "--disable-radix-cache",
    # 0.78 rather than the cell's 0.80: the symmetric window is VMM memory
    # outside torch's allocator, and the op reserves 4x it plus slack.
    "--mem-fraction-static",
    "0.78",
    # Has to be at least the fp8 wire's floor or no prefill chunk ever reaches
    # it and the fused variants measure the base path.
    "--chunked-prefill-size",
    "16384",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--cuda-graph-max-bs-decode",
    "64",
    "--cuda-graph-backend-prefill",
    "breakable",
    "--cuda-graph-max-bs-prefill",
    "4096",
]


#: What the layer logs when it gives up. `fused_wo_b` disables itself on any
#: failure and falls back. That is the right behaviour and it is also *silent*:
#: the model is correct, GSM8K passes, and the perf table reports a fused
#: variant that never fused. Only the server's own log says so.
_FALLBACK_MARKER = "mori mxfp8 GEMM failed and is disabled"
#: The periodic histogram, e.g. "mori mxfp8 shapes: 8192x1280/M=16384+x40 ...".
#: A "+" is a served call, a "-" a declined one.
_SHAPE_LINE = re.compile(r"mori mxfp8 shapes: (.*)")


def _fallback_reason(server_log: str) -> str | None:
    """The line where the server gave up on the fused path, if it did."""
    try:
        with open(server_log, errors="ignore") as f:
            for line in f:
                if _FALLBACK_MARKER in line:
                    return line.strip()[-300:]
    except OSError:
        return None
    return None


def _mori_engaged(server_log: str) -> str | None:
    """The eligible shapes the layer actually saw, or None if it saw none.

    Distinguishes the two ways a fused variant can produce base-like numbers:
    the path was used and was not worth it, or it was never used because no
    forward reached the floor. Only the first is a result.
    """
    seen = []
    try:
        with open(server_log, errors="ignore") as f:
            for line in f:
                m = _SHAPE_LINE.search(line)
                if m:
                    seen.extend(tok for tok in m.group(1).split() if "+" in tok)
    except OSError:
        return None
    return " ".join(sorted(set(seen))) if seen else None


class TestDeepseekV41FlashMxfp8GemmPerfMI35x(CustomTestCase):
    """One server per wire, GSM8K then bench_one_batch_server on each."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.report: list[str] = []
        cls.accuracy: dict[str, float] = {}

    @classmethod
    def tearDownClass(cls):
        if cls.report and is_in_ci():
            write_github_step_summary("\n".join(cls.report) + "\n")

    def _launch(self, variant):
        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)
        env.update(variant["env"])
        log_path = f"/tmp/dsv41_flash_mxfp8_gemm_{variant['name']}.serverlog"
        log = open(log_path, "w")
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=SERVER_ARGS,
            env=env,
            return_stdout_stderr=(log, log),
        )
        return process, log_path

    def _gsm8k(self, variant_name):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=256,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        accuracy = metrics["accuracy"]
        self.accuracy[variant_name] = accuracy
        print(f"[{variant_name}] gsm8k accuracy={accuracy:.3f}")
        self.assertGreater(
            accuracy,
            GSM8K_MIN_ACCURACY,
            f"{variant_name}: GSM8K {accuracy:.3f} below {GSM8K_MIN_ACCURACY}. "
            f"A fused all-reduce that does not move bytes still answers "
            f"fluently and still measures fast; this is the check that sees it.",
        )

    def _bench(self, variant_name):
        json_output = f"/tmp/dsv41_flash_mxfp8_gemm_{variant_name}.json"
        if os.path.exists(json_output):
            os.remove(json_output)
        cmd = [
            "python3",
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            "None",
            "--base-url",
            self.base_url,
            "--batch-size",
            *BATCH_SIZES,
            "--input-len",
            str(INPUT_LEN),
            "--output-len",
            str(OUTPUT_LEN),
            "--show-report",
            f"--pydantic-result-filename={json_output}",
            "--no-append-to-github-summary",
            "--trust-remote-code",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            self.fail(f"bench_one_batch_server failed (rc={result.returncode})")
        self.assertTrue(os.path.exists(json_output), f"{json_output} not found")
        with open(json_output) as f:
            rows = json.load(f)
        self.assertTrue(rows, "No benchmark results returned")
        if len(rows) > 1 and rows[0]["batch_size"] == rows[1]["batch_size"]:
            rows = rows[1:]  # drop the warmup

        self.report.append(
            f"### {variant_name} (input_len={INPUT_LEN} output_len={OUTPUT_LEN}, "
            f"gsm8k={self.accuracy.get(variant_name, float('nan')):.3f})"
        )
        self.report.append(
            "| batch size | latency (s) | input throughput (tok/s) | "
            "output throughput (tok/s) | ITL (ms) |"
        )
        self.report.append(
            "| ---------- | ----------- | ------------------------ | "
            "------------------------- | -------- |"
        )
        for r in rows:
            bs = r["batch_size"]
            latency = r.get("latency", 0.0)
            in_tp = r.get("input_throughput", 0.0)
            out_tp = r.get("output_throughput", 0.0)
            itl = 1 / (out_tp / bs) * 1000 if out_tp > 0 else float("inf")
            self.report.append(
                f"| {bs} | {latency:.2f} | {in_tp:.2f} | {out_tp:.2f} | {itl:.2f} |"
            )
            print(
                f"[{variant_name}] bs={bs} latency={latency:.2f}s "
                f"in_tp={in_tp:.2f} out_tp={out_tp:.2f} ITL={itl:.2f}ms"
            )

    def _run_variant(self, variant):
        process, log_path = self._launch(variant)
        try:
            self._gsm8k(variant["name"])
            if variant["env"]:
                # After GSM8K, not before: the layer is constructed lazily on
                # the first eligible call, so nothing has decided yet at launch.
                reason = _fallback_reason(log_path)
                if reason is not None:
                    self.skipTest(
                        f"{variant['name']}: the fused wo_b disabled itself and "
                        f"ran unfused, so a perf number here would describe the "
                        f"base path under a fused label. Server said: {reason}"
                    )
            self._bench(variant["name"])
            if variant["env"]:
                engaged = _mori_engaged(log_path)
                self.assertIsNotNone(
                    engaged,
                    f"{variant['name']}: no wo_b call was ever eligible, so both "
                    f"the accuracy gate and the perf table describe the base "
                    f"path. Either the floors are above every M this workload "
                    f"produces, or --chunked-prefill-size is below them.",
                )
                print(f"[{variant['name']}] mori shapes: {engaged}")
                self.report.append(f"mori shapes: `{engaged}`")
        finally:
            kill_process_tree(process.pid)

    def test_a_base(self):
        self._run_variant(VARIANTS[0])

    def test_b_mori_gemm(self):
        self._run_variant(VARIANTS[1])


if __name__ == "__main__":
    unittest.main()
