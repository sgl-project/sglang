"""MI35x agentic-replay throughput for Qwen3.5-397B-A17B MXFP4 with EAGLE MTP.

Port of InferenceX's AgentX recipe
``benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh``
(https://github.com/SemiAnalysisAI/InferenceX) into sglang's AMD CI, dispatched
by ``.github/workflows/nightly-benchmark-amd.yml``. It reports a throughput
number rather than gating anything, which is why it runs there and not in the
AMD nightly test workflow.

The recipe has two modes and so does this file, selected by ``AGENTIC_MODE``:
``replay`` (the default) measures the coding-agent replay, and ``eval`` is the
script's ``EVAL_ONLY`` pass, which scores the same serving configuration on
GSM8K. Each mode is one CI matrix point, so neither waits on the other's server.

Why this exists next to the batch-sweep perf tests: every other MI35x perf test
runs ``bench_one_batch_server`` at a fixed batch size and a fixed prompt length,
which is the workload shape this serving config is least like. The AgentX recipe
turns on hierarchical-cache CPU offload, ``page_size=16``, fp8 KV and EAGLE MTP
precisely because the target workload is long, growing, cache-heavy coding-agent
conversations. A uniform 4k/512 sweep cannot regress on any of that. This
replays real coding-agent trajectories instead, so the numbers move when
prefix-cache behaviour, host-tier offload or speculative decoding regress.

Server configuration is a faithful port: parallelism, aiter attention, fp8 KV,
page size, chunked prefill and prefill caps, scheduler/stream intervals,
tokenizer workers, reasoning and tool-call parsers, EAGLE MTP shape, and the
HiCache CPU tier are all the recipe's values, with the derived
``max_running_requests``/``cuda_graph_max_bs`` arithmetic carried over. The
parallelism and concurrency defaults come from the recipe's search space in
InferenceX ``configs/amd-master.yaml`` rather than from the shell script, which
takes them as required environment variables.

Two things intentionally differ from AgentX:

* The driver is :mod:`sglang.test.agentic_trace_replay` -- a loop over
  ``bench_serving``'s own chat request function and metrics -- not aiperf,
  whose ``inferencex-agentx-mvp`` scenario is not installable from this repo.
  ``sglang.test.agentic_trace_utils`` converts the same published SemiAnalysis
  corpus into sglang's agentic-trace format; see its docstring for what
  survives the conversion (per-turn prompt growth and decode length,
  within-conversation reuse, one unique prefix per trajectory) and what does
  not (recorded think time, sub-agent overlap, prompt content). The absolute
  number is therefore *not* comparable to a published AgentX submission; it is
  comparable to this test's own history, which is what a nightly needs.
* Acceptance length is measured, not pinned. AgentX exports
  ``SGLANG_SIMULATE_ACC_LEN=3.39`` so that a cross-framework comparison is not
  confounded by draft quality. Inside sglang's own nightly the draft model is
  fixed and a real acceptance drop is a regression we want to see, so the run
  uses real verification and reports the achieved accept length. Set
  ``AGENTIC_SIMULATE_ACC_LEN`` to restore the AgentX behaviour.

Reference point. AgentX's own sweep of this recipe at this concurrency
(InferenceX run 33298482346, tp4 + MTP + DRAM HiCache at c48, sglang
v0.5.18-rocm720-mi35x) measured 1,464 output tok/s at a 17.10 ms median ITL,
against 1,380 tok/s and 16.93 ms here. The two harnesses disagree by more on
the prompt side -- AgentX reports a 94.7% prompt-cache read rate against 83.4%
here -- because it starts each trajectory mid-trace behind a 30-minute warmup
grace period and measures an hour of steady state, where this run flushes the
cache and measures the cold prefill burst as well. Decode-side agreement is
what makes the number worth tracking; the prompt-side gap is the harness.

Everything is env-overridable so the AMD CI owners can retune the workload
without touching the recipe.

Registry: nightly-perf-mi35x-qwen35-mxfp4-agentic-mtp suite
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.agentic_trace_replay import run_agentic_replay
from sglang.test.agentic_trace_utils import ensure_agentic_trace_file, weka_trace_url
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-perf-mi35x-qwen35-mxfp4-agentic-mtp",
    nightly=True,
)

# The published MXFP4 checkpoint, resolved through the runners' shared HF cache.
# test_qwen35_mxfp4_eval_mi35x.py points at a /data copy of the same weights,
# but that path is a dev-host mount and is not visible inside the CI container.
MODEL_PATH = os.environ.get("QWEN35_MXFP4_MODEL_PATH", "amd/Qwen3.5-397B-A17B-MXFP4")

# AgentX runs this recipe twice: a throughput replay, and an EVAL_ONLY pass
# that scores the served model. Both modes share this file and its server
# configuration, and each skips the other's case rather than launching a second
# 400B server it would not use.
MODE = os.environ.get("AGENTIC_MODE", "replay")

# Parallelism and concurrency come from the recipe's own search space in
# InferenceX configs/amd-master.yaml (qwen3.5-fp4-mi355x-sglang-agentic-mtp):
# tp2 and tp4 at ep1, with the HiCache CPU tier only on the high-concurrency
# rows. The CI matrix runs the ten tp2 points on the 2-GPU MI35x runner; tp4
# needs the 8-GPU runner, so it stays a manual override rather than a scheduled
# point. The recipe never runs this checkpoint at tp8.
TP_SIZE = int(os.environ.get("AGENTIC_TP", "4"))
EP_SIZE = int(os.environ.get("AGENTIC_EP_SIZE", "1"))
CONCURRENCY = int(os.environ.get("AGENTIC_CONCURRENCY", "48"))
# AgentX runs every concurrency for the same wall-clock duration, so its low
# concurrencies simply serve fewer turns. This replay is closed-loop instead, so
# one fixed corpus would take concurrency 1 about ten hours and concurrency 32
# about twenty minutes. Sizing the corpus to the concurrency gives every point
# the same per-channel work, which keeps the sweep comparable and each point
# inside the job timeout; the floor keeps the lowest concurrencies from
# measuring a handful of turns. The default lands on 96 at c48, which is what
# the tp4 reference point below measured.
NUM_CONVERSATIONS = int(
    os.environ.get("AGENTIC_NUM_CONVERSATIONS") or max(8, 2 * CONCURRENCY)
)
# AgentX runs to a wall-clock duration; sglang's replay is closed-loop, so the
# run is bounded by conversations x turns instead. 16 turns already carries each
# trajectory past 80k tokens of context, which is where this config is aimed.
MAX_TURNS = int(os.environ.get("AGENTIC_MAX_TURNS", "16"))
# Unset replays the corpus' own per-turn reply lengths, which is what keeps the
# prompt growth faithful; set it to force a uniform decode length instead.
OUTPUT_LEN = int(os.environ.get("AGENTIC_OUTPUT_LEN", "0")) or None
KV_OFFLOADING = os.environ.get("AGENTIC_KV_OFFLOADING", "hicache")
SCHEDULER_RECV_INTERVAL = os.environ.get("SCHEDULER_RECV_INTERVAL", "30")
SIMULATE_ACC_LEN = os.environ.get("AGENTIC_SIMULATE_ACC_LEN", "")

# AgentX scores its eval pass with lm-eval's gsm8k over the full 1319-question
# set. sglang's own gsm8k runner asks the same questions few-shot through the
# completion API, which is what every other AMD accuracy test uses, so the score
# is comparable to those rather than to an lm-eval number. The gate sits below
# the 0.91 the plain MXFP4 accuracy test holds, because this recipe also serves
# fp8 KV and EAGLE MTP.
EVAL_NUM_EXAMPLES = int(os.environ.get("AGENTIC_EVAL_NUM_EXAMPLES", "1319"))
EVAL_ACC_THRESHOLD = float(os.environ.get("AGENTIC_EVAL_ACC_THRESHOLD", "0.88"))

HICACHE_RATIO = os.environ.get("HICACHE_RATIO", "1.5")
HICACHE_WRITE_POLICY = os.environ.get("HICACHE_WRITE_POLICY", "write_through")
HICACHE_IO_BACKEND = os.environ.get("HICACHE_IO_BACKEND", "direct")
HICACHE_MEM_LAYOUT = os.environ.get("HICACHE_MEM_LAYOUT", "page_first_direct")

# This checkpoint has 32 q / 2 kv heads, so tp4 gives a 8:1 GQA ratio, which is
# the one shape where aiter_backend routes target-verify into the hand-written
# gfx950 assembly kernel (vattn_asm_gfx950; the 16:1 branch does not need it).
# That kernel is assembled and hipModuleLoad-ed on first use, and its first
# launch lands inside decode CUDA graph capture, where it returns HIP 709
# (context is destroyed) and takes the server down before it ever serves a
# request -- reproduced on both launch attempts of
# https://github.com/sgl-project/sglang/actions/runs/34448150604. Turning the
# kernel off drops the 8:1 branch of that gate, so verify falls back to the same
# generic unified_attention path a tp8 run takes. Set this to 1 to measure the
# assembly kernel again once it survives capture.
ASM_VERIFY_ATTN = os.environ.get("AGENTIC_ASM_VERIFY_ATTN", "0")

SERVER_LAUNCH_TIMEOUT = int(os.environ.get("AGENTIC_SERVER_TIMEOUT", "5400"))

RESULT_DIR = "performance_results_qwen35_mxfp4_agentic_mi35x"

# Recipe env. PYTHONNOUSERSITE keeps a user site-packages directory from
# shadowing the container's sglang, as in the AgentX launcher.
COMMON_ENV = {
    "PYTHONNOUSERSITE": "1",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_FLYDSL_FORCE": "1",
    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT8",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
    "SGLANG_ASM_VERIFY_ATTN": ASM_VERIFY_ATTN,
}


def _trace_cache_dir() -> str:
    """Prefer /sgl-data, the runners' persistent mount, so this converts once."""
    override = os.environ.get("AGENTIC_TRACE_CACHE_DIR")
    if override:
        return override
    if os.path.isdir("/sgl-data") and os.access("/sgl-data", os.W_OK):
        return "/sgl-data/agentic-traces"
    return os.path.join(tempfile.gettempdir(), "sglang-agentic-traces")


def build_server_args() -> list:
    """Server flags from qwen3.5_fp4_mi355x_sglang_mtp.sh."""
    max_running_requests = 2 * CONCURRENCY
    cuda_graph_max_bs = min(max_running_requests, 128)

    args = [
        "--trust-remote-code",
        "--tp",
        str(TP_SIZE),
        "--ep-size",
        str(EP_SIZE),
        "--attention-backend",
        "aiter",
        "--mem-fraction-static",
        "0.80",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
        "--watchdog-timeout",
        "1200",
        "--page-size",
        "16",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--cuda-graph-max-bs-decode",
        str(cuda_graph_max_bs),
        "--max-running-requests",
        str(max_running_requests),
        "--max-prefill-tokens",
        "16384",
        "--chunked-prefill-size",
        "16384",
        "--scheduler-recv-interval",
        SCHEDULER_RECV_INTERVAL,
        "--stream-interval",
        "50",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-metrics",
        "--enable-cache-report",
    ]

    if TP_SIZE >= 4:
        args += ["--tokenizer-worker-num", "6"]

    if KV_OFFLOADING == "hicache":
        args += [
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            HICACHE_RATIO,
            "--hicache-write-policy",
            HICACHE_WRITE_POLICY,
            "--hicache-io-backend",
            HICACHE_IO_BACKEND,
            "--hicache-mem-layout",
            HICACHE_MEM_LAYOUT,
        ]

    return args


def build_server_env() -> dict:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    # AgentX pins acceptance on the throughput side only; its eval pass keeps
    # real target-model verification, so a pinned length is dropped here.
    if SIMULATE_ACC_LEN and MODE != "eval":
        env["SGLANG_SIMULATE_ACC_LEN"] = SIMULATE_ACC_LEN
        env["SGLANG_SIMULATE_ACC_METHOD"] = "match-expected"
        env["SGLANG_SIMULATE_ACC_TOKEN_MODE"] = "real-draft-token"
    return env


def render_report(result: dict, trace_summary: str) -> str:
    cache = result.get("cache_report") or {}
    accept_length = result.get("accept_length")
    accept_cell = f"{accept_length:.2f}" if accept_length else "n/a (not reported)"
    if SIMULATE_ACC_LEN:
        accept_cell += f" (simulated {SIMULATE_ACC_LEN})"
    asm_verify_cell = (
        ("on" if ASM_VERIFY_ATTN == "1" else "off")
        if TP_SIZE == 4
        else "n/a (tp2 GQA16 path)"
    )

    report = (
        f"### Qwen3.5-397B-A17B MXFP4 + EAGLE MTP, AgentX agentic replay "
        f"[{os.getenv('GPU_CONFIG', 'MI35x')}]\n\n"
        f"| workload | value |\n| --- | --- |\n"
        f"| parallelism | tp{TP_SIZE} / ep{EP_SIZE} |\n"
        f"| concurrency | {CONCURRENCY} |\n"
        f"| max turns / conversation | {MAX_TURNS} |\n"
        f"| kv offloading | {KV_OFFLOADING} |\n"
        f"| gfx950 asm verify attention | {asm_verify_cell} |\n"
        f"{trace_summary}\n"
        f"| metric | value |\n| --- | --- |\n"
        f"| completed turns | {result.get('completed')} |\n"
        f"| duration (s) | {result.get('duration', 0):.1f} |\n"
        f"| request throughput (turn/s) | {result.get('request_throughput', 0):.2f} |\n"
        f"| input throughput (tok/s) | {result.get('input_throughput', 0):.1f} |\n"
        f"| output throughput (tok/s) | {result.get('output_throughput', 0):.1f} |\n"
        f"| total throughput (tok/s) | {result.get('total_throughput', 0):.1f} |\n"
        f"| mean TTFT (ms) | {result.get('mean_ttft_ms', 0):.1f} |\n"
        f"| p99 TTFT (ms) | {result.get('p99_ttft_ms', 0):.1f} |\n"
        f"| median ITL (ms) | {result.get('median_itl_ms', 0):.2f} |\n"
        f"| p99 ITL (ms) | {result.get('p99_itl_ms', 0):.2f} |\n"
        f"| mean E2E latency (ms) | {result.get('mean_e2e_latency_ms', 0):.1f} |\n"
        f"| accept length | {accept_cell} |\n"
    )
    if cache:
        report += (
            f"| prefix cache hit rate | {cache.get('cache_hit_rate_pct', 0):.1f}% |\n"
            f"| cached tokens (device) | {cache.get('device_cached_tokens')} |\n"
            f"| cached tokens (host) | {cache.get('host_cached_tokens')} |\n"
        )
    return report


@unittest.skipUnless(MODE == "replay", f"AGENTIC_MODE={MODE} skips the replay")
class TestQwen35Mxfp4AgenticMtpMI35x(CustomTestCase):
    """AgentX coding-agent replay against the MI35x MXFP4 + MTP serving recipe."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.makedirs(RESULT_DIR, exist_ok=True)

    def _prepare_trace(self, tokenizer):
        """Convert the published AgentX corpus, or skip if it is unreachable."""
        source = os.environ.get("AGENTIC_TRACE_SOURCE") or weka_trace_url()
        try:
            return ensure_agentic_trace_file(
                tokenizer,
                output_dir=_trace_cache_dir(),
                num_conversations=NUM_CONVERSATIONS,
                output_len=OUTPUT_LEN,
                max_turns=MAX_TURNS,
                source=source,
            )
        except Exception as exc:
            # A corpus that cannot be fetched or converted is an infrastructure
            # problem, not an engine regression; do not report it as one.
            raise unittest.SkipTest(
                f"AgentX trace corpus unavailable from {source}: {exc!r}"
            )

    def test_agentic_replay(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        trace_path, stats = self._prepare_trace(tokenizer)
        print(f"Agentic trace: {trace_path} ({stats})", flush=True)

        result_file = os.path.join(RESULT_DIR, "agentic_replay.jsonl")
        if os.path.exists(result_file):
            os.remove(result_file)

        process = popen_launch_server(
            MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=build_server_args(),
            env=build_server_env(),
        )
        try:
            result = run_agentic_replay(
                base_url=self.base_url,
                model=MODEL_PATH,
                tokenizer=tokenizer,
                trace_path=trace_path,
                num_conversations=NUM_CONVERSATIONS,
                max_turns=MAX_TURNS,
                output_len=OUTPUT_LEN,
                max_concurrency=CONCURRENCY,
                output_file=result_file,
            )
        finally:
            kill_process_tree(process.pid)

        report = render_report(result, stats.as_markdown_rows())
        print(report, flush=True)
        if is_in_ci():
            write_github_step_summary(report)

        # No throughput gate: the numbers are tracked, and the run itself is the
        # regression guard -- this config only completes if MTP, the HiCache CPU
        # tier and the aiter MXFP4 path all survive a long-context agentic
        # workload. A dropped turn takes the rest of its trajectory with it, so
        # a partial replay is a failure and not a smaller benchmark.
        self.assertEqual(
            result["failed"], 0, f"{result['failed']} turns failed: {result['errors']}"
        )
        self.assertGreater(result["output_throughput"], 0, "No output tokens generated")


@unittest.skipUnless(MODE == "eval", f"AGENTIC_MODE={MODE} skips the eval pass")
class TestQwen35Mxfp4AgenticEvalMI35x(CustomTestCase):
    """AgentX's EVAL_ONLY pass: score the same serving recipe on GSM8K."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_gsm8k(self):
        process = popen_launch_server(
            MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=build_server_args(),
            env=build_server_env(),
        )
        try:
            requests.get(self.base_url + "/flush_cache")
            metrics = run_eval(
                SimpleNamespace(
                    base_url=self.base_url,
                    model=MODEL_PATH,
                    eval_name="gsm8k",
                    api="completion",
                    max_tokens=2048,
                    num_examples=EVAL_NUM_EXAMPLES,
                    num_threads=CONCURRENCY,
                )
            )
        finally:
            kill_process_tree(process.pid)

        report = (
            f"### Qwen3.5-397B-A17B MXFP4 + EAGLE MTP, AgentX eval "
            f"[{os.getenv('GPU_CONFIG', 'MI35x')}]\n\n"
            f"| workload | value |\n| --- | --- |\n"
            f"| parallelism | tp{TP_SIZE} / ep{EP_SIZE} |\n"
            f"| concurrency | {CONCURRENCY} |\n"
            f"| kv offloading | {KV_OFFLOADING} |\n"
            f"| metric | value |\n| --- | --- |\n"
            f"| gsm8k questions | {EVAL_NUM_EXAMPLES} |\n"
            f"| gsm8k score | {metrics['score']:.3f} "
            f"(threshold {EVAL_ACC_THRESHOLD}) |\n"
        )
        print(report, flush=True)
        if is_in_ci():
            write_github_step_summary(report)

        self.assertGreater(metrics["score"], EVAL_ACC_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
