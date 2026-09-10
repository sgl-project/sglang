"""MI35x agentic-replay throughput for Qwen3.5-397B-A17B MXFP4 with EAGLE MTP.

Port of InferenceX's AgentX recipe
``benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh``
(https://github.com/SemiAnalysisAI/InferenceX) into sglang's own nightly AMD CI.

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

* The driver is ``sglang.bench_serving --dataset-name agentic-trace``, not
  aiperf. aiperf's ``inferencex-agentx-mvp`` scenario is not installable from
  this repo, and sglang already replays multi-turn agentic conversations
  natively. ``sglang.test.agentic_trace_utils`` converts the same published
  SemiAnalysis corpus into that format; see its docstring for what survives the
  conversion (per-turn prompt growth and decode length, within-conversation
  reuse, one unique prefix per trajectory) and what does not (recorded think
  time, sub-agent overlap, prompt content). The absolute number is therefore
  *not* comparable to a published AgentX submission; it is comparable to this
  test's own history, which is what a nightly needs.
* Acceptance length is measured, not pinned. AgentX exports
  ``SGLANG_SIMULATE_ACC_LEN=3.39`` so that a cross-framework comparison is not
  confounded by draft quality. Inside sglang's own nightly the draft model is
  fixed and a real acceptance drop is a regression we want to see, so the run
  uses real verification and reports the achieved accept length. Set
  ``AGENTIC_SIMULATE_ACC_LEN`` to restore the AgentX behaviour.

Everything is env-overridable so the AMD CI owners can retune the workload
without touching the recipe.

Registry: nightly-perf-4-gpu-mi35x-qwen35-mxfp4-agentic-mtp suite
"""

import json
import os
import subprocess
import tempfile
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.agentic_trace_utils import ensure_agentic_trace_file, weka_trace_url
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-perf-4-gpu-mi35x-qwen35-mxfp4-agentic-mtp",
    nightly=True,
)

# The published MXFP4 checkpoint, resolved through the runners' shared HF cache.
# test_qwen35_mxfp4_eval_mi35x.py points at a /data copy of the same weights,
# but that path is a dev-host mount and is not visible inside the CI container.
MODEL_PATH = os.environ.get("QWEN35_MXFP4_MODEL_PATH", "amd/Qwen3.5-397B-A17B-MXFP4")

# Parallelism and concurrency come from the recipe's own search space in
# InferenceX configs/amd-master.yaml (qwen3.5-fp4-mi355x-sglang-agentic-mtp):
# tp2 and tp4 at ep1, with the HiCache CPU tier only on the high-concurrency
# rows -- tp4 runs it at concurrency 40/48/56/64. 48 is the middle of that row,
# so one CI point exercises the offload path the rest of this config exists to
# serve. tp4 leaves half of the 8-GPU runner idle, as the MiniMax-M3 tp4 job
# already does; the recipe never runs this checkpoint at tp8.
TP_SIZE = int(os.environ.get("AGENTIC_TP", "4"))
EP_SIZE = int(os.environ.get("AGENTIC_EP_SIZE", "1"))
CONCURRENCY = int(os.environ.get("AGENTIC_CONCURRENCY", "48"))
NUM_CONVERSATIONS = int(os.environ.get("AGENTIC_NUM_CONVERSATIONS", "96"))
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
BENCH_TIMEOUT = int(os.environ.get("AGENTIC_BENCH_TIMEOUT", "7200"))

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
    if SIMULATE_ACC_LEN:
        env["SGLANG_SIMULATE_ACC_LEN"] = SIMULATE_ACC_LEN
        env["SGLANG_SIMULATE_ACC_METHOD"] = "match-expected"
        env["SGLANG_SIMULATE_ACC_TOKEN_MODE"] = "real-draft-token"
    return env


def build_bench_command(base_url: str, trace_path: str, result_file: str) -> list:
    command = [
        "python3",
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        base_url,
        "--model",
        MODEL_PATH,
        "--tokenizer",
        MODEL_PATH,
        "--dataset-name",
        "agentic-trace",
        "--dataset-path",
        trace_path,
        "--num-prompts",
        str(NUM_CONVERSATIONS),
        "--max-concurrency",
        str(CONCURRENCY),
        # Warmup replays one trajectory; flushing afterwards keeps its prefix
        # out of the measured run.
        "--warmup-requests",
        "1",
        "--flush-cache",
        "--cache-report",
        "--disable-tqdm",
        "--output-file",
        result_file,
    ]
    # Without this the trace's own per-turn reply lengths are replayed, which is
    # what keeps the prompt growth faithful.
    if OUTPUT_LEN:
        command += ["--sharegpt-output-len", str(OUTPUT_LEN)]
    return command


def render_report(result: dict, trace_summary: str) -> str:
    cache = result.get("cache_report") or {}
    accept_length = result.get("accept_length")
    accept_cell = f"{accept_length:.2f}" if accept_length else "n/a (not reported)"
    if SIMULATE_ACC_LEN:
        accept_cell += f" (simulated {SIMULATE_ACC_LEN})"

    report = (
        f"### Qwen3.5-397B-A17B MXFP4 + EAGLE MTP, AgentX agentic replay "
        f"[{os.getenv('GPU_CONFIG', 'MI35x')}]\n\n"
        f"| workload | value |\n| --- | --- |\n"
        f"| parallelism | tp{TP_SIZE} / ep{EP_SIZE} |\n"
        f"| concurrency | {CONCURRENCY} |\n"
        f"| max turns / conversation | {MAX_TURNS} |\n"
        f"| kv offloading | {KV_OFFLOADING} |\n"
        f"| gfx950 asm verify attention | "
        f"{'on' if ASM_VERIFY_ATTN == '1' else 'off'} |\n"
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


class TestQwen35Mxfp4AgenticMtpMI35x(CustomTestCase):
    """AgentX coding-agent replay against the MI35x MXFP4 + MTP serving recipe."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.makedirs(RESULT_DIR, exist_ok=True)

    def _prepare_trace(self):
        """Convert the published AgentX corpus, or skip if it is unreachable."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
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
        trace_path, stats = self._prepare_trace()
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
            command = build_bench_command(self.base_url, trace_path, result_file)
            print(f"Running: {' '.join(command)}", flush=True)
            completed = subprocess.run(command, timeout=BENCH_TIMEOUT)
        finally:
            kill_process_tree(process.pid)

        self.assertEqual(
            completed.returncode, 0, "sglang.bench_serving agentic replay failed"
        )
        self.assertTrue(
            os.path.exists(result_file), f"No benchmark result at {result_file}"
        )

        with open(result_file, "r", encoding="utf-8") as f:
            result = json.loads(f.readlines()[-1])

        report = render_report(result, stats.as_markdown_rows())
        print(report, flush=True)
        if is_in_ci():
            write_github_step_summary(report)

        # No throughput gate: the numbers are tracked, and the run itself is the
        # regression guard -- this config only completes if MTP, the HiCache CPU
        # tier and the aiter MXFP4 path all survive a long-context agentic
        # workload. A turn that errored out never reaches the metrics.
        self.assertGreater(result.get("completed", 0), 0, "No agentic turns completed")
        self.assertGreater(
            result.get("output_throughput", 0), 0, "No output tokens generated"
        )


if __name__ == "__main__":
    unittest.main()
