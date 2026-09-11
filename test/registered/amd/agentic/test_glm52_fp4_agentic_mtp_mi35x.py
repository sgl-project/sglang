"""MI35x nightly agentic-coding performance benchmark for GLM-5.2-MXFP4 with MTP.

Ported from the AgentX recipe
``benchmarks/single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh`` and its
``glm5.2-fp4-mi355x-sglang-agentic-mtp`` config entry in
``SemiAnalysisAI/InferenceX``. That recipe measures ``amd/GLM-5.2-MXFP4`` on
MI355X under a multi-turn coding-agent workload with EAGLE/MTP speculative
decoding, over two arms:

1. TP4 / EP4 with HiCache offloading the KV overflow to host DRAM, swept over
   concurrency 1-12. This is the throughput arm.
2. TP8 / EP1 fully GPU-resident, swept over concurrency 1-10. EP=1 drops the
   MoE all-to-all, which wins on ITL at low concurrency.

What this port keeps and what it changes:

- The server configuration is reproduced flag for flag, because that is what a
  nightly regression check is for. The one restructuring is that the recipe
  launches a server per concurrency point and sizes ``--max-running-requests``
  and the decode CUDA-graph batch list from that point's concurrency, whereas
  here one server serves the whole sweep and both are sized from the largest
  concurrency in it.
- The workload is a real multi-turn corpus where the runner has one and a
  synthesized one otherwise. The SemiAnalysis ``cc-traces-weka`` corpus the
  recipe names is gated and reachable only through AIPerf's loader, but the
  MI35x runners carry staged traces under ``/sgl-data/agentic-traces`` (64
  conversations over 584 turns, averaging 86K prompt tokens per turn and
  peaking at 231K), and the nightly points ``SGLANG_AGENTIC_TRACE_PATH`` at
  one. Without that variable ``sglang.test.agentic_bench_utils`` builds a
  corpus with the same shape -- a shared agent scaffold, a per-session
  repository context, and per-turn tool output, sized so a session opens at
  ~39K tokens and reaches ~68K by its eighth turn -- which is lighter, so the
  two are not comparable with each other. Neither is comparable against
  AgentX's published numbers, which measure a fixed duration rather than fixed
  work; run-over-run numbers here are.
- The recipe pins ``SGLANG_SIMULATE_ACC_LEN=3.61``, which makes the server
  report a fixed acceptance length instead of running draft verification. That
  suits AgentX's projections and defeats a regression check, so this test runs
  real MTP and reports the measured acceptance length per point.
- The recipe also sets ``SGLANG_OPT_USE_TOPK_V2=false`` for gfx950. Upstream
  now decides that per architecture and deliberately turns the fused top-k v2
  kernel back on for the GLM-5.x DSA family on ROCm (see
  ``arg_groups/model_hook.py``), so the override is not carried over.

Registry: nightly-perf-8-gpu-mi35x-glm52-fp4-agentic-mtp suite
"""

import os
import unittest

from sglang.test.agentic_bench_utils import (
    AgenticTraceSpec,
    generate_agentic_markdown_report,
    run_agentic_benchmark,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
    is_in_ci,
    write_github_step_summary,
)

register_amd_ci(
    est_time=10800,
    suite="nightly-perf-8-gpu-mi35x-glm52-fp4-agentic-mtp",
    nightly=True,
)

GLM_52_MXFP4_MODEL_PATH = os.environ.get("GLM52_MXFP4_MODEL_PATH", "amd/GLM-5.2-MXFP4")
RESULT_DIR = "performance_results_glm52_fp4_agentic_mtp_mi35x"
# Generous enough to absorb a cold weight cache on the first run of a new
# checkpoint; the job's own timeout is the real backstop.
SERVER_LAUNCH_TIMEOUT = 5400
# Each point replays two waves of conversations, so this is a hang detector
# rather than a budget.
BENCH_TIMEOUT_PER_POINT = 1800

# The recipe sweeps [1, 2, 4, 8, 10, 12] and [1, 2, 4, 10]. Both are trimmed to
# the endpoints and midpoints that carry the regression signal, since each
# extra point costs a full replay of the corpus.
HICACHE_CONCURRENCIES = _parse_int_list_env("AGENTIC_HICACHE_CONCURRENCIES", "1,4,8,12")
GPU_RESIDENT_CONCURRENCIES = _parse_int_list_env("AGENTIC_TP8_CONCURRENCIES", "1,4,10")

# Defaults put ~39K tokens in each session's first turn and grow it to ~68K by
# the eighth, which is the range where prefix reuse and the host tier decide
# the result.
TRACE_SPEC = AgenticTraceSpec()

# Common to both arms, straight from the recipe.
BASE_SERVER_ARGS = [
    "--trust-remote-code",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--dsa-prefill-backend",
    "tilelang",
    "--dsa-decode-backend",
    "tilelang",
    # GLM-5.2 emits GLM-4.7-style tool calls, so glm47 is what populates
    # structured message.tool_calls; glm45 keeps hybrid thinking in
    # reasoning_content.
    "--tool-call-parser",
    "glm47",
    "--reasoning-parser",
    "glm45",
    # 32k chunks let the scheduler interleave decode between prefill chunks,
    # which is what keeps TPOT bounded once several long agent sessions are
    # prefilling at once.
    "--chunked-prefill-size",
    "32768",
    "--mem-fraction-static",
    "0.85",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "5",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "6",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1800",
    "--enable-metrics",
]

SERVER_ENV = {
    # A conversation waiting on the concurrency semaphore holds an idle pooled
    # connection, and the 5s default closes it out from under the client right
    # as its first turn is admitted.
    "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
}


def _concurrency_args(concurrencies):
    """Size the in-flight slots and decode graph list for the whole sweep.

    2x the largest concurrency, per the recipe: MTP draft+verify transiently
    batches more than one request per session, and without the headroom the
    scheduler stalls on burst.
    """
    max_running_requests = min(2 * max(concurrencies), 256)
    cuda_graph_max_bs = min(max_running_requests, 64)
    return [
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-max-bs-decode",
        str(cuda_graph_max_bs),
    ]


class TestGLM52FP4AgenticMTPMI35x(unittest.TestCase):
    """GLM-5.2-MXFP4 agentic-coding replay with MTP on AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.full_report = f"## {cls.__name__}\n"

    @classmethod
    def tearDownClass(cls):
        if is_in_ci():
            write_github_step_summary(cls.full_report)
        print(cls.full_report)

    def _run_arm(self, variant, parallel_args, cache_args, concurrencies):
        server_args = (
            BASE_SERVER_ARGS
            + parallel_args
            + cache_args
            + _concurrency_args(concurrencies)
        )
        points = run_agentic_benchmark(
            model_path=GLM_52_MXFP4_MODEL_PATH,
            base_url=self.base_url,
            server_args=server_args,
            result_dir=os.path.join(RESULT_DIR, variant),
            concurrencies=concurrencies,
            trace_spec=TRACE_SPEC,
            server_launch_timeout=SERVER_LAUNCH_TIMEOUT,
            bench_timeout_per_point=BENCH_TIMEOUT_PER_POINT,
            env=SERVER_ENV,
        )

        type(self).full_report += (
            generate_agentic_markdown_report(
                points, f"{GLM_52_MXFP4_MODEL_PATH} ({variant}) [MI35x]"
            )
            + "\n"
        )

        # A replay that dropped turns produces throughput numbers for a
        # workload nobody ran, so fail rather than publish the partial result.
        for point in points:
            self.assertEqual(
                point.failed_turns,
                0,
                f"{variant} concurrency {point.concurrency}: "
                f"{point.failed_turns} of {point.total_turns} turns failed",
            )

    def test_agentic_tp4_ep4_hicache(self):
        """Throughput arm: TP4/EP4 with the KV overflow offloaded to host DRAM."""
        self._run_arm(
            variant="tp4-ep4-hicache",
            parallel_args=["--tp", "4", "--ep-size", "4"],
            cache_args=[
                "--enable-hierarchical-cache",
                # ~2.9 TB pinned at TP4, within the ~3.0 TB these nodes carry.
                "--hicache-ratio",
                "1.5",
                # Skips host writes for KV blocks that cannot be reused, which
                # takes load off the host bus without costing hit rate.
                "--hicache-write-policy",
                "write_through_selective",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
            ],
            concurrencies=HICACHE_CONCURRENCIES,
        )

    def test_agentic_tp8_ep1_gpu_resident(self):
        """Latency arm: TP8/EP1, GPU-resident KV, no MoE all-to-all."""
        self._run_arm(
            variant="tp8-ep1-gpu-resident",
            parallel_args=["--tp", "8", "--ep-size", "1"],
            cache_args=[],
            concurrencies=GPU_RESIDENT_CONCURRENCIES,
        )


if __name__ == "__main__":
    unittest.main()
