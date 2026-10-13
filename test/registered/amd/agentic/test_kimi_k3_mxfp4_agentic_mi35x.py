"""MI35x agentic-replay throughput for Kimi-K3 MXFP4, the no-DCP / no-DSPARK arm.

Port of InferenceX's AgentX recipe
``benchmarks/single_node/agentic/kimik3_fp4_mi355x_sglang_nomtp.sh``
(https://github.com/SemiAnalysisAI/InferenceX, config key
``kimik3-fp4-mi355x-sglang-agentic-nomtp``, which is the orchestrator arm
``SERVER_TARGETS=release SERVER_RECIPES=base MTP_MODES=non_dspark DCP_SIZE=1
DCP_COMM_BACKEND=ag_rs TEST_SUITES=agentx``). Sibling of
``test_qwen35_mxfp4_agentic_mtp_mi35x.py``: same driver, same corpus, same
report, a different serving recipe. Dispatched by
``.github/workflows/nightly-benchmark-amd.yml``, which is where the
reporting-only benchmarks live.

Why this exists next to ``perf/mi35x/test_kimi_k3_perf_mi35x.py``: that test
sweeps ``bench_one_batch_server`` at 4k in / 512 out with
``--disable-radix-cache``, so it deliberately measures K3 with prefix reuse
turned *off*. This recipe is the opposite workload -- a 1M context window, a
145 GB host KV tier, ``page_size=128`` and fp8 KV are all in it because the
target is long, growing, cache-heavy coding-agent conversations. Nothing in the
existing K3 coverage moves when prefix-cache behaviour, the HiCache host tier,
or the KDA state pool under long context regresses; this does.

What the port keeps
-------------------
Parallelism and the DCP pair, aiter prefill/decode attention, bf16 weights with
fp8 KV, the mamba/KDA state-pool sizing, page size, chunked prefill and prefill
caps, the 1M context window, the ``kimi_k3`` reasoning and tool-call parsers,
the HiCache CPU tier, the multithreaded loader, the watchdog, and the derived
``max_running_requests`` / ``cuda_graph_max_bs_decode`` / ``max_mamba_cache_size``
arithmetic -- all the recipe's values, per concurrency (see :func:`recipe_arm`).

The recipe launches one server per concurrency point and sizes all three derived
numbers from that point, so this file does too: concurrency is a CI matrix axis,
not a loop inside one server. The cost is one 1.56 TB checkpoint load per point,
which is why the scheduled matrix carries three of the recipe's six
concurrencies rather than all of them.

What intentionally differs
--------------------------
* The driver is :mod:`sglang.test.agentic_trace_replay`, not aiperf, whose
  ``inferencex-agentx-mvp`` scenario is not installable from this repo, and the
  corpus comes from :mod:`sglang.test.agentic_trace_utils`. See that module for
  what survives the conversion. The absolute number is therefore *not*
  comparable to a published AgentX submission; it is comparable to this test's
  own history, which is what a nightly needs.
* The corpus is the uncapped ``cc-traces-weka-062126``, not the 256k-capped
  variant the Qwen port replays: K3's native window is 1M tokens, which is the
  cap AgentX picks the uncapped corpus for. The two only diverge past roughly
  turn 60 of a trajectory, so at the default turn cap they are equivalent; the
  uncapped one is the default so that raising ``AGENTIC_MAX_TURNS`` actually
  reaches the window this recipe configures.
* Of the recipe's ``ENABLE_YUYUN_KERNEL_ENV`` block only the flags that exist
  upstream are exported (see :data:`YUYUN_KERNEL_ENV`). The rest --
  ``SGLANG_K3_AITER_M16384_PROFILE``, ``SGLANG_K3_AITER_MLA_GATE``,
  ``SGLANG_K3_AITER_{KDA_GROUP64,B2_FUSIONS,MOE_PREROUTE_FP8,LATENT_TAIL_FP8}``,
  ``SGLANG_K3_AITER_MLA_Q_CACHE_FUSION``, ``SGLANG_K3_AITER_TUNED_MOE_FRONT*``,
  ``SGLANG_K3_PREROUTE_PREACTIVATED_SHARED``, ``SGLANG_K3_MOE_LATENT_MXFP4``,
  ``SGLANG_K3_PTPC_FP8*``, ``AITER_FLYDSL_DISABLE_MXMOE_V2`` and
  ``AITER_FLYDSL_STAGE1_SCRATCH_REUSE`` -- are read only by the pinned
  ``yuychang/sglang`` + ``yuychang/aiter`` ``kimi_k3_mxmoe_16k`` branches the
  recipe runs against, so exporting them here would be dead configuration that
  reads as if it were doing something. This is the recipe's own
  ``ENABLE_YUYUN_KERNEL_ENV=0`` ("AITER-only") arm plus the in-tree subset, and
  it is the main reason the throughput here is expected to sit below a published
  AgentX number on the same hardware.
* ``HSA_NO_SCRATCH_RECLAIM`` is not set: the recipe gates it on MEC firmware
  older than 177, and ``docker/rocm.Dockerfile`` already exports it for every
  ROCm CI container.
* DSPARK stays off, which is the arm this recipe pins
  (``MTP_MODES=non_dspark``). ``AGENTIC_SPEC_DECODE=auto`` restores the
  orchestrator's schedule (block size 7 up to concurrency 8, 3 up to 16, off
  above). When it is on, acceptance length is measured rather than pinned
  through ``SGLANG_SIMULATE_ACC_LEN``: AgentX pins it so a cross-framework
  comparison is not confounded by draft quality, but inside sglang's own
  nightly the draft model is fixed and a real acceptance drop is a regression we
  want to see. ``AGENTIC_SIMULATE_ACC_LEN`` restores the pin.
* DCP stays at 1, the recipe's arm. ``AGENTIC_DCP_SIZE`` above 1 switches the
  comm backend to ``a2a`` and turns on the in-tree Gluon MLA DCP path, but the
  recipe's other two DCP env vars (``SGLANG_USE_AITER_GLUON_MLA_DCP``,
  ``SGLANG_EXPERIMENTAL_AITER_DCP_FP8``) also only exist in the fork, so that
  arm measures the in-tree DCP path and not the recipe's.
* The recipe's GPU-drain wait, ``hf download`` and aiperf venv bootstrap are
  InferenceX harness concerns; CI has ``ensure_vram_clear.sh``, the runners'
  shared HF cache, and no aiperf.

There is no throughput gate -- the numbers go to the step summary and the run
itself is the guard, since this config only completes if the aiter MXFP4 MoE
path, the KDA state pool and the HiCache host tier all survive a long-context
agentic workload. The one hard check beyond "every turn completed" is that the
state pool did not clamp the batch below the concurrency in the report's title,
because a clamped run measures a different workload than the one it is labelled
with.

Registry: nightly-perf-8-gpu-mi35x-kimi-k3-agentic suite
"""

import os
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

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

# The 1.56 TB MXFP4 checkpoint dominates: the sibling K3 accuracy and perf jobs
# budget 150 min each for load plus a short measured phase, and a replay point
# adds ~45 min of workload on top at the largest concurrency.
register_amd_ci(
    est_time=12600,
    suite="nightly-perf-8-gpu-mi35x-kimi-k3-agentic",
    nightly=True,
)

# K3's weights are natively MXFP4, so there is no separate quantized repo to
# point at -- the same checkpoint the accuracy and perf tests serve.
MODEL_PATH = os.environ.get("KIMI_K3_MODEL_PATH", "moonshotai/Kimi-K3")

# AgentX runs this recipe twice: a throughput replay, and an EVAL_ONLY pass that
# scores the served model. Both modes share this file and its server
# configuration, and each skips the other's case rather than launching a second
# 1.56 TB server it would not use.
MODE = os.environ.get("AGENTIC_MODE", "replay")
if MODE not in ("replay", "eval"):
    # Both cases are skipUnless-gated on this, so a typo'd dispatch input would
    # otherwise publish an empty run as a passing benchmark.
    raise ValueError(f"AGENTIC_MODE must be 'replay' or 'eval', got {MODE!r}")

# The recipe refuses anything but TP8: MXFP4 K3 is ~195 GB per GPU of the 288 GB
# a gfx950 carries, so TP8 is the only topology it fits in. Not a knob.
TP_SIZE = 8
EP_SIZE = int(os.environ.get("AGENTIC_EP_SIZE", "1"))
CONCURRENCY = int(os.environ.get("AGENTIC_CONCURRENCY", "48"))

# AgentX runs every concurrency for the same wall-clock duration, so its low
# concurrencies simply serve fewer turns. This replay is closed-loop instead, so
# sizing the corpus to the concurrency gives every point the same per-channel
# work, which keeps the sweep comparable and each point inside the job timeout.
# The floor keeps the lowest concurrencies from measuring a handful of turns.
NUM_CONVERSATIONS = int(
    os.environ.get("AGENTIC_NUM_CONVERSATIONS") or max(8, 2 * CONCURRENCY)
)
# 24 turns carries each trajectory to 110-150k tokens of context in this corpus,
# which is what makes the host tier the thing being measured: K3's fp8 MLA KV is
# (kv_lora_rank 512 + qk_rope 64) bytes per token per full-attention layer over
# 24 such layers, i.e. ~13.5 KiB/token, so the recipe's 145 GB tier holds ~11M
# tokens -- about what 96 conversations at that context length occupy. Raising
# this is the lever for pushing further into the 1M window.
MAX_TURNS = int(os.environ.get("AGENTIC_MAX_TURNS", "24"))
# Unset replays the corpus' own per-turn reply lengths, which is what keeps the
# prompt growth faithful; set it to force a uniform decode length instead.
OUTPUT_LEN = int(os.environ.get("AGENTIC_OUTPUT_LEN", "0")) or None
KV_OFFLOADING = os.environ.get("AGENTIC_KV_OFFLOADING", "hicache")
# The recipe's MTP_MODES=non_dspark arm. "auto" restores the DSPARK schedule.
SPEC_DECODE = os.environ.get("AGENTIC_SPEC_DECODE", "false")
SIMULATE_ACC_LEN = os.environ.get("AGENTIC_SIMULATE_ACC_LEN", "")
DSPARK_DRAFT_PATH = os.environ.get("AGENTIC_DSPARK_DRAFT", "RadixArk/Kimi-K3-DSpark")
DCP_SIZE = int(os.environ.get("AGENTIC_DCP_SIZE", "1"))

# The uncapped corpus, because K3's native context is 1M tokens; see the module
# docstring for why that is the right cap for this model.
WEKA_TRACE_REPO = os.environ.get(
    "AGENTIC_TRACE_REPO", "semianalysisai/cc-traces-weka-062126"
)

# AgentX scores its eval pass with lm-eval's gsm8k. This asks the same questions
# through sglang's few-shot *completion* scorer, at the 8 shots
# test_kimi_k3_eval_mi35x.py uses, so the score is comparable to that test's.
# Completion rather than chat is not a style preference for K3: thinking is
# permanently on and the chat path routes the answer through reasoning_content,
# which would leave message.content empty and score 0. The gate sits below the
# 0.92 the plain K3 accuracy test holds (it measured 0.956), since this recipe
# also serves fp8 KV, the A4W4 MoE kernels and the aiter attention path.
EVAL_NUM_EXAMPLES = int(os.environ.get("AGENTIC_EVAL_NUM_EXAMPLES", "1319"))
EVAL_NUM_SHOTS = int(os.environ.get("AGENTIC_EVAL_NUM_SHOTS", "8"))
EVAL_MAX_NEW_TOKENS = int(os.environ.get("AGENTIC_EVAL_MAX_NEW_TOKENS", "512"))
EVAL_ACC_THRESHOLD = float(os.environ.get("AGENTIC_EVAL_ACC_THRESHOLD", "0.90"))

# A clamped state pool means the server admitted fewer requests than the
# concurrency this run reports, so the number would be mislabelled. Set to 1 to
# report it and keep going instead.
ALLOW_CLAMPED_BATCH = os.environ.get("AGENTIC_ALLOW_CLAMPED_BATCH", "0") == "1"

SERVER_LAUNCH_TIMEOUT = int(os.environ.get("AGENTIC_SERVER_TIMEOUT", "9000"))

RESULT_DIR = "performance_results_kimi_k3_mxfp4_agentic_mi35x"

# Fixed columns of the recipe's per-concurrency table.
MEM_FRACTION_STATIC = os.environ.get("AGENTIC_MEM_FRACTION_STATIC", "0.85")
PAGE_SIZE = os.environ.get("AGENTIC_PAGE_SIZE", "128")
CHUNKED_PREFILL_SIZE = os.environ.get("AGENTIC_CHUNKED_PREFILL_SIZE", "16384")
MAX_PREFILL_TOKENS = os.environ.get("AGENTIC_MAX_PREFILL_TOKENS", "16384")
CONTEXT_LENGTH = os.environ.get("AGENTIC_CONTEXT_LENGTH", "1048576")
WATCHDOG_TIMEOUT = os.environ.get("AGENTIC_WATCHDOG_TIMEOUT", "3600")
# The recipe's own ceiling on decode graph capture; K3 captures across 93
# attention and 92 MoE layers, so capturing above the admission ceiling is
# expensive and buys nothing.
CUDA_GRAPH_CAP = 256
# One state slot per running request, plus the retention/COW headroom and the
# overlap scheduler's ping-pong track buffer, is 5 under the default
# extra_buffer mamba radix strategy -- so the recipe's 5x multiplier is exactly
# max_running_requests' worth of slots and nothing more. ReplaySSM keeps this
# true under DSPARK: its intermediate states live on a fixed ring rather than on
# the per-request slot budget.
MAMBA_SLOTS_PER_REQUEST = int(os.environ.get("AGENTIC_MAMBA_SLOTS_PER_REQUEST", "5"))

HICACHE_WRITE_POLICY = os.environ.get("HICACHE_WRITE_POLICY", "write_through")
HICACHE_IO_BACKEND = os.environ.get("HICACHE_IO_BACKEND", "direct")
HICACHE_MEM_LAYOUT = os.environ.get("HICACHE_MEM_LAYOUT", "page_first_direct")

# Recipe env, restricted to what this tree reads. SGLANG_AITER_{MLA_GLUON,
# MLA_PERSIST,FP8_PREFILL_ATTN} all default to on, so the recipe is explicitly
# turning three aiter fast paths off; AITER_SITUV2_A4W4 selects the MXFP4-native
# A4W4 SiTU expert kernels, where the K3 accuracy and perf tests take the W4A8
# ones. PYTHONNOUSERSITE keeps a user site-packages directory from shadowing the
# container's sglang, as in the AgentX launcher. AITER_FLYDSL_FORCE is read by
# AITER itself rather than sglang and will not grep to anything in-tree.
COMMON_ENV = {
    "PYTHONNOUSERSITE": "1",
    "SGLANG_USE_AITER": "1",
    "SGLANG_AITER_K3_OPT": "1",
    "AITER_FLYDSL_FORCE": "1",
    "AITER_SITUV2_A4W4": "1",
    "AITER_SITUV2_A8W4": "0",
    "SGLANG_AITER_MLA_PERSIST": "0",
    "SGLANG_AITER_FP8_PREFILL_ATTN": "0",
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "SAFETENSORS_FAST_GPU": "1",
    "TORCH_BLAS_PREFER_HIPBLASLT": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "900",
}

# The in-tree half of the recipe's ENABLE_YUYUN_KERNEL_ENV block; the rest is
# fork-only, see the module docstring. Pinning the FlyDSL source to "sglang"
# rather than leaving it on "auto" also means a missing in-tree kernel fails the
# run instead of silently measuring AITER's copy.
YUYUN_KERNEL_ENV = {
    "SGLANG_K3_FLYDSL_SOURCE": "sglang",
    "SGLANG_K3_KDA_FUSED_BACKEND": "aiter",
    "SGLANG_K3_RADIX4_TOPK": "1",
    "SGLANG_MLA_DECODE_TUNE": "1",
}
ENABLE_YUYUN_KERNEL_ENV = os.environ.get("AGENTIC_YUYUN_KERNEL_ENV", "1") == "1"


@dataclass(frozen=True)
class RecipeArm:
    """One row of the recipe's per-concurrency table.

    ``concurrency`` is the workload; everything else is what the recipe derives
    from it to serve that workload.
    """

    concurrency: int
    max_running_requests: int
    cuda_graph_max_bs: int
    max_mamba_cache_size: int
    hicache_size_gb: int
    dcp_size: int
    dcp_comm_backend: str
    dspark_block_size: int
    golden_accept_len: float

    @property
    def spec_enabled(self) -> bool:
        return self.dspark_block_size > 0


def _int_env(name: str, default: int) -> int:
    """``${NAME:-default}``, so every derived number stays overridable."""
    return int(os.environ.get(name) or default)


def recipe_arm(
    concurrency: int,
    spec_decode: str = SPEC_DECODE,
    dcp_size: int = DCP_SIZE,
) -> RecipeArm:
    """The recipe's arm for one concurrency point.

    Reproduces the orchestrator's table::

        CONC  mem   mrr  graph  mamba(=5*mrr)  page  chunk  spec  dcp
          1   0.85    2      2        10        128  16384  off    1
          4   0.85    8      8        40        128  16384  off    1
          8   0.85   16     16        80        128  16384  off    1
         16   0.85   32     32       160        128  16384  off    1
         32   0.85   64     64       320        128  16384  off    1
         48   0.85   96     96       480        128  16384  off    1
    """
    max_running_requests = _int_env("AGENTIC_MAX_RUNNING_REQUESTS", 2 * concurrency)
    cuda_graph_max_bs = min(
        _int_env("AGENTIC_CUDA_GRAPH_MAX_BS", max_running_requests), CUDA_GRAPH_CAP
    )

    # DSPARK's schedule in the orchestrator: the draft only pays for itself
    # while decode is latency-bound, so it is off once the batch is large.
    dspark_block_size, golden_accept_len = 0, 0.0
    if spec_decode == "auto":
        if concurrency <= 8:
            dspark_block_size, golden_accept_len = 7, 3.84
        elif concurrency <= 16:
            dspark_block_size, golden_accept_len = 3, 3.00

    # Host DRAM per rank. The low row is the recipe's own: at concurrency 4 the
    # live working set is a fifth of what it is at 48, and the pinned host pool
    # is not free.
    hicache_size_gb = _int_env(
        "AGENTIC_HICACHE_SIZE_GB", 80 if concurrency <= 4 else 145
    )

    return RecipeArm(
        concurrency=concurrency,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        max_mamba_cache_size=_int_env(
            "AGENTIC_MAX_MAMBA_CACHE_SIZE",
            MAMBA_SLOTS_PER_REQUEST * max_running_requests,
        ),
        hicache_size_gb=hicache_size_gb,
        dcp_size=dcp_size,
        # ag_rs is the recipe's choice at dcp1, where there is no cross-rank
        # attention reduction to fuse and a2a would only add a hop.
        dcp_comm_backend=os.environ.get("AGENTIC_DCP_COMM_BACKEND")
        or ("ag_rs" if dcp_size == 1 else "a2a"),
        dspark_block_size=dspark_block_size,
        golden_accept_len=golden_accept_len,
    )


def build_server_args(arm: RecipeArm) -> list:
    """Server flags from kimik3_fp4_mi355x_sglang_nomtp.sh."""
    args = [
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--tp-size",
        str(TP_SIZE),
        "--dcp-size",
        str(arm.dcp_size),
        "--dcp-comm-backend",
        arm.dcp_comm_backend,
        "--prefill-attention-backend",
        "aiter",
        "--decode-attention-backend",
        "aiter",
        "--mem-fraction-static",
        MEM_FRACTION_STATIC,
        "--max-running-requests",
        str(arm.max_running_requests),
        "--cuda-graph-max-bs-decode",
        str(arm.cuda_graph_max_bs),
        "--max-mamba-cache-size",
        str(arm.max_mamba_cache_size),
        # bf16 halves the recurrent state against fp32 and so doubles the slots
        # the same budget holds; it also moves the linear-attention decode
        # backend on some platforms, which is why the recipe pins it explicitly.
        "--mamba-ssm-dtype",
        "bfloat16",
        # 4x the default: a coarser decode-time state snapshot costs resume
        # granularity, which an agentic workload spends on prefix hits rather
        # than mid-decode resumes, and saves the state copies.
        "--mamba-track-interval",
        "1024",
        "--page-size",
        PAGE_SIZE,
        "--chunked-prefill-size",
        CHUNKED_PREFILL_SIZE,
        "--max-prefill-tokens",
        MAX_PREFILL_TOKENS,
        "--context-length",
        CONTEXT_LENGTH,
        "--tool-call-parser",
        "kimi_k3",
        "--reasoning-parser",
        "kimi_k3",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
        "--watchdog-timeout",
        WATCHDOG_TIMEOUT,
        "--enable-metrics",
    ]

    if EP_SIZE > 1:
        args += ["--ep-size", str(EP_SIZE)]

    if arm.spec_enabled:
        args += [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            DSPARK_DRAFT_PATH,
            "--speculative-dspark-block-size",
            str(arm.dspark_block_size),
            # Keeps the draft's intermediate KDA states on a fixed ring instead
            # of the per-request state budget, which is what lets the 5x mamba
            # multiplier stay correct under speculation.
            "--enable-linear-replayssm-spec",
        ]

    if KV_OFFLOADING == "hicache":
        args += [
            "--enable-hierarchical-cache",
            "--hicache-size",
            str(arm.hicache_size_gb),
            "--hicache-write-policy",
            HICACHE_WRITE_POLICY,
            "--hicache-io-backend",
            HICACHE_IO_BACKEND,
            "--hicache-mem-layout",
            HICACHE_MEM_LAYOUT,
            "--enable-cache-report",
            "--skip-server-warmup",
        ]

    return args


def build_server_env(arm: RecipeArm) -> dict:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    if ENABLE_YUYUN_KERNEL_ENV:
        env.update(YUYUN_KERNEL_ENV)

    # At dcp1 there is no Gluon DCP path to take, which is the arm this recipe
    # pins; above 1 the recipe's other two DCP env vars are fork-only, so this
    # measures the in-tree path.
    env["SGLANG_AITER_MLA_GLUON"] = "1" if arm.dcp_size > 1 else "0"

    if arm.spec_enabled:
        env.setdefault("SGLANG_RAGGED_VERIFY_MODE", "static")
        # AgentX pins acceptance on the throughput side only; its eval pass
        # keeps real target-model verification, so a pinned length is dropped
        # there.
        if SIMULATE_ACC_LEN and MODE != "eval":
            env["SGLANG_SIMULATE_ACC_LEN"] = SIMULATE_ACC_LEN
            env["SGLANG_SIMULATE_ACC_METHOD"] = "match-expected"
            env["SGLANG_SIMULATE_ACC_TOKEN_MODE"] = "real-draft-token"
    return env


def _trace_cache_dir() -> str:
    """Prefer /sgl-data, the runners' persistent mount, so this converts once."""
    override = os.environ.get("AGENTIC_TRACE_CACHE_DIR")
    if override:
        return override
    if os.path.isdir("/sgl-data") and os.access("/sgl-data", os.W_OK):
        return "/sgl-data/agentic-traces"
    return os.path.join(tempfile.gettempdir(), "sglang-agentic-traces")


def resolved_batch_limits(base_url: str) -> dict:
    """What the server settled on, which is not always what it was asked for.

    The state pool is worst-case reserved and fail-loud, so a budget too small
    for ``--max-mamba-cache-size`` clamps ``max_running_requests`` down instead
    of overcommitting. The shell recipe greps the launch log for this; the
    server reports it directly.
    """
    try:
        response = requests.get(base_url + "/server_info", timeout=30)
        response.raise_for_status()
        server_info = response.json()
    except Exception as exc:  # pragma: no cover - reported, never fatal
        return {"error": repr(exc)}

    states = server_info.get("internal_states") or [{}]
    return {
        "max_running_requests": states[0].get("effective_max_running_requests_per_dp"),
        "max_mamba_cache_size": server_info.get("max_mamba_cache_size"),
        "max_total_num_tokens": server_info.get("max_total_num_tokens"),
    }


def _workload_rows(arm: RecipeArm, limits: dict) -> str:
    spec_cell = (
        f"DSPARK block {arm.dspark_block_size}"
        + (f" (simulated accept {SIMULATE_ACC_LEN})" if SIMULATE_ACC_LEN else "")
        if arm.spec_enabled
        else "off (non_dspark arm)"
    )
    rows = (
        f"| workload | value |\n| --- | --- |\n"
        f"| parallelism | tp{TP_SIZE} / ep{EP_SIZE} / dcp{arm.dcp_size}"
        f" ({arm.dcp_comm_backend}) |\n"
        f"| concurrency | {arm.concurrency} |\n"
        f"| max running requests | {arm.max_running_requests} |\n"
        f"| mamba cache slots | {arm.max_mamba_cache_size} |\n"
        f"| speculative decoding | {spec_cell} |\n"
        f"| kv offloading | {KV_OFFLOADING}"
    )
    if KV_OFFLOADING == "hicache":
        rows += f" ({arm.hicache_size_gb} GB host tier)"
    rows += " |\n"
    rows += f"| kernel env | {'recipe subset' if ENABLE_YUYUN_KERNEL_ENV else 'aiter only'} |\n"
    if limits.get("max_running_requests") is not None:
        rows += (
            f"| resolved max running requests | {limits['max_running_requests']} |\n"
        )
    if limits.get("max_total_num_tokens") is not None:
        rows += f"| kv pool (tokens) | {limits['max_total_num_tokens']} |\n"
    return rows


def render_report(
    arm: RecipeArm, result: dict, trace_summary: str, limits: dict
) -> str:
    cache = result.get("cache_report") or {}
    accept_length = result.get("accept_length")
    if accept_length:
        accept_cell = f"{accept_length:.2f}"
        if arm.golden_accept_len:
            accept_cell += f" (AgentX golden {arm.golden_accept_len})"
    else:
        accept_cell = "n/a (not reported)"

    report = (
        f"### Kimi-K3 MXFP4 agentic replay, no-DCP arm "
        f"[{os.getenv('GPU_CONFIG', 'MI35x')}]\n\n"
        f"{_workload_rows(arm, limits)}"
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


class _KimiK3AgenticServer(CustomTestCase):
    """Shared launcher: both modes serve the same recipe arm."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.arm = recipe_arm(CONCURRENCY)
        if KV_OFFLOADING != "hicache":
            print(
                f"WARNING: AGENTIC_KV_OFFLOADING={KV_OFFLOADING} -- a {CONTEXT_LENGTH}"
                " token context without the host tier clamps the KV pool.",
                flush=True,
            )
        print(f"Recipe arm: {cls.arm}", flush=True)

    def launch(self):
        return popen_launch_server(
            MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=build_server_args(self.arm),
            env=build_server_env(self.arm),
        )

    def publish(self, report: str):
        print(report, flush=True)
        if is_in_ci():
            write_github_step_summary(report)

    def assert_batch_not_clamped(self, limits: dict):
        resolved: Optional[int] = limits.get("max_running_requests")
        if resolved is None or resolved >= self.arm.concurrency:
            return
        message = (
            f"server resolved max_running_requests={resolved} below the measured "
            f"concurrency {self.arm.concurrency}: the state pool clamped the "
            f"batch, so raise AGENTIC_MAX_MAMBA_CACHE_SIZE (currently "
            f"{self.arm.max_mamba_cache_size}) or lower the concurrency"
        )
        if ALLOW_CLAMPED_BATCH:
            print(f"WARNING: {message}", flush=True)
        else:
            self.fail(message)


@unittest.skipUnless(MODE == "replay", f"AGENTIC_MODE={MODE} skips the replay")
class TestKimiK3Mxfp4AgenticMI35x(_KimiK3AgenticServer):
    """AgentX coding-agent replay against the MI35x MXFP4 K3 serving recipe."""

    def _prepare_trace(self, tokenizer):
        """Convert the published AgentX corpus, or skip if it is unreachable."""
        source = os.environ.get("AGENTIC_TRACE_SOURCE") or weka_trace_url(
            WEKA_TRACE_REPO
        )
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

        os.makedirs(RESULT_DIR, exist_ok=True)
        result_file = os.path.join(RESULT_DIR, "agentic_replay.jsonl")
        if os.path.exists(result_file):
            os.remove(result_file)

        process = self.launch()
        try:
            limits = resolved_batch_limits(self.base_url)
            print(f"Resolved batch limits: {limits}", flush=True)
            result = run_agentic_replay(
                base_url=self.base_url,
                model=MODEL_PATH,
                tokenizer=tokenizer,
                trace_path=trace_path,
                num_conversations=NUM_CONVERSATIONS,
                max_turns=MAX_TURNS,
                output_len=OUTPUT_LEN,
                max_concurrency=self.arm.concurrency,
                output_file=result_file,
            )
        finally:
            kill_process_tree(process.pid)

        self.publish(render_report(self.arm, result, stats.as_markdown_rows(), limits))

        # No throughput gate: the numbers are tracked, and the run itself is the
        # regression guard. A dropped turn takes the rest of its trajectory with
        # it, so a partial replay is a failure and not a smaller benchmark.
        self.assertEqual(
            result["failed"], 0, f"{result['failed']} turns failed: {result['errors']}"
        )
        self.assertGreater(result["output_throughput"], 0, "No output tokens generated")
        self.assert_batch_not_clamped(limits)


@unittest.skipUnless(MODE == "eval", f"AGENTIC_MODE={MODE} skips the eval pass")
class TestKimiK3Mxfp4AgenticEvalMI35x(_KimiK3AgenticServer):
    """AgentX's EVAL_ONLY pass: score the same serving recipe on GSM8K."""

    def test_gsm8k(self):
        process = self.launch()
        try:
            limits = resolved_batch_limits(self.base_url)
            requests.get(self.base_url + "/flush_cache")
            metrics = run_eval(
                SimpleNamespace(
                    base_url=self.base_url,
                    model=MODEL_PATH,
                    eval_name="gsm8k",
                    api="completion",
                    num_shots=EVAL_NUM_SHOTS,
                    max_tokens=EVAL_MAX_NEW_TOKENS,
                    num_examples=EVAL_NUM_EXAMPLES,
                    # More in flight than the server admits only queues; this
                    # keeps the scorer's fan-out at the batch being measured.
                    num_threads=self.arm.max_running_requests,
                )
            )
        finally:
            kill_process_tree(process.pid)

        accuracy = metrics["score"]
        passed = accuracy >= EVAL_ACC_THRESHOLD
        self.publish(
            f"### Kimi-K3 MXFP4 agentic recipe, GSM8K "
            f"[{os.getenv('GPU_CONFIG', 'MI35x')}]\n\n"
            f"{_workload_rows(self.arm, limits)}\n"
            f"| metric | value |\n| --- | --- |\n"
            f"| gsm8k questions | {EVAL_NUM_EXAMPLES} ({EVAL_NUM_SHOTS}-shot) |\n"
            f"| gsm8k accuracy | {accuracy:.3f} |\n"
            f"| threshold | {EVAL_ACC_THRESHOLD} |\n"
            f"| status | {'PASS' if passed else 'FAIL'} |\n"
        )

        self.assertGreaterEqual(
            accuracy,
            EVAL_ACC_THRESHOLD,
            f"Kimi-K3 agentic-recipe accuracy {accuracy:.3f} below threshold "
            f"{EVAL_ACC_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
