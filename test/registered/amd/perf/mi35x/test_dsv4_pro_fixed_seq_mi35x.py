"""MI35x fixed-sequence benchmark for DeepSeek-V4-Pro (FP4 experts), one server
cell per run.

Two owner recipes, copied flag for flag:

* 1K/1K and 8K/1K: per workload, conc 2-32 on TP4, 64-128 on TP4 with DP4
  attention, 256-1024 on TP8 with DP8 attention (cells ``1k1k-tp4``,
  ``8k1k-tp8-dp8``, ...). --cuda-graph-max-bs-decode and --max-running-requests
  are the cell's highest concurrency, the context fits ISL + OSL. The recipe's
  --chat-template is left out: neither GSM8K nor the benchmark uses the chat
  endpoint.
* 70K/300: conc 2-4 on TP8 (``70k300-tp8``), 8-32 on TP8 with DP8 attention and
  the prefill delayer off (``70k300-tp8-dp8``).

Both recipes use two spellings main has dropped: --cuda-graph-max-bs is now
--cuda-graph-max-bs-decode, and --dp N --enable-dp-attention is now
--attn-dp-size N (which still divides --chunked-prefill-size).

See ``sglang.test.amd.fixed_seq_common``.

Registry: nightly-perf-test-8-gpu-amd-mi35x-dsv4-pro-fixed-seq suite
"""

import unittest

from sglang.test.amd.fixed_seq_common import (
    CELL,
    WORKLOAD_1K1K,
    WORKLOAD_8K1K,
    WORKLOAD_70K300,
    Cell,
    powers_of_two,
    run_cell,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=10800,
    suite="nightly-perf-test-8-gpu-amd-mi35x-dsv4-pro-fixed-seq",
    nightly=True,
)

MODEL = "deepseek-ai/DeepSeek-V4-Pro"
COMMON_ARGS = [
    "--trust-remote-code",
    "--disable-radix-cache",
    "--attention-backend",
    "dsv4",
    "--page-size",
    "256",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--disable-shared-experts-fusion",
    "--tool-call-parser",
    "deepseekv4",
    "--reasoning-parser",
    "deepseek-v4",
]
# What the recipes set only with DP attention.
DP_ENV = {
    "SGLANG_SHARED_EXPERT_TP1": "1",
    "SGLANG_DP_SHARED_EXPERT_LOCAL": "1",
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_DP_USE_REDUCE_SCATTER": "1",
}

SHORT_ENV = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "AITER_BF16_FP8_MOE_BOUND": "0",
}


def _short_cell(workload, tp: int, dp: int, concs: list) -> Cell:
    """A cell of the 1K/1K + 8K/1K recipe."""
    graph_bs = max(concs)
    chunked_prefill_size = workload.isl
    args = ["--tensor-parallel-size", str(tp)]
    env = dict(SHORT_ENV)
    if dp > 1:
        chunked_prefill_size = workload.isl * tp
        args += [
            "--attn-dp-size",
            str(dp),
            "--enable-prefill-delayer",
            "--enable-two-batch-overlap",
        ]
        env.update(DP_ENV, GPU_MAX_HW_QUEUES="5")
    args += COMMON_ARGS + [
        "--cuda-graph-max-bs-decode",
        str(graph_bs),
        "--max-running-requests",
        str(graph_bs),
        "--mem-fraction-static",
        "0.90",
        "--swa-full-tokens-ratio",
        "0.15",
        "--context-length",
        str(workload.isl + workload.osl + 256),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--watchdog-timeout",
        "1800",
    ]
    return Cell(tp=tp, other_args=args, env=env, points=[(workload, c) for c in concs])


LONG_ENV = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "0",
    # This and the other two marked ones are no longer read by sglang; kept so
    # the env is the owner's recipe as given.
    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "SGLANG_OPT_FP8_WO_A_GEMM": "false",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
    "SGLANG_OPT_USE_TOPK_V2": "false",
    # No longer read by sglang.
    "SGLANG_OPT_USE_AITER_INDEXER": "true",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    # No longer read by sglang.
    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
    "SGLANG_ROCM_USE_MULTI_STREAM": "false",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_EAGER_INPUT_NO_COPY": "true",
}


def _long_cell(dp: int, chunk: int, mem: str, concs: list) -> Cell:
    """A cell of the 70K/300 recipe (TP8; DP attention when ``dp`` > 1)."""
    args = ["--tp", "8"]
    env = dict(LONG_ENV)
    if dp > 1:
        # DP attention divides --chunked-prefill-size by the DP size.
        chunk *= dp
        args += ["--attn-dp-size", str(dp)]
        env.update(DP_ENV)
    args += COMMON_ARGS + [
        "--mem-fraction-static",
        mem,
        "--swa-full-tokens-ratio",
        "0.1",
        "--context-length",
        "73728",
        "--chunked-prefill-size",
        str(chunk),
        "--cuda-graph-max-bs-decode",
        "64",
        "--max-running-requests",
        "64",
    ]
    return Cell(
        tp=8, other_args=args, env=env, points=[(WORKLOAD_70K300, c) for c in concs]
    )


CELLS = {
    f"{name}-{layout}": _short_cell(workload, tp, dp, powers_of_two(low, high))
    for name, workload in (("1k1k", WORKLOAD_1K1K), ("8k1k", WORKLOAD_8K1K))
    for layout, tp, dp, low, high in (
        ("tp4", 4, 1, 2, 32),
        ("tp4-dp4", 4, 4, 64, 128),
        ("tp8-dp8", 8, 8, 256, 1024),
    )
}
CELLS["70k300-tp8"] = _long_cell(dp=1, chunk=32768, mem="0.92", concs=[2, 4])
CELLS["70k300-tp8-dp8"] = _long_cell(dp=8, chunk=16384, mem="0.80", concs=[8, 16, 32])


# Not CustomTestCase: its CI retry would rerun a failed cell and report the
# second attempt's numbers as a pass.
@unittest.skipUnless(CELL, "FIXED_SEQ_CELL is not set")
class TestDSV4ProFixedSeqMI35x(unittest.TestCase):
    def test_fixed_seq_cell(self):
        run_cell(
            self,
            recipe="dsv4-pro",
            title="DeepSeek-V4-Pro FP4",
            model=MODEL,
            cells=CELLS,
            gsm8k_threshold=0.90,
            gsm8k_chat_template_kwargs={"thinking": False},
        )


if __name__ == "__main__":
    unittest.main()
