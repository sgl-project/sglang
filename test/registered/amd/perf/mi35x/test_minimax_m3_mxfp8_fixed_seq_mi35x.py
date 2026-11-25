"""MI35x fixed-sequence benchmark for MiniMax-M3 MXFP8 (TP4), one server cell
per run.

The owner's recipe (aiter attention and MoE, auto KV, block-fp8 dense path,
quick INT4 all-reduce, context 73728) serves every workload, so the cells only
split the sweep: ``tp4-1k1k``, ``tp4-8k1k``, ``tp4-70k300``. The GSM8K gate
runs with thinking disabled. See ``sglang.test.amd.fixed_seq_common``.

Registry: nightly-perf-test-8-gpu-amd-mi35x-minimax-m3-mxfp8-fixed-seq suite
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
    suite="nightly-perf-test-8-gpu-amd-mi35x-minimax-m3-mxfp8-fixed-seq",
    nightly=True,
)

MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_OPT_USE_BF16_ROUTER_GEMM": "0",
    "SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE": "1",
    "SGLANG_M3_ALLOW_CUSTOM_AR": "1",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
    "SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ": "1",
}
SERVER_ARGS = [
    "--quantization",
    "mxfp8",
    "--dtype",
    "bfloat16",
    "--tp",
    "4",
    "--trust-remote-code",
    "--attention-backend",
    "aiter",
    "--moe-runner-backend",
    "aiter",
    "--kv-cache-dtype",
    "auto",
    "--chunked-prefill-size",
    "8192",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.79",
    "--context-length",
    "73728",
    "--reasoning-parser",
    "auto",
    "--tool-call-parser",
    "auto",
    "--watchdog-timeout",
    "3600",
]
CELLS = {
    f"tp4-{name}": Cell(
        tp=4,
        other_args=SERVER_ARGS,
        env=ENV,
        points=[(workload, conc) for conc in concs],
    )
    for name, workload, concs in (
        ("1k1k", WORKLOAD_1K1K, powers_of_two(2, 64)),
        ("8k1k", WORKLOAD_8K1K, powers_of_two(1, 256)),
        ("70k300", WORKLOAD_70K300, powers_of_two(1, 8)),
    )
}


# Not CustomTestCase: its CI retry would rerun a failed cell and report the
# second attempt's numbers as a pass.
@unittest.skipUnless(CELL, "FIXED_SEQ_CELL is not set")
class TestMiniMaxM3MXFP8FixedSeqMI35x(unittest.TestCase):
    def test_fixed_seq_cell(self):
        run_cell(
            self,
            recipe="minimax-m3-mxfp8",
            title="MiniMax-M3 MXFP8 (TP4)",
            model=MODEL,
            cells=CELLS,
            gsm8k_threshold=0.85,
            gsm8k_chat_template_kwargs={"thinking_mode": "disabled"},
        )


if __name__ == "__main__":
    unittest.main()
