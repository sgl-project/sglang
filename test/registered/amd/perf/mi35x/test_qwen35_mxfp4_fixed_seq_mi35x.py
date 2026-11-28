"""MI35x fixed-sequence benchmark for Qwen3.5-397B-A17B MXFP4 with EAGLE MTP,
one server cell per run.

The owner's TP2 recipe (aiter unified attention, fp8 KV, page 16, EAGLE 3/1/4)
caps --max-running-requests at the concurrency it is measured at, so each
concurrency is its own server cell (``tp2-c<conc>``) and runs 1K/1K, 8K/1K and
70K/300 on it. See ``sglang.test.amd.fixed_seq_common``.

Registry: nightly-perf-test-8-gpu-amd-mi35x-qwen35-mxfp4-fixed-seq suite
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
    suite="nightly-perf-test-8-gpu-amd-mi35x-qwen35-mxfp4-fixed-seq",
    nightly=True,
)

MODEL = "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2"
ENV = {
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT8",
    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
    "AITER_FLYDSL_FORCE": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "SGLANG_USE_AITER": "1",
}


def _server_args(conc: int) -> list:
    return [
        "--tp",
        "2",
        "--attention-backend",
        "aiter",
        "--trust-remote-code",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
        "--watchdog-timeout",
        "1200",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        str(conc),
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--disable-radix-cache",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--page-size",
        "16",
    ]


CELLS = {
    f"tp2-c{conc}": Cell(
        tp=2,
        other_args=_server_args(conc),
        env=ENV,
        points=[(w, conc) for w in (WORKLOAD_1K1K, WORKLOAD_8K1K, WORKLOAD_70K300)],
    )
    for conc in powers_of_two(4, 128)
}


# Not CustomTestCase: its CI retry would rerun a failed cell and report the
# second attempt's numbers as a pass.
@unittest.skipUnless(CELL, "FIXED_SEQ_CELL is not set")
class TestQwen35MXFP4FixedSeqMI35x(unittest.TestCase):
    def test_fixed_seq_cell(self):
        run_cell(
            self,
            recipe="qwen35-mxfp4",
            title="Qwen3.5-397B-A17B MXFP4 MTP (TP2)",
            model=MODEL,
            cells=CELLS,
            gsm8k_threshold=0.85,
            gsm8k_chat_template_kwargs={"enable_thinking": False},
        )


if __name__ == "__main__":
    unittest.main()
