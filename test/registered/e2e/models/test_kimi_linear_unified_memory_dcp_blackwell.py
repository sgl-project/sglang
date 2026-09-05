"""Kimi-Linear on the unified memory pool with decode context parallelism,
served by the trtllm_mla family.

Two read-index routes exist under `--enable-unified-memory --dcp-size > 1`.
`test_kimi_linear_unified_memory.py` covers the flashinfer one, where
`plan_dcp_decode_metadata` compacts this rank's ids and
`translate_dcp_read_ids` converts them. This file covers the other: a block
table built by `create_mla_kv_page_table_for_dcp`, which gathers a rank's
cyclic slice and must take the pool's virtual->physical page table on the way.
An entry that skipped that hop names whichever page sits there now, so the
failure is stale KV rather than a crash -- which is what the GSM8K bar and the
prefix-cache replay below are here to catch.

`cutedsl_mla` stands in for the family: it is the DCP-native MLA decode kernel
on Blackwell (the only backend flashinfer accepts `enable_dcp=True` for) and
what the B300 hosts run, and prefill resolves to `trtllm_mla`, so one server
covers both halves of the page-table contract. `trtllm_mla` and
`tokenspeed_mla` share that builder and were checked by hand against the
static pool (see below) rather than given a CI server each.

Blackwell-only, hence its own module: the Hopper suite next door cannot host
it. In extra-b rather than nightly because what it guards is a per-PR argument
gate, which a regression closes at boot rather than degrading overnight.

Measured on B300 (1 sigma ~= 0.02). This file scores 0.920 and takes 2.5 min.
Matched unified-vs-static pairs, run by hand at 200 questions, to show the two
pools agree:

                                    unified / static
    TP2 DCP2  cutedsl_mla           0.905 / 0.905
    TP2 DCP2  trtllm_mla            0.895 / 0.895
    TP2 DCP2  tokenspeed_mla        0.890 / 0.895, repeat 0.890 / 0.885
    TP4 DCP4  cutedsl_mla           0.910 / 0.900

The cutedsl_mla and trtllm_mla pairs reproduced exactly. tokenspeed_mla moved
on a repeat of the same questions on BOTH pools, so its spread is the backend,
not the pool. The TP4 DCP4 row is there because it widens the virtual page to
4 x 64 = 256, past what this file exercises.

    python -m pytest test/registered/models_e2e/test_kimi_linear_unified_memory_dcp_blackwell.py -v
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=250, stage="extra-b", runner_config="4-gpu-b200")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


class TestKimiLinearUnifiedMemoryDCPCuteDsl(
    GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    """cutedsl_mla decode, trtllm_mla prefill (what cutedsl_mla resolves to)."""

    model = KIMI_LINEAR_MODEL
    cache_chunk_size = 64
    gsm8k_score_threshold = 0.88
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--dcp-size",
        "2",
        "--attention-backend",
        "cutedsl_mla",
        "--chunked-prefill-size",
        "2048",
        # The per-batch DCP gather buffer is sized by the batch's total KV, so
        # the default static fraction leaves a 200-question burst ~1 GB and it
        # OOMs inside an NCCL collective.
        "--mem-fraction-static",
        "0.80",
        "--max-running-requests",
        "128",
        "--cuda-graph-max-bs-decode",
        "128",
        "--enable-unified-memory",
    ]


if __name__ == "__main__":
    unittest.main()
