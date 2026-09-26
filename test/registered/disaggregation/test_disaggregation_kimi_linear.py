import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=300, stage="base-c", runner_config="4-gpu-h100")

KIMI_LINEAR_MODEL = "yujiepan/kimi-linear-tiny-random"
SERVER_ENV = {"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0"}

PAGE_SIZE = 16
CHUNKED_PREFILL_SIZE = 64
DCP_SIZE = 2
# Prefill checkpoints the linear-attn state every 64 tokens, so shorter
# prompts never get a radix hit.
LINEAR_ATTN_CHECKPOINT_TOKENS = 64

DETERMINISTIC_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-deterministic-inference",
    "--max-mamba-cache-size",
    "32",
    "--max-total-tokens",
    "4096",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]

DCP_ARGS = DETERMINISTIC_ARGS + [
    "--attention-backend",
    "flashinfer",
    "--page-size",
    str(PAGE_SIZE),
    "--chunked-prefill-size",
    str(CHUNKED_PREFILL_SIZE),
]


def _boundary_prompts():
    # Straddle the physical page, the DCP virtual page, and the prefill chunk.
    # Kept under 256 tokens: on this checkpoint a 256+ token cached prefix is
    # not bit-exact with a fresh prefill, and its near-flat logits flip on that.
    virtual_page_size = PAGE_SIZE * DCP_SIZE
    lengths = sorted(
        {
            boundary + delta
            for boundary in (PAGE_SIZE, virtual_page_size, CHUNKED_PREFILL_SIZE)
            for delta in (-1, 0, 1)
        }
        | {2 * CHUNKED_PREFILL_SIZE + 1, 3 * CHUNKED_PREFILL_SIZE + 1}
    )
    return [
        [1] + [100 + (length * 7 + i) % 1000 for i in range(length - 1)]
        for length in lengths
    ]


# Exact tokens need a bit-exact reference: this checkpoint's top-2 logprobs sit
# ~1e-3 apart, so P/D layouts with no matching monolithic run (heterogeneous TP,
# PP) flip on rounding alone.
class TestKimiLinearTPDisaggregation(PDLogprobParityMixin, PDDisaggregationServerBase):
    model = KIMI_LINEAR_MODEL
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    reference_parallel_args = ["--tp-size", "2"]
    baseline_args = DETERMINISTIC_ARGS
    extra_prefill_args = DETERMINISTIC_ARGS
    extra_decode_args = DETERMINISTIC_ARGS
    parity_prompts = _boundary_prompts()
    parity_max_new_tokens = 8
    parity_cached_prefix_min_prompt_tokens = LINEAR_ATTN_CHECKPOINT_TOKENS


class TestKimiLinearDCPDisaggregation(TestKimiLinearTPDisaggregation):
    reference_parallel_args = ["--tp-size", "2", "--dcp-size", str(DCP_SIZE)]
    baseline_args = DCP_ARGS
    extra_prefill_args = DCP_ARGS
    extra_decode_args = DCP_ARGS + ["--dcp-size", str(DCP_SIZE)]
    # Deterministic flashinfer runs without the radix cache.
    parity_cached_prefix_min_prompt_tokens = None


if __name__ == "__main__":
    unittest.main()
