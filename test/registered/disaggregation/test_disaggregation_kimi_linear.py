import unittest

from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=400, stage="extra-b", runner_config="4-gpu-h100")

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
SERVER_ENV = {"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0"}

PAGE_SIZE = 16
CHUNKED_PREFILL_SIZE = 64
DCP_SIZE = 2
# Prefill checkpoints the linear-attn state every 64 tokens, so shorter
# prompts never get a radix hit.
LINEAR_ATTN_CHECKPOINT_TOKENS = 64

# Deterministic so a cached-prefix prefill, which runs only the suffix, stays
# bit-exact with the reference's full prefill.
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


def _boundary_prompts(tokenizer):
    # Straddle the physical page, the DCP virtual page, and the prefill chunk.
    filler = tokenizer.encode(
        "Archive note: the weather was mild, the office lights were on, "
        "and no unusual event was reported. ",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        "\nSummarize the archive in one sentence:", add_special_tokens=False
    )
    virtual_page_size = PAGE_SIZE * DCP_SIZE
    lengths = sorted(
        {
            boundary + delta
            for boundary in (PAGE_SIZE, virtual_page_size, CHUNKED_PREFILL_SIZE)
            for delta in (-1, 0, 1)
        }
        | {2 * CHUNKED_PREFILL_SIZE + 1, 4 * CHUNKED_PREFILL_SIZE + 1}
    )
    body = filler * (max(lengths) // len(filler) + 1)
    return [body[: length - len(question)] + question for length in lengths]


# Every arm runs the same layout as its monolithic reference, so PD must match
# it token for token.
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
    parity_max_new_tokens = 8
    # Measured 0.0 on every prompt; a corrupted DCP relayout moved it by >= 0.05.
    parity_logprob_delta = 1e-3
    parity_cached_prefix_min_prompt_tokens = LINEAR_ATTN_CHECKPOINT_TOKENS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.parity_prompts = _boundary_prompts(tokenizer)


class TestKimiLinearDCPDisaggregation(TestKimiLinearTPDisaggregation):
    reference_parallel_args = ["--tp-size", "2", "--dcp-size", str(DCP_SIZE)]
    baseline_args = DCP_ARGS
    extra_prefill_args = DCP_ARGS
    extra_decode_args = DCP_ARGS + ["--dcp-size", str(DCP_SIZE)]
    # Deterministic flashinfer runs without the radix cache.
    parity_cached_prefix_min_prompt_tokens = None


if __name__ == "__main__":
    unittest.main()
