"""Dense MLA decode-context-parallel correctness on Intel XPU.

XPU-only. DCP there runs on the intel_xpu backend, whose absorbed MLA decode
(sgl-kernel-xpu flash_mla_decode) returns a base-2 softmax LSE alongside the
partial output, so the model's cross-rank merge has something to weight. The
paths this suite covers, none of which the GQA DCP tests reach:

  - decode: Q all-gathered by the model, rank-local partial + LSE returned by
    the backend, merged by cp_lse_ag_out_rs_mla / dcp_a2a_lse_reduce
  - prefill: chunked-prefix MLA, where the model gathers the DCP-sharded prefix
    latent across the group so each rank attends the whole sequence for its own
    Q heads

DeepSeek-V2-Lite is the smallest dense-MLA model that fits two Arc Pro B60s
(29 GiB bf16 at tp=2).
"""

import json
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.utils import is_xpu, kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_xpu_ci(est_time=2400, suite="nightly-xpu-4-gpu", nightly=True)

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite"

TP_SIZE = 2
DCP_SIZE = 2

# Small on purpose: the long-context and concurrency probes below need requests
# to split across chunks, which is the only way to reach a non-empty prefix.
CHUNKED_PREFILL_SIZE = 2048

_SERVER_ARGS = [
    "--tp-size",
    str(TP_SIZE),
    "--dcp-size",
    str(DCP_SIZE),
    "--attention-backend",
    "intel_xpu",
    "--device",
    "xpu",
    "--trust-remote-code",
    "--random-seed",
    "0",
    "--mem-fraction-static",
    "0.72",
    "--chunked-prefill-size",
    str(CHUNKED_PREFILL_SIZE),
    "--disable-radix-cache",
]

_NEEDLE = "48217"

_FILLER = (
    "The committee reviewed the quarterly logistics report and noted no "
    "material change in throughput. "
)


def _generate(base_url, prompt, max_new_tokens):
    body = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
    }
    req = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def _needle_prompt(num_filler_sentences):
    half = num_filler_sentences // 2
    return (
        "Read the following document carefully.\n\n"
        + _FILLER * half
        + f"The secret access code is {_NEEDLE}. "
        + _FILLER * (num_filler_sentences - half)
        + "\n\nQuestion: What is the secret access code mentioned in the "
        "document?\nAnswer: The secret access code is"
    )


@unittest.skipUnless(is_xpu(), "intel_xpu MLA DCP is XPU-only")
class TestDeepseekV2LiteMlaDcpXPU(GSM8KMixin, CustomTestCase):
    """Accuracy gate plus the two prefill shapes dense MLA DCP has to survive."""

    model = MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Measured 0.340 at dcp=2 against 0.335 for the same tp=2 non-DCP baseline
    # (200 questions, 5-shot completion). A broken merge scores near zero, so the
    # gate sits well below both rather than tracking the model.
    gsm8k_score_threshold = 0.20
    gsm8k_num_examples = 100 if is_in_ci() else 200
    gsm8k_num_threads = 32
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=_SERVER_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_long_context_needle(self):
        """Retrieve a fact from a prompt spanning several prefill chunks.

        Only chunk 2 and later carry a prefix, which is what routes the model to
        the chunked-prefix MLA path where the prefix latent is DCP sharded and
        gathered. A single-chunk prompt never reaches it.
        """
        result = _generate(self.base_url, _needle_prompt(400), max_new_tokens=12)
        self.assertGreater(
            result["meta_info"]["prompt_tokens"],
            2 * CHUNKED_PREFILL_SIZE,
            "the probe must span more than two prefill chunks",
        )
        self.assertIn(_NEEDLE, result["text"])

    def test_concurrent_ragged_prefix_chunks(self):
        """Concurrent prompts of differing length must not produce NaN logits.

        Batching requests whose prompts split differently leaves some rows of a
        prefix chunk with no keys to attend. The varlen kernel returns NaN for
        those rows, and merging them into the chunk accumulator used to poison
        every request in the batch: the output degenerated to repeats of token 0
        and the logprobs came back null. Sequential requests never expose it,
        because one prompt alone gives every row the same prefix length.
        """
        prompts = [_needle_prompt(n) for n in (40, 150, 40, 150, 260, 40, 260, 150)]
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            results = list(pool.map(lambda p: _generate(self.base_url, p, 8), prompts))
        for prompt, result in zip(prompts, results):
            with self.subTest(prompt_tokens=result["meta_info"]["prompt_tokens"]):
                logprobs = [
                    lp for lp, _, _ in result["meta_info"]["output_token_logprobs"]
                ]
                self.assertTrue(
                    all(lp is not None for lp in logprobs),
                    f"null logprob means NaN logits: {result['text']!r}",
                )
                self.assertNotIn(
                    "!!!!",
                    result["text"],
                    "token-0 repeats are the NaN-logit signature",
                )


if __name__ == "__main__":
    unittest.main()
