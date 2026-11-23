"""Dense MLA decode-context-parallel correctness on Intel XPU.

XPU-only. The CUDA MLA DCP path is covered by test_dsv31_dcp8_gsm8k.py
(DeepSeek-V3.1 + flashinfer); that file's XPU leg is GQA, so until now no test
exercised MLA under DCP on XPU at all. MLA there must route through Triton --
the xpu backend's decode kernels emit no usable softmax LSE -- and the Triton
backend is not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS, so the model takes the
concatenated-QK branch that this suite guards:

  - prefill (prefix == 0): the dense MHA companion layer, whose latent write has
    to resolve the DCP owner rule, and whose K/V heads must not be re-sliced
    across the DCP group
  - prefill (prefix > 0, i.e. chunk 2+ of a long prompt): the absorbed MLA
    extend path, where each rank attends its own prefix shard and the partials
    are merged by LSE
  - decode: Q all-gathered by the model, rank-local partial + natural-log LSE
    returned by the backend, merged by cp_lse_ag_out_rs_mla

DeepSeek-V2-Lite is the smallest dense-MLA model that fits 4 Arc Pro B60s
(29 GiB bf16 at tp=4); DeepSeek-V3.1 does not.
"""

import json
import unittest
import urllib.request

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

TP_SIZE = 4
DCP_SIZE = 2

# chunked_prefill_size is deliberately small so the long-context probe below
# spans several chunks and reaches the prefix > 0 extend path.
CHUNKED_PREFILL_SIZE = 2048

_SERVER_ARGS = [
    "--tp-size",
    str(TP_SIZE),
    "--attention-backend",
    "triton",
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

_DCP_ARGS = ["--dcp-size", str(DCP_SIZE)]

_NEEDLE = "48217"

_PARITY_PROMPTS = [
    "The capital city of France is",
    "The largest planet in our solar system is",
    "In 1969, humans first landed on",
]


def _generate(base_url, prompt, max_new_tokens, return_logprob=False):
    body = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
    }
    if return_logprob:
        body["return_logprob"] = True
        body["logprob_start_len"] = 0
    req = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def _build_needle_prompt(num_filler_sentences=400):
    """A prompt of ~6k tokens with the needle buried in the middle."""
    filler = (
        "The committee reviewed the quarterly logistics report and noted no "
        "material change in throughput. "
    )
    half = num_filler_sentences // 2
    return (
        "Read the following document carefully.\n\n"
        + filler * half
        + f"The secret access code is {_NEEDLE}. "
        + filler * (num_filler_sentences - half)
        + "\n\nQuestion: What is the secret access code mentioned in the "
        "document?\nAnswer: The secret access code is"
    )


@unittest.skipUnless(is_xpu(), "Dense MLA DCP via Triton is validated on XPU only")
class TestDeepseekV2LiteMlaDcpXPU(GSM8KMixin, CustomTestCase):
    """GSM8K gate plus a multi-chunk prefill probe under dense MLA DCP.

    The gate is sized to catch a broken cross-rank merge, not to certify a 16B
    base model: a wrong merge does not degrade accuracy gracefully, it collapses
    GSM8K to near zero and emits fluent but unrelated text.
    """

    model = MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Measured 0.335 at dcp=2 against 0.315 for the same tp=4 non-DCP baseline
    # (200 questions, 5-shot completion). A broken merge scores near zero, so
    # the gate sits well below both rather than tracking the model.
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
            other_args=_DCP_ARGS + _SERVER_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_long_context_needle(self):
        """Retrieve a fact from a prompt spanning several prefill chunks.

        Only chunk 2 and later carry a non-empty prefix, which is what routes
        the model to the absorbed MLA extend path where the prefix KV is DCP
        sharded. A single-chunk prompt never reaches it.
        """
        prompt = _build_needle_prompt()
        result = _generate(self.base_url, prompt, max_new_tokens=12)
        prompt_tokens = result["meta_info"]["prompt_tokens"]
        self.assertGreater(
            prompt_tokens,
            2 * CHUNKED_PREFILL_SIZE,
            "the probe must span more than two prefill chunks to reach the "
            "prefix > 0 extend path",
        )
        self.assertIn(_NEEDLE, result["text"])


@unittest.skipUnless(is_xpu(), "Dense MLA DCP via Triton is validated on XPU only")
@unittest.skipIf(is_in_ci(), "Needs two server launches; run locally.")
class TestDeepseekV2LiteMlaDcpParityXPU(CustomTestCase):
    """Compare DCP against a same-TP non-DCP baseline on the prefill path.

    Restricted to the first generated token on purpose. Prefill is bit-exact
    between the two configurations, so any drift is a real bug; decode is not,
    because the cross-rank LSE merge reorders the softmax sum, and at
    temperature 0 a near-tie can flip the argmax and send the continuations down
    different but equally valid paths.
    """

    model = MODEL_PATH
    baseline_url = "http://127.0.0.1:31501"
    dcp_url = "http://127.0.0.1:31502"

    @classmethod
    def setUpClass(cls):
        cls.baseline_process = popen_launch_server(
            cls.model,
            cls.baseline_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=_SERVER_ARGS,
        )
        cls.dcp_process = popen_launch_server(
            cls.model,
            cls.dcp_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=_DCP_ARGS + _SERVER_ARGS + ["--base-gpu-id", str(TP_SIZE)],
        )

    @classmethod
    def tearDownClass(cls):
        for process in (cls.baseline_process, cls.dcp_process):
            kill_process_tree(process.pid, wait_timeout=60)

    def test_prefill_logprob_parity(self):
        for prompt in _PARITY_PROMPTS:
            with self.subTest(prompt=prompt):
                baseline = _generate(self.baseline_url, prompt, 1, return_logprob=True)
                dcp = _generate(self.dcp_url, prompt, 1, return_logprob=True)
                self.assertEqual(baseline["text"], dcp["text"])
                baseline_lp = baseline["meta_info"]["output_token_logprobs"][0][0]
                dcp_lp = dcp["meta_info"]["output_token_logprobs"][0][0]
                self.assertAlmostEqual(baseline_lp, dcp_lp, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
