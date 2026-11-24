"""Dense MLA decode context parallelism on Intel XPU.

The GQA DCP tests never reach the absorbed MLA decode or the chunked-prefix
MLA prefill, which are the two paths DCP changes for an MLA model.
"""

import json
import math
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=2400, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite"

TP_SIZE = 2
DCP_SIZE = 2

# Small enough that the probes below span several chunks, which is the only way
# to reach a non-empty prefix.
CHUNKED_PREFILL_SIZE = 2048

_BASE_ARGS = [
    "--tp-size",
    str(TP_SIZE),
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

_DCP_ARGS = ["--dcp-size", str(DCP_SIZE)] + _BASE_ARGS
_NON_DCP_ARGS = ["--dcp-size", "1"] + _BASE_ARGS

_NEEDLE = "48217"

_FILLER = (
    "The committee reviewed the quarterly logistics report and noted no "
    "material change in throughput. "
)

# Measured spreads on this model stay under 0.21 nats; a mis-weighted merge
# moves logprobs by whole nats.
_LOGPROB_TOL = 0.5
# Batch composition is not reproducible, so a greedy near-tie can flip either
# way. Measured 7/8 at both DCP=1 and DCP=2.
_MIN_TOKEN_AGREEMENT = 0.75


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
        payload = json.loads(resp.read())
    return {
        "text": payload["text"],
        "ids": payload["output_ids"],
        "logprobs": [lp for lp, _, _ in payload["meta_info"]["output_token_logprobs"]],
        "prompt_tokens": payload["meta_info"]["prompt_tokens"],
    }


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


def _tagged_prompt(tag, num_filler_sentences):
    return (
        f"Document {tag}. The verification code for document {tag} is "
        f"{1000 + tag}. " + _FILLER * num_filler_sentences + f"\n\nQuestion: What "
        f"is the verification code for document {tag}?\n"
        "Answer: The verification code is"
    )


# Ragged, so requests in one prefill batch split differently and some rows of a
# prefix chunk end up with no keys.
_RAGGED_SPECS = [(0, 10), (1, 150), (2, 10), (3, 260), (4, 150), (5, 10), (6, 260)]

_PARITY_PROMPTS = [
    "The capital city of France is",
    "Question: What is 17 multiplied by 23? Answer:",
    "The largest planet in our solar system is",
    "In 1969, humans first landed on",
    "Water boils at a temperature of",
    _needle_prompt(40),
    _needle_prompt(150),
]


def _assert_sane(test, result, label):
    test.assertTrue(
        result["ids"], f"{label}: empty output, the request produced nothing"
    )
    test.assertTrue(
        all(lp is not None for lp in result["logprobs"]),
        f"{label}: a null logprob means NaN logits: {result['text']!r}",
    )
    test.assertTrue(
        all(math.isfinite(lp) for lp in result["logprobs"]),
        f"{label}: non-finite logprob: {result['logprobs']}",
    )


def _worst_logprob_delta(a, b):
    # Stop at the first differing token: past it the two runs score different
    # tokens and the gap no longer measures numerical drift.
    ids_a, ids_b = a.get("ids") or [], b.get("ids") or []
    limit = len(ids_a)
    for i, (x, y) in enumerate(zip(ids_a, ids_b)):
        if x != y:
            limit = i
            break
    pairs = [
        (x, y)
        for x, y in zip(a["logprobs"][:limit], b["logprobs"][:limit])
        if x is not None and y is not None
    ]
    return max((abs(x - y) for x, y in pairs), default=0.0)


@unittest.skipUnless(is_xpu(), "intel_xpu MLA DCP is XPU-only")
class TestDeepseekV2LiteMlaDcpXPU(GSM8KMixin, CustomTestCase):
    model = MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # A smoke gate, not an accuracy-parity bar: a broken merge scores near zero.
    # Parity against DCP=1 is the class below.
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
            other_args=_DCP_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            terminate_and_kill_process_tree(cls.process)

    def test_long_context_needle(self):
        """Only chunk 2 and later carry a prefix, the chunked-prefix MLA path."""
        result = _generate(self.base_url, _needle_prompt(400), max_new_tokens=12)
        self.assertGreater(
            result["prompt_tokens"],
            2 * CHUNKED_PREFILL_SIZE,
            "the probe must span more than two prefill chunks",
        )
        self.assertIn(_NEEDLE, result["text"])

    def test_concurrent_ragged_prefix_chunks(self):
        """A zero-key prefix row must not poison the rest of its batch.

        Such a row came back NaN and merge_state spread it: every request in the
        batch degenerated to repeats of token 0 with null logprobs.
        """
        prompts = [_tagged_prompt(t, r) for t, r in _RAGGED_SPECS]
        sequential = [_generate(self.base_url, p, 10) for p in prompts]
        for (tag, _), result in zip(_RAGGED_SPECS, sequential):
            _assert_sane(self, result, f"sequential tag={tag}")

        for concurrency in (4, len(prompts)):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                concurrent = list(
                    pool.map(lambda p: _generate(self.base_url, p, 10), prompts)
                )
            self.assertEqual(
                len(concurrent), len(prompts), "a concurrent request went missing"
            )

            agree = 0
            for (tag, _), seq, con in zip(_RAGGED_SPECS, sequential, concurrent):
                label = f"concurrency={concurrency} tag={tag}"
                with self.subTest(concurrency=concurrency, tag=tag):
                    _assert_sane(self, con, label)
                    for other, _ in _RAGGED_SPECS:
                        if other != tag:
                            self.assertNotIn(
                                str(1000 + other),
                                con["text"],
                                f"{label}: cross-request contamination",
                            )
                    self.assertLessEqual(
                        _worst_logprob_delta(seq, con),
                        _LOGPROB_TOL,
                        f"{label}: logprobs drifted from the sequential run",
                    )
                agree += seq["ids"] == con["ids"]

            self.assertGreaterEqual(
                agree / len(prompts),
                _MIN_TOKEN_AGREEMENT,
                f"concurrency={concurrency}: only {agree}/{len(prompts)} "
                "generations reproduced their sequential tokens",
            )


@unittest.skipUnless(is_xpu(), "intel_xpu MLA DCP is XPU-only")
@unittest.skipIf(is_in_ci(), "Needs two concurrent servers (4 XPUs); run locally.")
class TestDeepseekV2LiteMlaDcpParityXPU(CustomTestCase):
    # Greedy at concurrency 1 on both sides, so the cross-rank merge is the only
    # difference left.
    model = MODEL_PATH
    non_dcp_url = "http://127.0.0.1:31501"
    dcp_url = "http://127.0.0.1:31502"

    @classmethod
    def setUpClass(cls):
        cls.non_dcp_process = popen_launch_server(
            cls.model,
            cls.non_dcp_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=_NON_DCP_ARGS,
        )
        cls.dcp_process = popen_launch_server(
            cls.model,
            cls.dcp_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=_DCP_ARGS + ["--base-gpu-id", str(TP_SIZE)],
        )

    @classmethod
    def tearDownClass(cls):
        for name in ("non_dcp_process", "dcp_process"):
            if hasattr(cls, name):
                terminate_and_kill_process_tree(getattr(cls, name))

    def test_prefill_parity(self):
        # A single-chunk prompt runs no cross-rank reduction, so it is bit-exact.
        # With a prefix the chunked-prefix merge reorders the sum: 4.4e-02 nats
        # measured here, argmax unchanged.
        for prompt in _PARITY_PROMPTS:
            with self.subTest(prompt=prompt[:40]):
                base = _generate(self.non_dcp_url, prompt, 1)
                dcp = _generate(self.dcp_url, prompt, 1)
                _assert_sane(self, dcp, "dcp prefill")
                self.assertEqual(base["ids"], dcp["ids"])
                single_chunk = base["prompt_tokens"] <= CHUNKED_PREFILL_SIZE
                tol = 1e-9 if single_chunk else 0.25
                self.assertAlmostEqual(
                    base["logprobs"][0],
                    dcp["logprobs"][0],
                    delta=tol,
                    msg=(
                        f"prompt_tokens={base['prompt_tokens']} "
                        f"single_chunk={single_chunk}"
                    ),
                )

    def test_decode_parity(self):
        agree = 0
        for prompt in _PARITY_PROMPTS:
            base = _generate(self.non_dcp_url, prompt, 16)
            dcp = _generate(self.dcp_url, prompt, 16)
            with self.subTest(prompt=prompt[:40]):
                _assert_sane(self, dcp, "dcp decode")
                agree += base["ids"] == dcp["ids"]
                self.assertLessEqual(_worst_logprob_delta(base, dcp), _LOGPROB_TOL)
        self.assertGreaterEqual(
            agree / len(_PARITY_PROMPTS),
            _MIN_TOKEN_AGREEMENT,
            f"only {agree}/{len(_PARITY_PROMPTS)} continuations matched DCP=1",
        )


if __name__ == "__main__":
    unittest.main()
