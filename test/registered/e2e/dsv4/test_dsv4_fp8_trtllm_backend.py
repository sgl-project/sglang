"""SM100/SM103 coverage for DSV4's uniform-FP8 trtllm backend.

Covers decode correctness, GSM8K accuracy, varlen and cached-prefix prefill,
chunking, and decode CUDA-graph replay. Long outputs use sanity checks because
the FlashMLA and uniform-FP8 cache formats need not be bit-reproducible.
"""

import concurrent.futures
import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-b200")

DSV4_FLASH_MODEL_PATH = try_cached_model("deepseek-ai/DeepSeek-V4-Flash")
SERVER_LAUNCH_TIMEOUT = 3600
DSV4_BASE_ENV = {
    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
}

SERVER_ARGS = [
    "--trust-remote-code",
    "--dsv4-attn-backend",
    "trtllm",
    "--tp",
    "4",
    "--max-running-requests",
    "32",
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "4096",
    # V4-Flash ships MXFP4 routed experts, and the auto-selected Triton MoE runner
    # cannot consume the packed layout. Matches the B200 Flash cookbook recipe.
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--disable-flashinfer-autotune",
]

# Mixed lengths cover c4/c128 selection and VarSeq packing; the longest prompt
# exceeds the 4096-token prefill chunk.
_FILLER_SENTENCES = [
    "The expedition recorded water temperature, salinity, and current speed "
    "at every station along the transect. ",
    "Archival records from the observatory describe decades of nightly "
    "measurements taken with remarkable consistency. ",
    "Each greenhouse module recycles condensate through a gravel bed before "
    "returning it to the irrigation loop. ",
    "The survey team catalogued the masonry of the aqueduct arch by arch, "
    "noting repairs from three distinct centuries. ",
]
_LONG_PROMPT_QUESTION = (
    "\n\nIn one short sentence, what kind of activity do the paragraphs above describe?"
)


def _make_long_prompt(idx: int, target_chars: int) -> str:
    sentence = _FILLER_SENTENCES[idx % len(_FILLER_SENTENCES)]
    body = ""
    n = 0
    while len(body) < target_chars:
        body += f"[Entry {idx}-{n}] " + sentence
        n += 1
    return body + _LONG_PROMPT_QUESTION


# Roughly 2.5k, 4.5k, and 7k tokens.
LONG_PROMPTS = [
    _make_long_prompt(0, 10_000),
    _make_long_prompt(1, 18_000),
    _make_long_prompt(2, 28_000),
]
LONG_MAX_NEW_TOKENS = 32
MIN_PRINTABLE_ASCII_RATIO = 0.85

GSM8K_NUM_EXAMPLES = 200
GSM8K_MIN_SCORE = 0.90

_REQUEST_TIMEOUT = 600


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


def _greedy_generate(base_url: str, prompt: str, max_new_tokens: int) -> str:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["text"]


def _printable_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(32 <= ord(c) < 127 or c in "\n\t" for c in text) / len(text)


class TestDSV4Fp8TrtllmBackend(BasicDecodeCorrectnessMixin, CustomTestCase):
    """TP4 DSv4-Flash-FP8 with --dsv4-attn-backend trtllm."""

    @classmethod
    def setUpClass(cls):
        if not _is_sm100():
            raise unittest.SkipTest(
                "DSv4 trtllm uniform-FP8 attention requires SM100/SM103 (Blackwell)"
            )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DSV4_FLASH_MODEL_PATH,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=SERVER_ARGS,
            env=dict(DSV4_BASE_ENV),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _assert_sane(self, out: str, what: str) -> None:
        self.assertGreater(len(out.strip()), 0, f"{what}: empty output")
        ratio = _printable_ascii_ratio(out)
        self.assertGreater(
            ratio,
            MIN_PRINTABLE_ASCII_RATIO,
            f"{what}: output looks like gibberish (ascii ratio={ratio:.2f}): {out!r}",
        )

    def test_long_prompt_varlen_prefill(self):
        """Exercise mixed-length VarSeq and cached-prefix chunked prefill.

        Sanity checks avoid flaky exact matches from split-KV reduction order.
        """

        with concurrent.futures.ThreadPoolExecutor(len(LONG_PROMPTS)) as pool:
            outs = list(
                pool.map(
                    lambda p: _greedy_generate(self.base_url, p, LONG_MAX_NEW_TOKENS),
                    LONG_PROMPTS,
                )
            )
        for i, out in enumerate(outs):
            print(f"[long-prefill] prompt_chars={len(LONG_PROMPTS[i])} out={out!r}")
            self._assert_sane(out, f"concurrent long prompt {i}")

        cached = _greedy_generate(self.base_url, LONG_PROMPTS[-1], LONG_MAX_NEW_TOKENS)
        print(f"[long-prefill] cached-prefix rerun out={cached!r}")
        self._assert_sane(cached, "cached-prefix extend")

    def test_gsm8k_sanity(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=DSV4_FLASH_MODEL_PATH,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=GSM8K_NUM_EXAMPLES,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"GSM8K sanity on trtllm decode: {metrics=}")
        self.assertGreater(metrics["score"], GSM8K_MIN_SCORE)

    def test_cuda_graph_capture_replay_smoke(self):
        """Replay several decode graph buckets and recheck a greedy anchor."""
        anchor_prompt = "Q: What is the capital of France?\nA:"
        anchor_out = _greedy_generate(self.base_url, anchor_prompt, 32)

        for concurrency in (2, 4, 8, 16):
            prompts = [f"Count from {i} to {i + 5}: " for i in range(concurrency)]
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                outs = list(
                    pool.map(lambda p: _greedy_generate(self.base_url, p, 32), prompts)
                )
            self.assertEqual(len(outs), concurrency)
            for out in outs:
                self.assertGreater(len(out), 0)

        anchor_out_replayed = _greedy_generate(self.base_url, anchor_prompt, 32)
        self.assertEqual(
            anchor_out,
            anchor_out_replayed,
            "greedy output changed after batched decode-graph replays",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
