"""DeepSeek-V4 uniform-FP8 trtllm-gen backend (SM100/B200 only).

Validates SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen — DSv4 decode AND sparse
varlen prefill through flashinfer's ``trtllm_batch_decode_sparse_mla_dsv4``
on a uniform 512-dim FP8-e4m3 KV cache with an FP8 query — against the
default packed-FP8 FlashMLA path:

1. A short greedy-output baseline is collected from a FlashMLA server, then
   the same prompts are replayed on a trtllm_gen server and compared
   (output-level; byte-level comparison is impossible — the KV formats
   differ).
2. Long multi-k-token prompts do the same comparison for the varlen prefill
   path (mixed lengths fired concurrently to exercise cum_seq_lens_q
   packing; a repeat run exercises the radix-cache-hit / cached-prefix
   extend, and the longest prompt exceeds --chunked-prefill-size so chunked
   prefill is exercised too).
3. Decode-correctness probes + a GSM8K sanity eval run on the trtllm_gen
   server.
4. A CUDA-graph capture/replay smoke: concurrent decode batches of varying
   size (replaying different captured decode-graph buckets) must stay
   consistent with a single-request greedy run.
"""

import concurrent.futures
import difflib
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
)

register_cuda_ci(est_time=7200, suite="nightly-8-gpu-b200", nightly=True)

# The canonical checkpoint used by every other DSv4 test (test/manual/dsv4).
# The sgl-project/DeepSeek-V4-Flash-FP8 repack does NOT load on this code
# base: it keeps wo_a / compressor / indexer / gate weights in BF16 while its
# quantization_config declares whole-model fp8, so weight loading fails with
# "Downcasting not allowed" on the fp8-created params.
DSV4_FLASH_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"

# DSV4 model load + DeepGEMM warmup exceeds the 600s default by a wide margin.
SERVER_LAUNCH_TIMEOUT = 3600

# No SGLANG_DSV4_FP4_EXPERTS override: when unset, model_config auto-detects
# the routed-expert layout from the checkpoint (V4-Flash ships MXFP4 experts;
# forcing "0" makes expert weight loading fail on shape mismatch).
DSV4_BASE_ENV = {
    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
}

SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--max-running-requests",
    "32",
    "--mem-fraction-static",
    "0.85",
    # Below the longest LONG_PROMPTS entry so at least one prompt prefills
    # in multiple chunks (cached-prefix extends through the varlen prefill).
    "--chunked-prefill-size",
    "4096",
    # V4-Flash ships MXFP4 routed experts; the auto-selected Triton MoE
    # runner cannot consume the packed layout ("Hidden size mismatch" at
    # graph capture). Same pairing as the B200 Flash cookbook recipe.
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--disable-flashinfer-autotune",
]

# Deterministic prompts for the FlashMLA-vs-trtllm_gen output comparison.
COMPARE_PROMPTS = [
    "The capital of France is",
    "In one sentence, explain why the sky is blue.",
    "List the first five prime numbers:",
    "Water boils at",
    "The author of Romeo and Juliet is",
    "Translate 'good morning' to French:",
    "2 + 2 * 3 =",
    "Photosynthesis is the process by which",
]
COMPARE_MAX_NEW_TOKENS = 64
# FP8 formats differ (packed per-block scales + BF16 rope vs uniform
# per-tensor e4m3), so greedy outputs may diverge after some tokens; require
# strong average prefix similarity rather than exact equality.
COMPARE_MIN_MEAN_SIMILARITY = 0.6

# Long multi-k-token prompts that actually exercise the trtllm-gen varlen
# prefill: real c4 indexer top-k selection needs >~2k tokens of context and
# the c128 far tier needs whole 128-token pages; the mixed lengths also
# exercise cum_seq_lens_q packing when fired concurrently, and the longest
# exceeds --chunked-prefill-size (4096) so it prefills in multiple chunks.
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
    "\n\nIn one short sentence, what kind of activity do the paragraphs "
    "above describe?"
)


def _make_long_prompt(idx: int, target_chars: int) -> str:
    sentence = _FILLER_SENTENCES[idx % len(_FILLER_SENTENCES)]
    body = ""
    n = 0
    while len(body) < target_chars:
        body += f"[Entry {idx}-{n}] " + sentence
        n += 1
    return body + _LONG_PROMPT_QUESTION


# ~4 chars/token: roughly 2.5k, 4.5k, and 7k tokens.
LONG_PROMPTS = [
    _make_long_prompt(0, 10_000),
    _make_long_prompt(1, 18_000),
    _make_long_prompt(2, 28_000),
]
LONG_MAX_NEW_TOKENS = 32
LONG_MIN_MEAN_SIMILARITY = 0.6

GSM8K_NUM_EXAMPLES = 200
GSM8K_MIN_SCORE = 0.90

_REQUEST_TIMEOUT = 600


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


def _launch(backend: str):
    env = dict(DSV4_BASE_ENV)
    env["SGLANG_DSV4_ATTN_DECODE_BACKEND"] = backend
    return popen_launch_server(
        DSV4_FLASH_MODEL_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=SERVER_LAUNCH_TIMEOUT,
        other_args=SERVER_ARGS,
        env=env,
    )


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


class TestDSV4Fp8TrtllmGenBackend(BasicDecodeCorrectnessMixin, CustomTestCase):
    """TP8 DSv4-Flash-FP8 with SGLANG_DSV4_ATTN_DECODE_BACKEND=trtllm_gen."""

    @classmethod
    def setUpClass(cls):
        if not _is_sm100():
            raise unittest.SkipTest(
                "DSv4 trtllm-gen uniform-FP8 decode requires SM100/SM103 (B200)"
            )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None

        # Phase 1: collect the packed-FP8 FlashMLA greedy baselines (short
        # decode-focused prompts + long prefill-focused prompts).
        baseline_process = _launch("flashmla")
        try:
            cls.flashmla_outputs = [
                _greedy_generate(cls.base_url, p, COMPARE_MAX_NEW_TOKENS)
                for p in COMPARE_PROMPTS
            ]
            cls.flashmla_long_outputs = [
                _greedy_generate(cls.base_url, p, LONG_MAX_NEW_TOKENS)
                for p in LONG_PROMPTS
            ]
        finally:
            kill_process_tree(baseline_process.pid)

        # Phase 2: the server under test (kept alive for all test methods).
        cls.process = _launch("trtllm_gen")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_greedy_outputs_match_flashmla(self):
        """Output-level comparison vs the packed-FP8 FlashMLA path."""
        similarities = []
        for prompt, ref_out in zip(COMPARE_PROMPTS, self.flashmla_outputs):
            out = _greedy_generate(self.base_url, prompt, COMPARE_MAX_NEW_TOKENS)
            sim = difflib.SequenceMatcher(None, ref_out, out).ratio()
            similarities.append(sim)
            print(
                f"[compare] sim={sim:.3f} prompt={prompt!r}\n"
                f"  flashmla : {ref_out!r}\n"
                f"  trtllm   : {out!r}"
            )
        mean_sim = sum(similarities) / len(similarities)
        self.assertGreater(
            mean_sim,
            COMPARE_MIN_MEAN_SIMILARITY,
            f"trtllm_gen greedy outputs diverge from flashmla: "
            f"mean similarity {mean_sim:.3f}, per-prompt {similarities}",
        )

    def test_long_prompt_prefill_matches_flashmla(self):
        """Varlen trtllm-gen prefill vs FlashMLA on multi-k-token prompts.

        The three prompts are fired concurrently (mixed extend lengths in
        one batch exercise cum_seq_lens_q packing and per-token sparse-table
        construction), then the longest is re-sent alone (radix-cache hit →
        cached-prefix extend, where seq_lens > extend len). The longest
        prompt also exceeds --chunked-prefill-size, covering chunked
        prefill. Triage order on failure: per-token index construction
        first, varlen q packing second (see
        test/manual/dsv4/TESTING_TRTLLM_GEN_FP8.md).
        """
        with concurrent.futures.ThreadPoolExecutor(len(LONG_PROMPTS)) as pool:
            outs = list(
                pool.map(
                    lambda p: _greedy_generate(self.base_url, p, LONG_MAX_NEW_TOKENS),
                    LONG_PROMPTS,
                )
            )

        similarities = []
        for i, (ref_out, out) in enumerate(zip(self.flashmla_long_outputs, outs)):
            sim = difflib.SequenceMatcher(None, ref_out, out).ratio()
            similarities.append(sim)
            print(
                f"[long-compare] sim={sim:.3f} prompt_chars={len(LONG_PROMPTS[i])}\n"
                f"  flashmla : {ref_out!r}\n"
                f"  trtllm   : {out!r}"
            )
        mean_sim = sum(similarities) / len(similarities)
        self.assertGreater(
            mean_sim,
            LONG_MIN_MEAN_SIMILARITY,
            f"trtllm_gen long-prompt (varlen prefill) outputs diverge from "
            f"flashmla: mean similarity {mean_sim:.3f}, per-prompt {similarities}",
        )

        # Cached-prefix extend: re-run the longest prompt; the radix cache
        # holds its prefix, so this prefill extends from cached tokens
        # (seq_lens total > extend tokens). Greedy output must be unchanged.
        rerun = _greedy_generate(self.base_url, LONG_PROMPTS[-1], LONG_MAX_NEW_TOKENS)
        self.assertEqual(
            outs[-1],
            rerun,
            "greedy long-prompt output changed on the cached-prefix "
            "(radix-cache hit) extend path",
        )

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
        print(f"GSM8K sanity on trtllm_gen decode: {metrics=}")
        self.assertGreater(metrics["score"], GSM8K_MIN_SCORE)

    def test_cuda_graph_capture_replay_smoke(self):
        """Exercise decode CUDA-graph replay across batch-size buckets.

        Bursts of concurrent requests at varying concurrency replay different
        captured decode-graph buckets; a repeated greedy single must remain
        identical to its first run afterwards (address-stable trtllm sparse
        buffers, no capture/replay corruption).
        """
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
