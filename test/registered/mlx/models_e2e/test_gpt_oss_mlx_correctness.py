"""Correctness tests for gpt-oss served on the SGLang MLX backend.

gpt-oss interleaves sliding-window (window=128) and full-attention layers and
uses per-head attention sinks, so it exercises the MLX backend's
sliding-window path end to end. Two guards:

1. ``TestGptOssMlxCorrectness`` — black-box serving smoke against a running
   server with the radix cache enabled (the default KV path), including a
   >128-token prompt so the sliding window actually engages and a repeated
   prompt so a radix prefix hit must reproduce the cold greedy output.
2. ``TestGptOssMlxReferenceCorrectness`` — token-for-token equivalence of
   ``MlxModelRunner`` greedy decoding against raw, unpatched mlx_lm greedy
   generation. SGLang keeps full KV and applies banded masks /
   trailing-window truncation, while vanilla mlx_lm uses RotatingKVCache
   for sliding layers — mathematically equivalent, so tokens must match
   exactly.

Both follow the structure of the qwen MoE MLX correctness tests
(PR #29440).

Prompt length matters for both: sequences up to 128 tokens never engage the
window (banded and causal masks coincide), so a short-prompt test passes even
if window handling is completely broken. Prompts here are >128 tokens. They
also stay well under 2048 tokens: past mlx_lm's prefill chunk size the
RotatingKVCache reference trims differently and exact token equality no
longer holds by construction.

MLX-gated like its siblings: registered on the CPU suite but skipped wherever
``mlx`` is absent (all current CI runners); runs for real only on Apple
Silicon. The default 20B model needs ~11 GB of weights — override with
``SGLANG_MLX_TEST_MODEL`` (e.g. a local download of
``mlx-community/gpt-oss-20b-MXFP4-Q8``).
"""

from __future__ import annotations

import gc
import importlib.util
import os
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm (Apple Silicon only)"

MODEL_PATH = envs.SGLANG_MLX_TEST_MODEL.get() or "mlx-community/gpt-oss-20b-MXFP4-Q8"
MEM_FRACTION_STATIC = str(envs.SGLANG_MLX_TEST_MEM_FRACTION.get() or 0.9)
# Skip (do NOT crash) unless this much system memory is free; an MLX Metal
# OOM is uncatchable and can reboot the machine. ~12 GB suits the default
# 20B MXFP4-Q8 repo (11 GB of weights).
MIN_FREE_GB = envs.SGLANG_MLX_TEST_MIN_FREE_GB.get() or 12.0

# Filler that pushes every prompt past 128 tokens (the gpt-oss sliding
# window) while staying far below 2048. The question at the end keeps greedy
# answers short and deterministic.
_NUMBER_LIST = "The following is a list of numbers: " + ", ".join(
    str(i) for i in range(1, 121)
)
LONG_PROMPTS = [
    _NUMBER_LIST + ". Which number comes right after 57? Answer briefly.",
    _NUMBER_LIST + ". What is the sum of the first three numbers? Answer briefly.",
]
MAX_NEW_TOKENS = 64  # equivalence horizon; analysis-channel tokens count too
BATCH_HORIZON = 24  # fixed step count for the batching-isolation test


def _available_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None  # psutil absent -> skip the pre-flight check


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestGptOssMlxCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        avail = _available_gb()
        if avail is not None and avail < MIN_FREE_GB:
            raise unittest.SkipTest(
                f"insufficient free memory: {avail:.1f} GB < {MIN_FREE_GB} GB "
                f"needed to safely serve {MODEL_PATH} "
                f"(override SGLANG_MLX_TEST_MIN_FREE_GB)"
            )

        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                # Radix cache stays enabled (the default): sliding-window
                # layers keep windowed per-request KV, the shared pool holds
                # full-attention layers, and prefix hits recompute the
                # prefix, so serving must stay correct without
                # --disable-radix-cache.
                "--trust-remote-code",
                "--tp-size",
                "1",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                MEM_FRACTION_STATIC,
                "--max-running-requests",
                "1",
                "--context-length",
                "2048",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _chat(self, messages, max_tokens=64, temperature=0):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def test_basic_generation_nonempty(self):
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Say hello briefly."},
            ],
            max_tokens=32,
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_simple_arithmetic(self):
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2+2? Reply with just the number."},
            ],
        )
        self.assertIn("4", text)

    def test_long_prompt_engages_sliding_window(self):
        # >128 prompt tokens: prefill and decode both run with the sliding
        # window engaged on half the layers. The needle sits near the end of
        # the prompt, inside the window of the final positions.
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {
                    "role": "user",
                    "content": (
                        _NUMBER_LIST + ". The secret word is BLUEBERRY. "
                        "What is the secret word? Answer briefly."
                    ),
                },
            ],
        )
        self.assertIn("BLUEBERRY", text.upper())

    def test_radix_prefix_hit_reproduces_greedy_output(self):
        # The server runs with the radix cache enabled. Sending the same
        # >128-token prompt twice makes the second request hit the cached
        # prefix; on sliding-window models the runner recomputes the prefix
        # (windowed KV keeps no pool history), and greedy output must be
        # identical to the cold request.
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": _NUMBER_LIST
                + ". Which number comes right after 41? Answer briefly.",
            },
        ]
        cold = self._chat(messages, max_tokens=48)
        hit = self._chat(messages, max_tokens=48)
        self.assertEqual(cold, hit)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestGptOssMlxReferenceCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        from mlx_lm import load

        avail = _available_gb()
        if avail is not None and avail < MIN_FREE_GB:
            raise unittest.SkipTest(
                f"insufficient free memory: {avail:.1f} GB < {MIN_FREE_GB} GB needed "
                f"to safely load {MODEL_PATH} (override SGLANG_MLX_TEST_MIN_FREE_GB)"
            )

        model_path = try_cached_model(MODEL_PATH)

        # --- Phase 1: reference tokens from UNPATCHED mlx_lm (one copy resident) ---
        try:
            ref_model, cls.tokenizer = load(
                model_path, tokenizer_config={"trust_remote_code": True}
            )
        except Exception as exc:  # not cached / offline / bad path
            raise unittest.SkipTest(f"could not load {MODEL_PATH}: {exc}")

        eos = getattr(cls.tokenizer, "eos_token_ids", None) or {
            cls.tokenizer.eos_token_id
        }
        cls.eos_ids = set(eos)

        cls.cases = []  # (prompt, prompt_ids, reference_token_ids)
        for prompt in LONG_PROMPTS:
            prompt_ids = list(
                cls.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True
                )
            )
            # The whole point of this test: the sliding window only engages
            # past 128 tokens, and the RotatingKVCache reference only stays
            # trim-free below mlx_lm's prefill chunking threshold.
            assert 128 < len(prompt_ids) <= 2048, (
                f"prompt must be >128 and <=2048 tokens to exercise the "
                f"sliding window, got {len(prompt_ids)}"
            )
            ref_ids = cls._reference_greedy(
                ref_model, cls.tokenizer, prompt_ids, MAX_NEW_TOKENS
            )
            cls.cases.append((prompt, prompt_ids, ref_ids))

        # --- Release the reference BEFORE building the runner (cap peak at 1x) ---
        del ref_model
        gc.collect()
        mx.clear_cache()
        active_gb = mx.get_active_memory() / 1024**3
        if active_gb > 2.0:
            raise unittest.SkipTest(
                f"reference model not released (active={active_gb:.1f} GB); "
                "skipping to avoid a double-resident OOM"
            )

        # --- Phase 2: SGLang runner (one copy resident) ---
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        cls.runner = MlxModelRunner(
            model_path=model_path,
            trust_remote_code=True,
            disable_radix_cache=True,  # per-request contiguous caches; no big pool
            mem_fraction_static=float(MEM_FRACTION_STATIC),
        )
        cls.runner.init_cache_pools(req_to_token_pool=None)

    @classmethod
    def tearDownClass(cls):
        runner = getattr(cls, "runner", None)
        if runner is not None:
            runner.clear()
        cls.runner = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _reference_greedy(model, tokenizer, prompt_ids, max_new):
        """Ground-truth token ids from raw, unpatched mlx_lm greedy generation."""
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.0)  # greedy / argmax
        out = []
        for resp in stream_generate(
            model, tokenizer, mx.array(prompt_ids), max_tokens=max_new, sampler=sampler
        ):
            out.append(int(resp.token))
        return out

    def _prefill(self, rid, prompt_ids):
        return int(
            self.runner.prefill(
                req_id=rid,
                new_token_ids=list(prompt_ids),
                full_token_ids=list(prompt_ids),
                prefix_slot_ids=[],
                new_slot_ids=[],
                req_pool_idx=0,
            )
        )

    def _decode(self, rids):
        return [int(t) for t in self.runner.decode_batch(rids)]

    def _sglang_greedy(self, rid, prompt_ids, max_new):
        """SGLang MLX greedy generation, stopping at EOS like the reference."""
        tok = self._prefill(rid, prompt_ids)
        out = [tok]
        while len(out) < max_new and tok not in self.eos_ids:
            tok = self._decode([rid])[0]
            out.append(tok)
        self.runner.remove_request(rid)
        return out

    def _diff_msg(self, prompt, ref, sgl):
        horizon = min(len(ref), len(sgl))
        first = next((j for j in range(horizon) if ref[j] != sgl[j]), horizon)
        return (
            f"\nprompt: {prompt[:80]!r}..."
            f"\n  first divergence @ index {first} (len ref={len(ref)} sgl={len(sgl)})"
            f"\n  ref text: {self.tokenizer.decode(ref)!r}"
            f"\n  sgl text: {self.tokenizer.decode(sgl)!r}"
        )

    # --- tests ------------------------------------------------------------

    def test_greedy_matches_reference_exact(self):
        """SGLang MLX greedy output == unpatched mlx_lm greedy output, token-for-token."""
        for i, (prompt, prompt_ids, ref) in enumerate(self.cases):
            sgl = self._sglang_greedy(f"ref-{i}", prompt_ids, MAX_NEW_TOKENS)
            self.assertEqual(sgl, ref, self._diff_msg(prompt, ref, sgl))

    def test_batched_decode_matches_solo(self):
        """A request's tokens are identical whether decoded alone or in a batch.

        Pins slot/cache isolation for the sliding-window decode path: the
        per-request trailing-window truncation and locally rebuilt padding
        mask must not let one request's state bleed into another's.
        """
        ids_list = [prompt_ids for (_, prompt_ids, _) in self.cases]

        # Solo: prefill, decode a fixed horizon, remove -- one request at a time.
        solo = []
        for i, ids in enumerate(ids_list):
            seq = [self._prefill(f"solo-{i}", ids)]
            for _ in range(BATCH_HORIZON - 1):
                seq.append(self._decode([f"solo-{i}"])[0])
            self.runner.remove_request(f"solo-{i}")
            solo.append(seq)

        # Batched: prefill all, then advance them together in one decode_batch.
        rids = [f"batch-{i}" for i in range(len(ids_list))]
        batched = [[self._prefill(rid, ids)] for rid, ids in zip(rids, ids_list)]
        for _ in range(BATCH_HORIZON - 1):
            for j, t in enumerate(self._decode(rids)):
                batched[j].append(t)
        for rid in rids:
            self.runner.remove_request(rid)

        for i, (prompt, _, _) in enumerate(self.cases):
            self.assertEqual(
                batched[i], solo[i], self._diff_msg(prompt, solo[i], batched[i])
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
