"""Correctness tests for Muse Glimmer served on the SGLang MLX backend.

Muse Glimmer interleaves sliding-window (window=2048) and NoPE full-attention
layers, gates attention output through a sigmoid projection fused into
q_proj, and ships as an mlx-lm ``model_file`` artifact — so it exercises
the MLX backend's remote-code gate, container-level window discovery, and
banded-mask paths end to end. Two guards:

1. ``TestMuseGlimmerMlxServing`` — black-box serving smoke against a running
   server, including a prompt long enough to engage the sliding window.
2. ``TestMuseGlimmerMlxReferenceCorrectness`` — token-for-token equivalence of
   SGLang greedy decoding against raw, unpatched mlx_lm greedy generation
   on identical ``input_ids``. Both sides keep full KV history (the Muse Glimmer
   model file's ``make_cache`` deliberately avoids ``RotatingKVCache``),
   so exact equality holds even past the window and across prefill
   chunkings.

The artifact is weight-derived and private: its path comes exclusively from the
``SGLANG_MLX_TEST_ONYX_ARTIFACT`` environment variable and the whole module SKIPS
cleanly when it is unset or missing — CI has no dependency on private
assets. Window-engagement prompt lengths are derived from the artifact's
own config, so the suite also runs against tiny packaged fixtures.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Registered on the CPU suite as an import check; skipped wherever mlx or
# the private artifact is absent. stage-b-e2e-mlx is manual-dispatch only.
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=420, suite="stage-b-e2e-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)

ARTIFACT = os.environ.get("SGLANG_MLX_TEST_ONYX_ARTIFACT")
MEM_FRACTION_STATIC = os.environ.get("SGLANG_MLX_TEST_MEM_FRACTION", "0.85")
MIN_FREE_GB = float(os.environ.get("SGLANG_MLX_TEST_MIN_FREE_GB", "20"))
MAX_NEW_TOKENS = 32


def _artifact_or_skip() -> Path:
    if not _HAS_MLX:
        raise unittest.SkipTest("requires mlx + mlx_lm (Apple Silicon only)")
    if not ARTIFACT:
        raise unittest.SkipTest(
            "SGLANG_MLX_TEST_ONYX_ARTIFACT is not set; the Muse Glimmer artifact is private "
            "and must be supplied via the environment"
        )
    path = Path(ARTIFACT).expanduser()
    if not (path / "config.json").is_file():
        raise unittest.SkipTest(f"no packaged artifact at {path}")
    config = json.loads((path / "config.json").read_text())
    if config.get("onyx_mlx_format") != 1:
        raise unittest.SkipTest(
            f"{path} is not a packaged Muse Glimmer MLX artifact (missing marker)"
        )
    return path


def _artifact_config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


def _available_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def _check_memory(path: Path):
    # Tiny fixtures need no headroom; only guard for real-size artifacts.
    weights_gb = (
        sum(p.stat().st_size for p in path.glob("model*.safetensors")) / 1024**3
    )
    if weights_gb <= 1:
        return
    # A previous test class's just-killed server can hold memory for a few
    # seconds; wait for the release instead of skipping on a transient dip.
    import time

    deadline = time.monotonic() + 90
    avail = _available_gb()
    while avail is not None and avail < MIN_FREE_GB and time.monotonic() < deadline:
        time.sleep(5)
        avail = _available_gb()
    if avail is not None and avail < MIN_FREE_GB:
        raise unittest.SkipTest(
            f"insufficient free memory: {avail:.1f} GB < {MIN_FREE_GB} GB "
            f"needed to safely serve a {weights_gb:.0f} GB artifact"
        )


def _case_lengths(config: dict) -> list[int]:
    window = int(config.get("sliding_window", 2048))
    max_pos = int(config.get("max_position_embeddings", 16384))
    return sorted(
        {
            max(4, window - 1),
            min(window + 64, max_pos - MAX_NEW_TOKENS - 1),
            min(2 * window, max_pos - MAX_NEW_TOKENS - 1),
        }
    )


def _prompt_id_cases(config: dict) -> list[list[int]]:
    """Deterministic random input_ids around the artifact's window size.

    Serving-smoke material only: random tokens produce flat next-token
    distributions, fine for shape/length assertions but useless for exact
    greedy matching.
    """
    import numpy as np

    vocab = int(config["vocab_size"])
    hi = min(vocab, 200000)
    lo = min(1000, hi // 2)
    rng = np.random.default_rng(20260731)
    return [rng.integers(lo, hi, size=n).tolist() for n in _case_lengths(config)]


def _natural_prompt_cases(config: dict, artifact: Path) -> list[list[int]]:
    """Natural-text prompts at the same window-straddling lengths.

    Real text gives the model real top-1 margins, so exact greedy
    equivalence across different batching shapes is meaningful — a benign
    kernel-shape difference cannot flip a decisive argmax, while a masking
    or batching bug still can.
    """
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(artifact)
    filler = (
        "Day %d: we walked along the ridge, catalogued mosses and lichens, "
        "measured stream flow, and noted the weather turning. "
    )
    cases = []
    for n in _case_lengths(config):
        # Prompts end MID-NARRATIVE: continuing a strictly cyclic journal is
        # near-deterministic, so greedy continuations sit on wide top-1
        # margins (an open-ended question would put the decision point on a
        # near-tie, where benign kernel-shape noise can flip the argmax).
        text, day = "The following is a field journal. ", 1
        ids: list[int] = []
        while len(ids) < n:
            text += filler % day
            day += 1
            ids = tok.encode(text)
        cases.append(ids[:n])
    return cases


def _reference_greedy(model, prompt_ids, max_tokens, prefill_step_size=None):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    kwargs = {}
    if prefill_step_size is not None:
        # Match the server's chunked-prefill size: Metal matmuls accumulate
        # at reduced precision and are shape-dependent, so exact greedy
        # equality requires both sides to prefill in identical chunk shapes.
        kwargs["prefill_step_size"] = prefill_step_size
    out = []
    for token, _ in generate_step(
        mx.array(prompt_ids), model, max_tokens=max_tokens, **kwargs
    ):
        out.append(int(token))
        if len(out) >= max_tokens:
            break
    return out


class TestMuseGlimmerMlxServing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = _artifact_or_skip()
        _check_memory(cls.artifact)
        cls.config = _artifact_config(cls.artifact)
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"
        cls.process = popen_launch_server(
            str(cls.artifact),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--disable-radix-cache",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                MEM_FRACTION_STATIC,
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _generate(self, payload):
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()

    def test_solo_generation(self):
        ids = _prompt_id_cases(self.config)[0]
        out = self._generate(
            {
                "input_ids": ids,
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            }
        )
        self.assertEqual(len(out["output_ids"]), 16)

    def test_window_engaging_prompt(self):
        # Longest case exceeds the sliding window: prefill and decode both
        # run with banded masks engaged on the sliding layers.
        ids = _prompt_id_cases(self.config)[-1]
        self.assertGreater(len(ids), int(self.config["sliding_window"]))
        out = self._generate(
            {
                "input_ids": ids,
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            }
        )
        self.assertEqual(len(out["output_ids"]), 16)

    def test_batch_generation(self):
        cases = _prompt_id_cases(self.config)
        out = self._generate(
            {
                "input_ids": cases,
                "sampling_params": {
                    "max_new_tokens": 12,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            }
        )
        self.assertEqual(len(out), len(cases))
        for r in out:
            self.assertEqual(len(r["output_ids"]), 12)

    def test_over_context_request_rejected(self):
        # A prompt beyond the context window must produce a clear error,
        # not unbounded cache growth or a process kill.
        max_pos = int(self.config.get("max_position_embeddings", 16384))
        too_long = [1000 + (i % 1000) for i in range(max_pos + 64)]
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": too_long,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=600,
        )
        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        rejected = resp.status_code >= 400 or (
            isinstance(body, dict)
            and body.get("meta_info", {}).get("finish_reason", {}).get("type")
            == "abort"
        )
        self.assertTrue(
            rejected,
            f"over-context request was not rejected: {resp.status_code} {body}",
        )
        # The server must still be healthy afterwards.
        ok = requests.get(f"{self.base_url}/health", timeout=60)
        self.assertEqual(ok.status_code, 200)

    def test_abort_then_rerun_is_isolated(self):
        # Closing a stream mid-decode aborts the request server-side; a
        # fresh identical request afterwards must produce the same output
        # as one issued before the abort (no cache/state leakage).
        ids = _prompt_id_cases(self.config)[0]
        params = {"max_new_tokens": 24, "temperature": 0, "ignore_eos": True}

        def full_run():
            r = requests.post(
                f"{self.base_url}/generate",
                json={"input_ids": ids, "sampling_params": params},
                timeout=600,
            )
            r.raise_for_status()
            return r.json()["output_ids"]

        before = full_run()
        with requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": ids,
                "sampling_params": params,
                "stream": True,
            },
            stream=True,
            timeout=600,
        ) as resp:
            for line in resp.iter_lines():
                if line and line.startswith(b"data:") and b"[DONE]" not in line:
                    break  # first token seen -> drop the connection mid-decode
        after = full_run()
        self.assertEqual(before, after, "post-abort rerun diverged")

    def test_explicit_eom_stop_honored(self):
        # 200007 must not be a default stop, but the stop machinery must
        # still be ABLE to stop on it when a request asks — proving its
        # absence from the default set is configuration, not inability.
        # The prompt pre-opens the reasoning channel, so the only way for
        # greedy decoding to close it is <|eom|> — whether the model would
        # have chosen to reason on its own varies with server config.
        if self.config["vocab_size"] < 200008:
            self.skipTest("tiny fixture vocab has no <|eom|> token")
        out = self._generate(
            {
                "text": (
                    "<|start|>user<|message|>Hi, who are you?<|eot|>"
                    "<|start|>assistant to=self<|message|>"
                ),
                "sampling_params": {
                    "max_new_tokens": 512,
                    "temperature": 0,
                    "stop_token_ids": [200007],
                },
            }
        )
        finish = out["meta_info"]["finish_reason"]
        self.assertEqual(finish["type"], "stop")
        self.assertEqual(finish["matched"], 200007)

    def test_eom_not_terminal_for_real_artifact(self):
        # <|eom|> (200007) closes a message, not a turn; the packaged
        # generation_config must not list it as EOS, or every response
        # truncates at end-of-thinking. Only meaningful on the real vocab.
        if self.config["vocab_size"] < 200008:
            self.skipTest("tiny fixture vocab has no <|eom|> token")
        gen = json.loads((self.artifact / "generation_config.json").read_text())
        self.assertNotIn(200007, gen["eos_token_id"])
        self.assertEqual(sorted(gen["eos_token_id"]), [200001, 200008])


class TestMuseGlimmerMlxReferenceCorrectness(CustomTestCase):
    """SGLang vs raw mlx_lm greedy equivalence on identical input_ids.

    Real-weight artifacts only: random-weight fixtures produce softcapped
    logits with near-zero top-1 margins, so ANY benign kernel-shape
    difference (e.g. the wrapper's trailing-window KV truncation vs the
    reference's full-history banded mask, both mathematically equivalent)
    flips argmax ties and exact matching carries no signal.
    """

    @classmethod
    def setUpClass(cls):
        cls.artifact = _artifact_or_skip()
        _check_memory(cls.artifact)
        cls.config = _artifact_config(cls.artifact)
        if cls.config["vocab_size"] < 200008:
            raise unittest.SkipTest(
                "exact greedy equivalence needs real Muse Glimmer weights; "
                "tiny random-weight fixtures have no top-1 margin"
            )

        from mlx_lm.utils import load_model

        cls.chunk = max(256, int(cls.config.get("sliding_window", 2048)) // 2)
        ref_model, _ = load_model(cls.artifact)
        cls.cases = []
        for prompt_ids in _natural_prompt_cases(cls.config, cls.artifact):
            ref_ids = _reference_greedy(
                ref_model,
                prompt_ids,
                MAX_NEW_TOKENS,
                prefill_step_size=cls.chunk,
            )
            cls.cases.append((prompt_ids, ref_ids))
        del ref_model
        import gc

        gc.collect()

        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"
        # Chunk size below the sliding window forces the longest prompt to
        # prefill in several chunks that cross the window boundary — exact
        # equality then also covers chunked-prefill mask stitching. The
        # reference used the SAME chunk size (set in the class attribute
        # above) so both sides prefill in identical kernel shapes.
        chunk = cls.chunk
        cls.process = popen_launch_server(
            str(cls.artifact),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--disable-radix-cache",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                MEM_FRACTION_STATIC,
                "--chunked-prefill-size",
                str(chunk),
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _sglang_greedy(self, prompt_ids, max_tokens):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": prompt_ids,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["output_ids"]

    def test_solo_greedy_matches_reference(self):
        for prompt_ids, ref_ids in self.cases:
            got = self._sglang_greedy(prompt_ids, MAX_NEW_TOKENS)
            self.assertEqual(
                got,
                ref_ids,
                f"greedy divergence at prompt length {len(prompt_ids)}",
            )

    def test_concurrent_ragged_batch_matches_reference(self):
        # All cases in flight together: ragged lengths force mixed
        # prefill/decode batching; every stream must still match its solo
        # mlx_lm reference exactly (no cross-request contamination).
        with concurrent.futures.ThreadPoolExecutor(len(self.cases)) as pool:
            futures = [
                pool.submit(self._sglang_greedy, prompt_ids, MAX_NEW_TOKENS)
                for prompt_ids, _ in self.cases
            ]
            results = [f.result() for f in futures]
        for (prompt_ids, ref_ids), got in zip(self.cases, results):
            self.assertEqual(
                got,
                ref_ids,
                f"batched divergence at prompt length {len(prompt_ids)}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
