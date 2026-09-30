"""Serving-level parity of the Cake KDA prefill against the Triton prefill.

The reference server runs ``--linear-attn-prefill-backend triton``; each
candidate server runs the Cake prefill export (BF16 or TF32) with everything
else identical. Both serve the same prompts, and the candidate must reproduce
the reference's greedy continuation token for token, keep every logprob finite
and stay within ``parity_mean_logprob_delta`` on the mean absolute input-token
logprob difference. The prompt set covers the shapes that route to different
exported kernels: single long prefills, a concurrent batch of short prompts
(the one-wave multi-sequence grids), and a cached-prefix residual (a short
suffix prefilled from a non-zero recurrent state).

Route telemetry (the ``SGLANG_KDA_ROUTE_EVENT`` info-log lines) is read back
from the candidate server's log: a candidate arm that quietly fell back to Triton would
trivially pass the numeric checks, so the arm must show Cake launches and no
fallbacks unless it declares ``expect_cake_fallback``.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

_ROUTE_EVENT = re.compile(r"SGLANG_KDA_ROUTE_EVENT (\{.*\})")


@dataclass(frozen=True)
class CakePrefillArm:
    """One candidate server configuration."""

    name: str
    extra_args: tuple[str, ...] = ("--linear-attn-prefill-backend", "cake")
    env: dict[str, str] = field(
        default_factory=lambda: {"SGLANG_KDA_CAKE_PREFILL_API": "prepared"}
    )
    # The TF32 export serves bounded gates only; a model with an unbounded
    # softplus gate runs the Triton prefill under ``tf32`` and must say so.
    expect_cake_fallback: bool = False


BF16_ARM = CakePrefillArm(name="cake_bf16")
TF32_ARM = CakePrefillArm(
    name="cake_tf32",
    extra_args=(
        "--linear-attn-prefill-backend",
        "cake",
        "--kda-cake-prefill-precision",
        "tf32",
    ),
)


@dataclass
class _Observation:
    input_logprobs: list[float]
    output_tokens: list[int]


class KDACakePrefillParityMixin:
    """Mix into a ``CustomTestCase``; set ``model`` and ``tp_size``."""

    model: str
    tp_size: int
    arms: tuple[CakePrefillArm, ...] = (BF16_ARM,)
    # Shared by the reference and every candidate arm.
    base_args: tuple[str, ...] = (
        "--trust-remote-code",
        "--mamba-ssm-dtype",
        "float32",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--linear-attn-decode-backend",
        "triton",
        "--mem-fraction-static",
        "0.80",
        "--context-length",
        "32768",
        "--chunked-prefill-size",
        "16384",
        "--max-running-requests",
        "64",
        "--random-seed",
        "1",
    )
    reference_args: tuple[str, ...] = ("--linear-attn-prefill-backend", "triton")
    launch_timeout: float = 3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    single_prompt_lengths: tuple[int, ...] = (512, 2048, 8192)
    batch_prompt_count: int = 8
    batch_prompt_length: int = 1024
    residual_prefix_length: int = 2048
    residual_suffix_length: int = 64
    max_new_tokens: int = 16
    parity_mean_logprob_delta: float = 0.05

    # ---- prompts -----------------------------------------------------------
    @classmethod
    def _token_stream(cls, length: int, seed: int) -> list[int]:
        """Deterministic pseudo-text token ids (no tokenizer download)."""
        ids = [1]
        state = 12345 + seed
        while len(ids) < length:
            state = (1103515245 * state + 12345) % (1 << 31)
            ids.append(100 + state % 20000)
        return ids[:length]

    @classmethod
    def _prompt_set(cls) -> dict[str, list[int]]:
        prompts = {}
        for length in cls.single_prompt_lengths:
            prompts[f"single_t{length}"] = cls._token_stream(length, seed=length)
        for i in range(cls.batch_prompt_count):
            prompts[f"batch{i}_t{cls.batch_prompt_length}"] = cls._token_stream(
                cls.batch_prompt_length, seed=1000 + i
            )
        return prompts

    # ---- server lifecycle --------------------------------------------------
    def _launch(self, extra_args, env_overrides, log_prefix):
        env = os.environ.copy()
        env.update(env_overrides)
        log_dir = tempfile.mkdtemp(prefix=f"kda_parity_{log_prefix}_")
        stdout = open(os.path.join(log_dir, "stdout.log"), "w")
        stderr = open(os.path.join(log_dir, "stderr.log"), "w")
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=self.launch_timeout,
            other_args=[*self.base_args, "--tp", str(self.tp_size), *extra_args],
            env=env,
            return_stdout_stderr=(stdout, stderr),
        )
        return process, log_dir, (stdout, stderr)

    @staticmethod
    def _shutdown(process, files):
        try:
            kill_process_tree(process.pid)
        finally:
            for f in files:
                f.flush()
                f.close()

    # ---- requests ----------------------------------------------------------
    def _generate(self, input_ids) -> _Observation:
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=600,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        # Token 0 has no logprob; entries are [logprob, token_id, ...].
        input_logprobs = [
            item[0] for item in meta["input_token_logprobs"] if item[0] is not None
        ]
        output_tokens = [item[1] for item in meta["output_token_logprobs"]]
        self.assertEqual(len(output_tokens), self.max_new_tokens)
        return _Observation(input_logprobs, output_tokens)

    def _flush_cache(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/flush_cache", params={"timeout": 30}, timeout=120
        )
        response.raise_for_status()

    def _observe_all(self) -> dict[str, _Observation]:
        prompts = self._prompt_set()
        observed = {}
        for name in (n for n in prompts if n.startswith("single_")):
            observed[name] = self._generate(prompts[name])
        batch_names = [n for n in prompts if n.startswith("batch")]
        with ThreadPoolExecutor(max_workers=len(batch_names)) as pool:
            for name, obs in zip(
                batch_names, pool.map(lambda n: self._generate(prompts[n]), batch_names)
            ):
                observed[name] = obs
        # Cached-prefix residual: the prefix is prefilled first, then a short
        # suffix continues from the cached (non-zero) recurrent state.
        prefix = self._token_stream(self.residual_prefix_length, seed=777)
        suffix = self._token_stream(
            self.residual_prefix_length + self.residual_suffix_length, seed=777
        )
        self._flush_cache()
        self._generate(prefix)
        observed["residual_after_prefix"] = self._generate(suffix)
        return observed

    # ---- route telemetry ---------------------------------------------------
    @staticmethod
    def _route_counts(log_dir) -> dict[str, int]:
        counts = {"cake_success": 0, "cake_fallback": 0, "fatal": 0}
        for name in ("stdout.log", "stderr.log"):
            with open(os.path.join(log_dir, name), errors="replace") as f:
                for line in f:
                    m = _ROUTE_EVENT.search(line)
                    if not m:
                        continue
                    try:
                        event = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        continue
                    if event.get("mode") != "prefill":
                        continue
                    if event.get("cake_success"):
                        counts["cake_success"] += 1
                    elif event.get("attempted_cake") or event.get("triton_fallback"):
                        counts["cake_fallback"] += 1
                    if event.get("fatal"):
                        counts["fatal"] += 1
        return counts

    # ---- the test ----------------------------------------------------------
    def _assert_parity(self, reference, actual, arm_name):
        for name, ref in reference.items():
            obs = actual[name]
            label = f"{arm_name}/{name}"
            self.assertTrue(
                all(math.isfinite(x) for x in obs.input_logprobs),
                f"{label}: non-finite input logprob",
            )
            self.assertEqual(
                obs.output_tokens, ref.output_tokens, f"{label}: greedy continuation"
            )
            self.assertEqual(len(obs.input_logprobs), len(ref.input_logprobs), label)
            mean_delta = statistics.fmean(
                abs(a - b) for a, b in zip(obs.input_logprobs, ref.input_logprobs)
            )
            self.assertLessEqual(
                mean_delta,
                self.parity_mean_logprob_delta,
                f"{label}: mean |Δ input logprob| {mean_delta:.4f}",
            )

    def test_cake_prefill_matches_triton_prefill(self):
        process, log_dir, files = self._launch(self.reference_args, {}, "reference")
        try:
            reference = self._observe_all()
        finally:
            self._shutdown(process, files)
        self.assertGreater(len(reference), 0)

        for arm in self.arms:
            with self.subTest(arm=arm.name):
                process, log_dir, files = self._launch(
                    arm.extra_args, arm.env, arm.name
                )
                try:
                    actual = self._observe_all()
                finally:
                    self._shutdown(process, files)
                self._assert_parity(reference, actual, arm.name)
                counts = self._route_counts(log_dir)
                self.assertEqual(counts["fatal"], 0, f"{arm.name}: fatal route events")
                if arm.expect_cake_fallback:
                    self.assertEqual(
                        counts["cake_success"], 0, f"{arm.name}: unexpected Cake launch"
                    )
                    self.assertGreater(counts["cake_fallback"], 0, arm.name)
                else:
                    self.assertGreater(
                        counts["cake_success"],
                        0,
                        f"{arm.name}: no Cake prefill launched",
                    )
                    self.assertEqual(
                        counts["cake_fallback"], 0, f"{arm.name}: Triton fallbacks"
                    )
