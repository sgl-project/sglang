"""Serving-level parity of the Cake KDA kernels against the Triton kernels.

The reference server runs the Triton prefill and the Triton decode; each
candidate arm swaps in a Cake kernel with everything else identical: the Cake
prefill export (BF16 or TF32) with the Triton decode, the Cake decode with the
Triton prefill (isolates the decode kernel), or both. All serve the same
deterministic natural-text prompts (a seeded shuffle of a built-in sentence
corpus) covering the shapes that route to different exported kernels: single
long prefills, one batched request of short prompts (the one-wave
multi-sequence grids), and a cached-prefix residual (a short suffix prefilled
from a non-zero recurrent state), each followed by ``max_new_tokens`` greedy
decode steps.

Each arm is compared with a Triton reference served with the same state pool
dtype (``--mamba-ssm-dtype``): the prepared Cake prefill exports serve the
FP32 pool, while the Cake decode kernel requires the BF16 pool on SM100+, so
the decode arms and their reference run with ``bfloat16``.

Prefill parity is judged on the input positions. Pooled over every input
position of every prompt, the candidate must agree with the reference on the
top-1 next-token prediction for at least ``parity_top1_agreement`` of the
positions and stay within ``parity_mean_logprob_delta`` on the mean absolute
input-token logprob difference; per prompt, every logprob must be finite and
the top-1 agreement must stay above ``parity_top1_floor`` (a broken kernel,
such as the layer-44 regression this test was written after, drops agreement
to noise level on the affected shape).

Decode parity is judged on the greedy continuation. Two servers only see the
same decode input while their greedy tokens agree, so the decode metrics cover
the output positions up to the first divergence: pooled over every prompt, the
candidate must reproduce at least ``parity_output_agreement`` of the reference
output tokens, and the mean absolute logprob difference of the agreeing output
tokens must stay within ``parity_mean_logprob_delta``. Exact per-prompt greedy
match is reported but not asserted: on Kimi-Linear-48B the *same* Triton
configuration shows one prompt of a batched request at top-1 agreement 0.95 /
max |Δ logprob| 1.8 between two launches (the prefix-cache state-tracking
path), so per-prompt exact-match rules fail on the serving stack rather than
on the kernel.

Every arm is measured on every prompt and a per-prompt table is printed
before any assertion fires. ``SGLANG_TEST_KDA_PARITY_CONTROL=1`` adds a
``triton_control`` arm (a second launch of the reference configuration) whose
row is the served stack's own noise floor; ``SGLANG_TEST_KDA_PARITY_DUMP=DIR``
writes the raw observations for offline analysis;
``SGLANG_TEST_KDA_PARITY_ARMS=a,b`` runs a subset of the arms.

Multi-node TP (e.g. Kimi-K3 TP8 on two 4-GPU GB300 nodes): set
``SGLANG_TEST_KDA_PARITY_NNODES=N``, ``SGLANG_TEST_KDA_PARITY_DIST_INIT_ADDR``
(rank-0 host:port) and ``SGLANG_TEST_KDA_PARITY_PEER_CONTROL_DIR`` (a directory
shared by all nodes). Rank 0 runs here; for every server launch the kit writes
a launch request per peer rank into the control directory and a peer agent
(``python -m sglang.test.kits.kda_cake_parity_peer``) running on each
other node starts/stops the matching ``sglang serve --node-rank r`` process.

Route telemetry (the ``SGLANG_KDA_ROUTE_EVENT`` info-log lines) is read back
from the candidate server's log per mode: a candidate arm that quietly fell
back to Triton would trivially pass the numeric checks, so an arm that serves
the Cake prefill must show Cake prefill launches and no prefill fallbacks
(unless it declares ``expect_cake_fallback``), an arm that serves the Cake
decode must show Cake decode launches and no decode fallbacks, and neither
mode may report a fatal route.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import statistics
import tempfile
import time
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
class CakeArm:
    """One candidate server configuration.

    ``extra_args`` selects the prefill and decode backends; ``cake_prefill`` /
    ``cake_decode`` state which modes must show Cake launches in the route
    telemetry.
    """

    name: str
    extra_args: tuple[str, ...] = (
        "--linear-attn-prefill-backend",
        "cake",
        "--linear-attn-decode-backend",
        "triton",
    )
    env: dict[str, str] = field(
        default_factory=lambda: {"SGLANG_KDA_CAKE_PREFILL_API": "prepared"}
    )
    # ``--mamba-ssm-dtype`` of the arm *and* of its Triton reference (one
    # reference server per distinct state dtype). The prepared Cake prefill
    # exports serve the FP32 state pool; the Cake decode kernel requires the
    # BF16 pool on SM100+ (``--linear-attn-decode-backend cake``).
    state_dtype: str = "float32"
    cake_prefill: bool = True
    cake_decode: bool = False
    # The TF32 export serves bounded gates only; a model with an unbounded
    # softplus gate runs the Triton prefill under ``tf32`` and must say so.
    expect_cake_fallback: bool = False
    # The exported Cake decode is the Kimi-K3 TP8 contract (12 heads x 128,
    # bounded gate); any other model routes every decode to Triton and must
    # say so.
    expect_cake_decode_fallback: bool = False
    # A re-launch of the reference configuration (noise floor); only checked
    # for finiteness and for the absence of Cake launches.
    control: bool = False


BF16_ARM = CakeArm(name="cake_bf16")
TF32_ARM = CakeArm(
    name="cake_tf32",
    extra_args=(
        "--linear-attn-prefill-backend",
        "cake",
        "--linear-attn-decode-backend",
        "triton",
        "--kda-cake-prefill-precision",
        "tf32",
    ),
)
# Cake decode behind the Triton prefill: the prefill outputs must match the
# reference, so every difference in the continuation is the decode kernel's.
DECODE_ARM = CakeArm(
    name="cake_decode",
    extra_args=(
        "--linear-attn-prefill-backend",
        "triton",
        "--linear-attn-decode-backend",
        "cake",
    ),
    state_dtype="bfloat16",
    cake_prefill=False,
    cake_decode=True,
)
# The production configuration: Cake prefill and Cake decode on the BF16
# pool, i.e. ``--linear-attn-backend cake --mamba-ssm-dtype bfloat16``. The
# prepared BF16 prefill export requires the FP32 pool, so this arm keeps the
# default ``SGLANG_KDA_CAKE_PREFILL_API=auto`` policy (facade prefill on a
# BF16 pool) instead of forcing ``prepared``.
BF16_DECODE_ARM = CakeArm(
    name="cake_bf16_decode",
    extra_args=(
        "--linear-attn-prefill-backend",
        "cake",
        "--linear-attn-decode-backend",
        "cake",
    ),
    env={},
    state_dtype="bfloat16",
    cake_decode=True,
)


TRITON_CONTROL_ARM = CakeArm(
    name="triton_control",
    extra_args=(
        "--linear-attn-prefill-backend",
        "triton",
        "--linear-attn-decode-backend",
        "triton",
    ),
    env={},
    cake_prefill=False,
    control=True,
)

# Deterministic natural-text source for the prompts (no dataset download).
_CORPUS = (
    "The tensor cores on a modern GPU multiply small matrix tiles in a single instruction.",
    "Shared memory bandwidth limits how fast a block can stage its operands.",
    "A compiler pass rewrites the loop nest so that every thread reads consecutive addresses.",
    "The scheduler issues one warp instruction per cycle when no dependency stalls it.",
    "Recurrent state must be carried across chunks without losing precision.",
    "Kernel launch overhead becomes visible when each launch does very little work.",
    "The profiler attributes time to kernels by correlating activity records with launches.",
    "Mixed precision keeps the accumulator wide while the inputs stay narrow.",
    "A cache flush before each measurement removes the warm-line advantage.",
    "The barrier waits until every participating thread has arrived.",
    "Asynchronous copies let the next tile arrive while the current one is consumed.",
    "Register pressure forces the compiler to spill when too many values are live.",
    "The decode step processes one new token per sequence and is bound by memory traffic.",
    "Chunked prefill splits a long prompt so that the batch never exceeds its budget.",
    "The radix cache shares a common prefix between requests that start the same way.",
    "A gated delta rule updates the state with a learned forgetting factor.",
    "Softplus gates stay negative when the lower bound clamps them.",
    "Validation compares the kernel output against a reference computed in higher precision.",
    "The benchmark reports the median of repeated runs to resist outliers.",
    "Tile shapes are chosen so that the pipeline stages fit in shared memory.",
    "Every exported kernel is keyed by the schedule specialization that produced it.",
    "The tokenizer maps text to identifiers before the model sees a single number.",
    "Numerical drift accumulates when a chunked recurrence rounds its state each step.",
    "Load balancing spreads the sequences across the streaming multiprocessors.",
    "The host waits on an event to know that the device finished the copy.",
    "Warp specialization dedicates producer warps to data movement and consumer warps to math.",
    "A regression test pins the inputs so that a change in output is a change in code.",
    "The context length bounds how many tokens the attention state can cover.",
    "Prefix caching stores the recurrent state at the end of the cached tokens.",
    "The gateway node only submits jobs; the compute nodes run them.",
)


@dataclass
class _Observation:
    input_logprobs: list[float]
    input_top1: list[int]
    output_tokens: list[int]
    output_logprobs: list[float]


class KDACakeParityMixin:
    """Mix into a ``CustomTestCase``; set ``model`` and ``tp_size``."""

    model: str
    tp_size: int
    arms: tuple[CakeArm, ...] = (BF16_ARM,)
    # Shared by the reference and every candidate arm.
    base_args: tuple[str, ...] = (
        "--trust-remote-code",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
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
    reference_args: tuple[str, ...] = (
        "--linear-attn-prefill-backend",
        "triton",
        "--linear-attn-decode-backend",
        "triton",
    )
    # Override when the checkpoint streams from a slow shared filesystem (Kimi-K3 is 1.56 TB): seconds per server launch.
    launch_timeout: float = float(
        os.environ.get(
            "SGLANG_TEST_KDA_PARITY_LAUNCH_TIMEOUT",
            3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
    )
    single_prompt_lengths: tuple[int, ...] = (512, 2048, 8192)
    batch_prompt_count: int = 8
    batch_prompt_length: int = 1024
    residual_prefix_length: int = 2048
    residual_suffix_length: int = 64
    max_new_tokens: int = 32
    parity_mean_logprob_delta: float = 0.1
    parity_top1_agreement: float = 0.97
    parity_top1_floor: float = 0.90
    # Greedy output agreement is bounded by near-tie positions in natural
    # text: on Kimi-Linear-48B a second launch of the Triton reference agreed
    # on 0.92 of the output tokens and the Cake BF16 prefill (Triton decode)
    # on 0.76 (16-token continuations), so the floor catches a broken decode
    # (which diverges within the first steps of every prompt) rather than
    # kernel-level rounding.
    parity_output_agreement: float = 0.60
    tokens_per_word: float = 1.3

    # ---- prompts -----------------------------------------------------------
    @classmethod
    def _text(cls, approx_tokens: int, seed: int) -> str:
        """Deterministic natural text of roughly ``approx_tokens`` tokens."""
        import random

        rng = random.Random(seed)
        words_needed = int(approx_tokens / cls.tokens_per_word)
        sentences, words = [], 0
        while words < words_needed:
            sentence = rng.choice(_CORPUS)
            sentences.append(sentence)
            words += len(sentence.split())
        return " ".join(sentences)

    @classmethod
    def _prompt_set(cls) -> dict[str, str]:
        prompts = {}
        for length in cls.single_prompt_lengths:
            prompts[f"single_t{length}"] = cls._text(length, seed=length)
        for i in range(cls.batch_prompt_count):
            prompts[f"batch{i}_t{cls.batch_prompt_length}"] = cls._text(
                cls.batch_prompt_length, seed=1000 + i
            )
        return prompts

    # ---- server lifecycle --------------------------------------------------
    # ---- multi-node peers ------------------------------------------------------
    @staticmethod
    def _nnodes() -> int:
        return int(os.environ.get("SGLANG_TEST_KDA_PARITY_NNODES", "1"))

    def _multi_node_args(self) -> list[str]:
        if self._nnodes() <= 1:
            return []
        addr = os.environ.get("SGLANG_TEST_KDA_PARITY_DIST_INIT_ADDR") or (
            f"{socket.gethostname()}:20000"
        )
        return [
            "--nnodes",
            str(self._nnodes()),
            "--node-rank",
            "0",
            "--dist-init-addr",
            addr,
        ]

    def _request_peers(self, server_args: list[str], env: dict, log_prefix: str) -> str:
        """Ask the peer agents to start ``sglang serve --node-rank r`` for r >= 1."""
        control = os.environ["SGLANG_TEST_KDA_PARITY_PEER_CONTROL_DIR"]
        os.makedirs(control, exist_ok=True)
        seq = f"{int(time.time())}-{log_prefix}"
        for rank in range(1, self._nnodes()):
            args = [("--node-rank" if a == "--node-rank" else a) for a in server_args]
            args[args.index("--node-rank") + 1] = str(rank)
            request = {
                "seq": seq,
                "rank": rank,
                "argv": ["sglang", "serve", *args],
                "env": env,
            }
            tmp = os.path.join(control, f".launch-{seq}-r{rank}.json.tmp")
            with open(tmp, "w") as f:
                json.dump(request, f)
            os.replace(tmp, os.path.join(control, f"launch-{seq}-r{rank}.json"))
        return seq

    def _stop_peers(self, seq: str, timeout: float = 300) -> None:
        control = os.environ["SGLANG_TEST_KDA_PARITY_PEER_CONTROL_DIR"]
        for rank in range(1, self._nnodes()):
            open(os.path.join(control, f"stop-{seq}-r{rank}"), "w").close()
        deadline = time.time() + timeout
        for rank in range(1, self._nnodes()):
            ack = os.path.join(control, f"stopped-{seq}-r{rank}")
            while not os.path.exists(ack) and time.time() < deadline:
                time.sleep(2)

    # ---- server lifecycle --------------------------------------------------
    def _launch(self, extra_args, env_overrides, log_prefix, state_dtype):
        env = os.environ.copy()
        env.update(env_overrides)
        log_dir = tempfile.mkdtemp(prefix=f"kda_parity_{log_prefix}_")
        stdout = open(os.path.join(log_dir, "stdout.log"), "w")
        stderr = open(os.path.join(log_dir, "stderr.log"), "w")
        other_args = [
            *self.base_args,
            "--mamba-ssm-dtype",
            state_dtype,
            "--tp",
            str(self.tp_size),
            *extra_args,
            *self._multi_node_args(),
        ]
        peer_seq = None
        if self._nnodes() > 1:
            peer_seq = self._request_peers(
                ["--model-path", self.model, *other_args],
                dict(env_overrides),
                log_prefix,
            )
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=self.launch_timeout,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(stdout, stderr),
        )
        process._kda_parity_peer_seq = peer_seq
        return process, log_dir, (stdout, stderr)

    def _shutdown(self, process, files):
        try:
            kill_process_tree(process.pid)
        finally:
            for f in files:
                f.flush()
                f.close()
        seq = getattr(process, "_kda_parity_peer_seq", None)
        if seq is not None:
            self._stop_peers(seq)

    # ---- requests ----------------------------------------------------------
    def _generate(self, text) -> list[_Observation]:
        """Serve one prompt (``str``) or one batched request (``list[str]``)."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 1,
            },
            timeout=1200,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload if isinstance(payload, list) else [payload]
        observations = []
        for result in results:
            meta = result["meta_info"]
            # Position 0 has no logprob; entries are [logprob, token_id, ...] and
            # the top-1 list holds [[logprob, token_id, ...]] per position.
            logprobs, top1 = [], []
            for item, top in zip(
                meta["input_token_logprobs"], meta["input_top_logprobs"]
            ):
                if item[0] is None or not top:
                    continue
                logprobs.append(item[0])
                top1.append(top[0][1])
            output_tokens = [item[1] for item in meta["output_token_logprobs"]]
            output_logprobs = [item[0] for item in meta["output_token_logprobs"]]
            self.assertEqual(len(output_tokens), self.max_new_tokens)
            observations.append(
                _Observation(logprobs, top1, output_tokens, output_logprobs)
            )
        return observations

    def _flush_cache(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/flush_cache", params={"timeout": 30}, timeout=120
        )
        response.raise_for_status()

    def _observe_all(self) -> dict[str, _Observation]:
        prompts = self._prompt_set()
        observed = {}
        for name in (n for n in prompts if n.startswith("single_")):
            observed[name] = self._generate(prompts[name])[0]
        # One batched request so both servers see the same batch composition.
        batch_names = [n for n in prompts if n.startswith("batch")]
        self._flush_cache()
        for name, obs in zip(
            batch_names, self._generate([prompts[n] for n in batch_names])
        ):
            observed[name] = obs
        # Cached-prefix residual: the prefix is prefilled first, then a short
        # suffix continues from the cached (non-zero) recurrent state.
        prefix = self._text(self.residual_prefix_length, seed=777)
        suffix = prefix + " " + self._text(self.residual_suffix_length, seed=778)
        self._flush_cache()
        self._generate(prefix)
        observed["residual_after_prefix"] = self._generate(suffix)[0]
        return observed

    # ---- route telemetry ---------------------------------------------------
    @staticmethod
    def _route_counts(log_dir) -> dict[str, dict[str, int]]:
        """Per-mode (``prefill`` / ``decode``) Cake route counts from the server log."""
        counts = {
            mode: {"cake_success": 0, "cake_fallback": 0, "fatal": 0}
            for mode in ("prefill", "decode")
        }
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
                    row = counts.get(event.get("mode"))
                    if row is None:
                        continue
                    # A one-token prefill (the ``/health_generate`` probe) is a
                    # decode-shaped request; the Cake prefill never serves it.
                    if event.get("reason") == "t1_decode_shape":
                        continue
                    if event.get("cake_success"):
                        row["cake_success"] += 1
                    elif event.get("attempted_cake") or event.get("triton_fallback"):
                        row["cake_fallback"] += 1
                    if event.get("fatal"):
                        row["fatal"] += 1
        return counts

    # ---- comparison ----------------------------------------------------------
    @staticmethod
    def _compare(ref: _Observation, obs: _Observation) -> dict:
        n = min(len(ref.input_logprobs), len(obs.input_logprobs))
        deltas = [
            abs(a - b) for a, b in zip(obs.input_logprobs[:n], ref.input_logprobs[:n])
        ]
        agree = sum(a == b for a, b in zip(obs.input_top1[:n], ref.input_top1[:n]))
        n_output = min(len(ref.output_tokens), len(obs.output_tokens))
        divergence = next(
            (
                i
                for i in range(n_output)
                if obs.output_tokens[i] != ref.output_tokens[i]
            ),
            None,
        )
        # Decode steps see the same input only while the greedy tokens agree.
        n_output_agree = n_output if divergence is None else divergence
        output_deltas = [
            abs(a - b)
            for a, b in zip(
                obs.output_logprobs[:n_output_agree],
                ref.output_logprobs[:n_output_agree],
            )
        ]
        return {
            "finite": all(
                math.isfinite(x) for x in obs.input_logprobs + obs.output_logprobs
            ),
            "same_length": len(ref.input_logprobs) == len(obs.input_logprobs),
            "n_input": n,
            "n_agree": agree,
            "sum_delta": sum(deltas),
            "top1_agreement": agree / n if n else float("nan"),
            "greedy_match": obs.output_tokens == ref.output_tokens,
            "first_divergence": divergence,
            "mean_delta": statistics.fmean(deltas) if deltas else float("nan"),
            "max_delta": max(deltas) if deltas else float("nan"),
            "n_output": n_output,
            "n_output_agree": n_output_agree,
            "output_sum_delta": sum(output_deltas),
            "output_mean_delta": (
                statistics.fmean(output_deltas) if output_deltas else float("nan")
            ),
            "output_max_delta": max(output_deltas) if output_deltas else float("nan"),
        }

    @staticmethod
    def _dump(arm_name: str, observed: dict[str, _Observation]) -> None:
        """Write the raw observations when ``SGLANG_TEST_KDA_PARITY_DUMP`` names a directory."""
        dump_dir = os.environ.get("SGLANG_TEST_KDA_PARITY_DUMP")
        if not dump_dir:
            return
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, f"{arm_name}.json"), "w") as f:
            json.dump(
                {
                    name: {
                        "input_logprobs": obs.input_logprobs,
                        "input_top1": obs.input_top1,
                        "output_tokens": obs.output_tokens,
                        "output_logprobs": obs.output_logprobs,
                    }
                    for name, obs in observed.items()
                },
                f,
            )

    @staticmethod
    def _pooled(metrics: dict[str, dict]) -> dict:
        n = sum(m["n_input"] for m in metrics.values())
        n_output = sum(m["n_output"] for m in metrics.values())
        n_output_agree = sum(m["n_output_agree"] for m in metrics.values())
        output_sum_delta = sum(m["output_sum_delta"] for m in metrics.values())
        return {
            "n_input": n,
            "top1_agreement": sum(m["n_agree"] for m in metrics.values()) / n,
            "mean_delta": sum(m["sum_delta"] for m in metrics.values()) / n,
            "max_delta": max(m["max_delta"] for m in metrics.values()),
            "n_output": n_output,
            "n_output_agree": n_output_agree,
            "output_agreement": n_output_agree / n_output if n_output else float("nan"),
            "output_mean_delta": (
                output_sum_delta / n_output_agree if n_output_agree else float("nan")
            ),
        }

    def _report(self, arm_name: str, metrics: dict[str, dict]) -> None:
        print(f"\n[kda-parity] arm={arm_name}")
        print(
            f"[kda-parity] {'prompt':<24} {'n_in':>6} {'top1':>7} {'mean|dlogp|':>12} "
            f"{'max|dlogp|':>11} {'out_agree':>9} {'dec|dlogp|':>11}"
        )
        for name, m in metrics.items():
            out_agree = f"{m['n_output_agree']}/{m['n_output']}"
            print(
                f"[kda-parity] {name:<24} {m['n_input']:>6} {m['top1_agreement']:>7.4f} "
                f"{m['mean_delta']:>12.4f} {m['max_delta']:>11.4f} "
                f"{out_agree:>9} {m['output_mean_delta']:>11.4f}"
            )
        pooled = self._pooled(metrics)
        out_agree = f"{pooled['n_output_agree']}/{pooled['n_output']}"
        print(
            f"[kda-parity] {'POOLED':<24} {pooled['n_input']:>6} "
            f"{pooled['top1_agreement']:>7.4f} {pooled['mean_delta']:>12.4f} "
            f"{pooled['max_delta']:>11.4f} {out_agree:>9} "
            f"{pooled['output_mean_delta']:>11.4f}"
        )

    def _assert_parity(self, arm: CakeArm, metrics: dict[str, dict]) -> None:
        for name, m in metrics.items():
            label = f"{arm.name}/{name}"
            self.assertTrue(m["finite"], f"{label}: non-finite logprob")
            self.assertTrue(m["same_length"], f"{label}: input length differs")
        if arm.control:
            return
        for name, m in metrics.items():
            self.assertGreaterEqual(
                m["top1_agreement"],
                self.parity_top1_floor,
                f"{arm.name}/{name}: input top-1 agreement {m['top1_agreement']:.4f}",
            )
        pooled = self._pooled(metrics)
        self.assertGreaterEqual(
            pooled["top1_agreement"],
            self.parity_top1_agreement,
            f"{arm.name}: pooled input top-1 agreement {pooled['top1_agreement']:.4f}",
        )
        self.assertLessEqual(
            pooled["mean_delta"],
            self.parity_mean_logprob_delta,
            f"{arm.name}: pooled mean |Δ input logprob| {pooled['mean_delta']:.4f}",
        )
        self.assertGreaterEqual(
            pooled["output_agreement"],
            self.parity_output_agreement,
            f"{arm.name}: pooled greedy output agreement "
            f"{pooled['n_output_agree']}/{pooled['n_output']}",
        )
        self.assertLessEqual(
            pooled["output_mean_delta"],
            self.parity_mean_logprob_delta,
            f"{arm.name}: pooled mean |Δ output logprob| "
            f"{pooled['output_mean_delta']:.4f}",
        )

    # ---- the test ----------------------------------------------------------
    def _arms(self) -> tuple[CakeArm, ...]:
        arms = tuple(self.arms)
        if os.environ.get("SGLANG_TEST_KDA_PARITY_CONTROL") == "1":
            arms = (TRITON_CONTROL_ARM,) + arms
        # Comma-separated arm names to run a subset (debugging / reruns).
        selected = os.environ.get("SGLANG_TEST_KDA_PARITY_ARMS")
        if selected:
            wanted = set(selected.split(","))
            arms = tuple(arm for arm in arms if arm.name in wanted)
            self.assertEqual(
                {arm.name for arm in arms},
                wanted,
                "unknown arm in SGLANG_TEST_KDA_PARITY_ARMS",
            )
        return arms

    def _reference(self, state_dtype: str) -> dict[str, _Observation]:
        """Serve the Triton reference with the given state pool dtype (once per dtype)."""
        cache = self.__dict__.setdefault("_references", {})
        if state_dtype not in cache:
            process, _, files = self._launch(
                self.reference_args, {}, f"reference_{state_dtype}", state_dtype
            )
            try:
                reference = self._observe_all()
            finally:
                self._shutdown(process, files)
            self.assertGreater(len(reference), 0)
            self._dump(f"reference_{state_dtype}", reference)
            cache[state_dtype] = reference
        return cache[state_dtype]

    def test_cake_kda_kernels_match_triton(self):
        results = []
        launch_failures = []
        for arm in self._arms():
            reference = self._reference(arm.state_dtype)
            # A crashed or unhealthy arm must not hide the other arms' tables.
            try:
                process, log_dir, files = self._launch(
                    arm.extra_args, arm.env, arm.name, arm.state_dtype
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[kda-parity] arm={arm.name}: server launch failed: {exc}")
                launch_failures.append((arm, exc))
                continue
            try:
                actual = self._observe_all()
            except Exception as exc:  # noqa: BLE001
                print(f"[kda-parity] arm={arm.name}: serving failed: {exc}")
                launch_failures.append((arm, exc))
                continue
            finally:
                self._shutdown(process, files)
            self._dump(arm.name, actual)
            metrics = {
                name: self._compare(reference[name], actual[name]) for name in reference
            }
            counts = self._route_counts(log_dir)
            self._report(f"{arm.name} (state pool {arm.state_dtype})", metrics)
            print(f"[kda-parity] {arm.name} route counts: {counts}")
            results.append((arm, metrics, counts))

        for arm, exc in launch_failures:
            with self.subTest(arm=arm.name):
                self.fail(f"{arm.name}: server did not serve the prompts: {exc}")
        for arm, metrics, counts in results:
            with self.subTest(arm=arm.name):
                self._assert_parity(arm, metrics)
                for mode in ("prefill", "decode"):
                    self.assertEqual(
                        counts[mode]["fatal"], 0, f"{arm.name}: fatal {mode} routes"
                    )
                self._assert_route(
                    arm.name,
                    "prefill",
                    counts["prefill"],
                    cake=arm.cake_prefill and not arm.control,
                    fallback=arm.expect_cake_fallback,
                )
                self._assert_route(
                    arm.name,
                    "decode",
                    counts["decode"],
                    cake=arm.cake_decode and not arm.control,
                    fallback=arm.expect_cake_decode_fallback,
                )

    def _assert_route(
        self,
        arm_name: str,
        mode: str,
        counts: dict[str, int],
        *,
        cake: bool,
        fallback: bool,
    ) -> None:
        if not cake:
            self.assertEqual(
                counts["cake_success"], 0, f"{arm_name}: unexpected Cake {mode} launch"
            )
            return
        if fallback:
            self.assertEqual(
                counts["cake_success"], 0, f"{arm_name}: unexpected Cake {mode} launch"
            )
            self.assertGreater(
                counts["cake_fallback"], 0, f"{arm_name}: no Triton {mode} fallback"
            )
            return
        self.assertGreater(
            counts["cake_success"], 0, f"{arm_name}: no Cake {mode} launched"
        )
        self.assertEqual(
            counts["cake_fallback"], 0, f"{arm_name}: Triton {mode} fallbacks"
        )
