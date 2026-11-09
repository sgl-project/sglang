# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Validate a serving command's CUDA graphs: lifetime, numerics, coverage.

Run the same traffic against an eager reference, the ordinary serving command,
and an instrumented launch with the lifetime tracer injected through PYTHONPATH.
The sweep varies batch sizes, row order, prompt lengths and configured adapters.

Report three verdicts separately: allocator-backed pointer lifetimes, token-aligned
logprob agreement, and replay of the required graph paths on every expected worker.
``--repeat-reference`` adds an eager batch-sensitivity control; ``--recompute``
re-judges saved evidence without launching a server.

PASS covers only the recorded configuration and exercised traffic. Library-owned
memory, device-side pointer tables and stale-but-live metadata remain outside the
lifetime check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LOGPROB_THRESHOLD = 1.0  # nats; the bound of the LoRA prefill-graph tests
MIN_DECODE_POSITIONS = 8
TOP_LOGPROBS = 5
# A greedy token's own logprob and its entry in the same position's top list
# come from one tensor; a larger gap means the response is inconsistent.
SCORE_TOLERANCE = 1e-3
_INJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inject")

_BASE_PROMPTS = [
    "The capital of France is",
    "Write a short poem about the sea.",
    "Explain the difference between a process and a thread in one paragraph.",
    "List three prime numbers larger than one hundred and explain how you checked them.",
    "In a distant future, humanity discovers",
    "Summarize the rules of chess for a beginner, then describe one famous opening.",
]
# Word budgets of the long prompts (about 1.3 tokens per word): a prefill batch
# replays a prefill graph only when its padded bucket is at most twice its
# token count, so with the usual 512-token first bucket a single request needs
# a few hundred tokens.
_LONG_PROMPT_WORDS = (260, 560)


# -- the serving command ---------------------------------------------------------


@dataclass
class ServingCommand:
    argv: List[str]
    host: str = "127.0.0.1"
    port: int = 30000
    lora_names: List[str] = field(default_factory=list)
    expected_workers: int = 1  # an estimate from the command line; see topology()

    @classmethod
    def parse(
        cls, command: str, lora_names: Optional[List[str]] = None
    ) -> ServingCommand:
        argv = shlex.split(command)
        cmd = cls(argv)
        opts = _options(argv)
        cmd.host = opts.get("--host", cmd.host)
        cmd.port = int(opts.get("--port", cmd.port))
        if lora_names is not None:
            cmd.lora_names = list(lora_names)
        else:
            cmd.lora_names = [
                value.split("=", 1)[0] for value in _option_values(argv, "--lora-paths")
            ]
        tp = _int_option(opts, "--tp-size", "--tensor-parallel-size", "--tp")
        pp = _int_option(opts, "--pp-size", "--pipeline-parallel-size", "--pp")
        dp = _int_option(opts, "--dp-size", "--data-parallel-size", "--dp")
        cmd.expected_workers = tp * pp * dp
        return cmd


def _options(argv: List[str]) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for i, arg in enumerate(argv):
        if arg.startswith("--") and "=" in arg:
            key, value = arg.split("=", 1)
            opts[key] = value
        elif (
            arg.startswith("--")
            and i + 1 < len(argv)
            and not argv[i + 1].startswith("--")
        ):
            opts[arg] = argv[i + 1]
    return opts


def _int_option(opts: Dict[str, str], *flags: str, default: int = 1) -> int:
    """Read the first present spelling, including argparse abbreviations."""
    for flag in flags:
        if flag in opts:
            return int(opts[flag])
    return default


def _option_values(argv: List[str], flag: str) -> List[str]:
    values: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                values.append(argv[i])
                i += 1
            continue
        if argv[i].startswith(flag + "="):
            values.append(argv[i].split("=", 1)[1])
        i += 1
    return values


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# -- server lifecycle ----------------------------------------------------------------


class Server:
    def __init__(
        self, cmd: ServingCommand, extra: List[str], env: Dict[str, str], log_path: str
    ):
        self.cmd = cmd
        self.argv = [*cmd.argv, *extra]
        self.env = env
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None

    def start(self, timeout_s: float) -> None:
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(
            self.argv,
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + timeout_s
        url = f"http://{self.cmd.host}:{self.cmd.port}/health"
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.process.returncode} before it was ready; "
                    f"see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError(
            f"server not healthy within {timeout_s:.0f}s; see {self.log_path}"
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                # The launcher alone: it drains and stops its own children
                # (a group-wide SIGTERM makes it treat them as crashes).
                self.process.send_signal(signal.SIGTERM)
                self.process.wait(timeout=90)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)  # stragglers
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self.log.close()

    def generate(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"http://{self.cmd.host}:{self.cmd.port}/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())


# -- traffic -------------------------------------------------------------------------


@dataclass
class Request:
    rid: str
    text: str
    lora_path: Optional[str]
    group: str


def build_sweep(
    batch_sizes: List[int], lora_names: List[str], rounds: int = 2
) -> List[List[Request]]:
    """Groups of requests sent concurrently. Larger batches first, then smaller
    ones, each group repeated with permuted rows; adapter-free, adapter and
    mixed groups when adapters exist."""
    prompts = []
    for i, base in enumerate(_BASE_PROMPTS):
        prompts.append(base)
        prompts.append(
            " ".join([base] * (2 + i % 3))
        )  # longer prefixes, different KV lengths
    long_prompts = []
    for i, words in enumerate(_LONG_PROMPT_WORDS):
        text = []
        j = i
        while len(" ".join(text).split()) < words:
            text.append(_BASE_PROMPTS[j % len(_BASE_PROMPTS)])
            j += 1
        long_prompts.append(" ".join(text))
    kinds = [None]
    if lora_names:
        kinds = ["none", "adapter", "mixed"]
    groups: List[List[Request]] = []
    counter = 0
    for size in sorted(set(batch_sizes), reverse=True):
        for kind in kinds:
            # The last round sends the long prompts (prefill-graph buckets).
            for round_index in range(rounds + 1):
                requests = []
                for row in range(size):
                    if round_index == rounds:
                        prompt = long_prompts[(counter + row) % len(long_prompts)]
                    else:
                        prompt = prompts[(counter + row) % len(prompts)]
                    lora = None
                    if kind == "adapter":
                        lora = lora_names[row % len(lora_names)]
                    elif kind == "mixed":
                        lora = (
                            None if row % 3 == 1 else lora_names[row % len(lora_names)]
                        )
                    requests.append(
                        Request(
                            f"bs{size}-{kind}-r{round_index}-{row}",
                            prompt,
                            lora,
                            f"bs{size}-{kind}",
                        )
                    )
                if round_index % 2 == 1:
                    requests.reverse()  # same rows, permuted order
                groups.append(requests)
                counter += size
    return groups


def regroup_alone(groups: List[List[Request]]) -> List[List[Request]]:
    """The same requests, one per group: a differently batched eager launch
    measures how far the eager path itself moves when batch composition (and so
    every kernel's row count) changes."""
    return [[request] for group in groups for request in group]


def run_sweep(
    server: Server, groups: List[List[Request]], max_new_tokens: int
) -> Dict[str, dict]:
    """One batched request per group: the rows reach the scheduler together, so
    every launch prefills and decodes the same batches (batch composition
    alone moves logprobs by more than the tolerance on MoE models)."""
    results: Dict[str, dict] = {}
    for group in groups:
        payload = {
            "text": [request.text for request in group],
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,  # every request yields max_new_tokens positions
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": TOP_LOGPROBS,
        }
        if any(request.lora_path is not None for request in group):
            payload["lora_path"] = [request.lora_path for request in group]
        outputs = server.generate(payload)
        if not isinstance(outputs, list) or len(outputs) != len(group):
            raise RuntimeError(
                f"group {group[0].group}: expected {len(group)} outputs, got "
                f"{outputs if not isinstance(outputs, list) else len(outputs)}"
            )
        for request, output in zip(group, outputs):
            results[request.rid] = output
    return results


# -- comparison (per-position decode logic of the LoRA prefill-graph kit) ------------


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_output(rid: str, output: Any, max_new_tokens: int) -> List[str]:
    """Validate complete, finite, internally consistent /generate evidence."""
    meta = output.get("meta_info") if isinstance(output, dict) else None
    if not isinstance(meta, dict):
        return [f"{rid}: response has no meta_info"]
    problems: List[str] = []
    prompt = meta.get("input_token_logprobs")
    if not isinstance(prompt, list) or len(prompt) < 2:
        problems.append(f"{rid}: prompt logprobs missing or shorter than two tokens")
    else:
        for position, entry in enumerate(prompt):
            if not (
                isinstance(entry, (list, tuple))
                and len(entry) >= 2
                and isinstance(entry[1], int)
            ):
                problems.append(f"{rid}: prompt position {position}: malformed entry")
                break
            if position > 0 and not _finite(entry[0]):
                problems.append(
                    f"{rid}: prompt position {position}: logprob {entry[0]!r} is not finite"
                )
                break
    chosen, tops = meta.get("output_token_logprobs"), meta.get("output_top_logprobs")
    if not isinstance(chosen, list) or len(chosen) != max_new_tokens:
        problems.append(
            f"{rid}: {len(chosen) if isinstance(chosen, list) else 'no'} generated "
            f"positions, expected {max_new_tokens} (ignore_eos)"
        )
        return problems
    if not isinstance(tops, list) or len(tops) != len(chosen):
        problems.append(
            f"{rid}: {len(tops) if isinstance(tops, list) else 'no'} top-logprob "
            f"lists for {len(chosen)} generated positions"
        )
        return problems
    for position, (entry, top) in enumerate(zip(chosen, tops)):
        if not (
            isinstance(entry, (list, tuple))
            and len(entry) >= 2
            and _finite(entry[0])
            and isinstance(entry[1], int)
        ):
            problems.append(f"{rid} position {position}: malformed generated entry")
            break
        if not (isinstance(top, list) and len(top) == TOP_LOGPROBS):
            problems.append(
                f"{rid} position {position}: {len(top) if isinstance(top, list) else 'no'} "
                f"top logprobs, expected {TOP_LOGPROBS}"
            )
            break
        if not all(
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and _finite(item[0])
            and isinstance(item[1], int)
            for item in top
        ):
            problems.append(f"{rid} position {position}: malformed top-logprob entry")
            break
        scores = {item[1]: item[0] for item in top}
        if len(scores) != len(top):
            problems.append(
                f"{rid} position {position}: repeated token ids in the top-{TOP_LOGPROBS}"
            )
            break
        if entry[1] not in scores:
            problems.append(
                f"{rid} position {position}: greedy token {entry[1]} is not in its own top-{TOP_LOGPROBS}"
            )
            break
        if abs(scores[entry[1]] - entry[0]) > SCORE_TOLERANCE:
            problems.append(
                f"{rid} position {position}: greedy token {entry[1]} scores {entry[0]:.3f} "
                f"but {scores[entry[1]]:.3f} in its own top-{TOP_LOGPROBS}"
            )
            break
    return problems


def _scores_per_position(output: dict) -> List[Tuple[int, Dict[int, float]]]:
    meta = output["meta_info"]
    chosen, tops = meta["output_token_logprobs"], meta["output_top_logprobs"]
    return [
        (token, {top_token: lp for lp, top_token, *_ in top})
        for (_, token, *_), top in zip(chosen, tops)
    ]


def compare(
    reference: Dict[str, dict],
    candidate: Dict[str, dict],
    label: str,
    max_new_tokens: int,
) -> dict:
    """Compare validated prompt scores and decode scores under matching prefixes.

    Stop at token divergence. A token missing from the other top list provides
    only a lower bound: FAIL if it exceeds the tolerance, otherwise insufficient
    evidence. Malformed responses fail; too few comparable positions cannot pass.
    """
    structural: List[str] = []
    drift_problems: List[str] = []
    evidence: List[str] = []
    expected, got = set(reference), set(candidate)
    for rid in sorted(expected - got):
        structural.append(f"{rid}: missing in {label}")
    for rid in sorted(got - expected):
        structural.append(f"{rid}: unexpected in {label}")
    compared_positions = 0
    uncompared: List[str] = []
    group_positions: Dict[str, int] = {}
    request_drift: Dict[str, Dict[str, Any]] = {}
    failing: Dict[str, Dict[str, bool]] = {}
    for rid in sorted(expected & got):
        group = rid.rsplit("-r", 1)[0]
        group_positions.setdefault(group, 0)
        ref, cand = reference[rid], candidate[rid]
        bad = _validate_output(rid, ref, max_new_tokens) + _validate_output(
            f"{rid} ({label})", cand, max_new_tokens
        )
        if bad:
            structural.extend(bad)
            continue
        ref_prompt = ref["meta_info"]["input_token_logprobs"]
        cand_prompt = cand["meta_info"]["input_token_logprobs"]
        if [t for _, t, *_ in ref_prompt] != [t for _, t, *_ in cand_prompt]:
            structural.append(f"{rid}: prompt tokens differ between the runs")
            continue
        prompt_drift = max(
            (abs(a[0] - b[0]) for a, b in zip(ref_prompt[1:], cand_prompt[1:])),
            default=0.0,
        )
        if prompt_drift >= LOGPROB_THRESHOLD:
            drift_problems.append(
                f"{rid}: prompt logprob drift {prompt_drift:.3f} nats"
            )
            failing.setdefault(rid, {})["prompt"] = True
        decode_drift = 0.0
        row = 0
        unbounded = 0
        for position, ((r_token, r_top), (c_token, c_top)) in enumerate(
            zip(_scores_per_position(ref), _scores_per_position(cand))
        ):
            measured = True
            for token in sorted({r_token, c_token}):
                if token in r_top and token in c_top:
                    diff = abs(r_top[token] - c_top[token])
                    decode_drift = max(decode_drift, diff)
                    if diff >= LOGPROB_THRESHOLD:
                        drift_problems.append(
                            f"{rid} position {position}: token {token} logprob "
                            f"{r_top[token]:.3f} vs {c_top[token]:.3f}"
                        )
                        failing.setdefault(rid, {})["decode"] = True
                    continue
                present, absent = (r_top, c_top) if token in r_top else (c_top, r_top)
                lower = max(0.0, present[token] - min(absent.values()))
                measured = False
                unbounded += 1
                decode_drift = max(decode_drift, lower)
                if lower >= LOGPROB_THRESHOLD:
                    drift_problems.append(
                        f"{rid} position {position}: token {token} logprob differs by at "
                        f"least {lower:.3f} nats (absent from the other run's top-{TOP_LOGPROBS})"
                    )
                    failing.setdefault(rid, {})["decode"] = True
                else:
                    evidence.append(
                        f"{rid} position {position}: token {token} is absent from the other "
                        f"run's top-{TOP_LOGPROBS}; the difference is at least {lower:.3f} "
                        "nats and not measurable"
                    )
            row += position > 0 and measured
            if r_token != c_token:
                break  # later positions condition on different prefixes
        if row == 0:
            uncompared.append(rid)
        compared_positions += row
        group_positions[group] += row
        request_drift[rid] = {
            "prompt": round(prompt_drift, 3),
            "decode": round(decode_drift, 3),
            "decode_positions": row,
            "decode_unbounded": unbounded,
        }
    for rid in uncompared:
        evidence.append(
            f"{rid}: no decode position comparable (the first generated token differs "
            "or its logprobs were not measurable)"
        )
    for group, positions in sorted(group_positions.items()):
        if positions == 0 and not any(p.startswith(group + "-r") for p in structural):
            evidence.append(f"group {group}: no decode position compared")
    if compared_positions < MIN_DECODE_POSITIONS:
        evidence.append(
            f"only {compared_positions} decode positions compared (minimum {MIN_DECODE_POSITIONS})"
        )
    if structural:
        verdict, kind = "FAIL", "malformed"
    elif drift_problems:
        verdict, kind = "FAIL", "drift"
    elif evidence:
        verdict, kind = "INCONCLUSIVE", "insufficient-evidence"
    else:
        verdict, kind = "PASS", "drift"
    return {
        "verdict": verdict,
        "kind": kind,
        "problems": (structural + drift_problems + evidence)[:50],
        "structural_problems": structural[:50],
        "drift_problems": drift_problems[:50],
        "evidence_problems": evidence[:50],
        "compared_decode_positions": compared_positions,
        "decode_positions_by_group": group_positions,
        "requests_without_decode_positions": uncompared,
        "max_drift": max(
            (max(d["prompt"], d["decode"]) for d in request_drift.values()),
            default=0.0,
        ),
        "request_drift": request_drift,
        "failing_requests": failing,
    }


def _measured_decode(drift: Optional[dict]) -> bool:
    """At least one measured decode position, with no lower-bound-only token."""
    return bool(
        drift
        and drift.get("decode_positions", 0) > 0
        and not drift.get("decode_unbounded")
    )


def _apply_batch_sensitivity_control(result: dict, control: dict) -> None:
    """Explain drift only within twice the same request/kind's eager control.

    Decode evidence must be measured on both sides. Only a valid drift FAIL may
    become INCONCLUSIVE; this batch-sensitivity heuristic never proves correctness.
    """
    if result.get("kind") != "drift" or result.get("verdict") != "FAIL":
        return
    if control.get("kind") == "malformed":
        result["problems"].insert(
            0, "batch-sensitivity control is malformed; the failure stands"
        )
        return
    explained: List[str] = []
    unexplained: List[str] = []
    for rid, kinds in sorted(result.get("failing_requests", {}).items()):
        control_drift = control.get("request_drift", {}).get(rid)
        candidate_drift = result.get("request_drift", {}).get(rid, {})
        for kind in sorted(kinds):
            c = control_drift.get(kind, 0.0) if control_drift else 0.0
            d = candidate_drift.get(kind, 0.0)
            note = f"{rid} {kind} {d:.3f} nats (control {c:.3f})"
            if kind == "decode" and not _measured_decode(candidate_drift):
                unexplained.append(
                    note + " is not a measured difference (a lower bound from a token "
                    "absent from a top list, or no comparable decode position)"
                )
            elif kind == "decode" and not _measured_decode(control_drift):
                unexplained.append(
                    note + ": the control's decode evidence for this request is not "
                    "measured (a lower bound or no comparable decode position), so it "
                    "explains nothing"
                )
            elif control_drift is not None and c >= LOGPROB_THRESHOLD and d <= 2 * c:
                explained.append(note)
            else:
                unexplained.append(note)
    result["control_explained"] = explained
    result["control_unexplained"] = unexplained
    if unexplained:
        result["problems"].insert(
            0,
            f"batch-sensitivity control does not explain {len(unexplained)} failing "
            "request(s): " + ", ".join(unexplained[:5]),
        )
    else:
        result["verdict"] = "INCONCLUSIVE"
        result["kind"] = "batch-sensitivity"
        result["problems"].insert(
            0,
            "every failing request moves as much (within twice) between two eager "
            "launches that batch the requests differently; the bound cannot separate "
            "graph effects from batch composition on this serving path (heuristic, "
            "not proof of a correct graph)",
        )


def numerical_verdict(
    outputs: Dict[str, Dict[str, dict]],
    phases: Dict[str, dict],
    candidate: str,
    max_new_tokens: int,
    skip_reference: bool = False,
    control: str = "reference2",
) -> dict:
    """``candidate`` against the eager ``reference``; a crashed launch is a
    FAIL of kind ``crash`` whatever outputs exist for it and whichever side
    crashed, a skipped or absent launch NOT-RUN."""
    if skip_reference:
        return {
            "verdict": "NOT-RUN",
            "kind": "not-run",
            "candidate": candidate,
            "problems": ["--skip-reference"],
        }
    # Both launches' errors come before either launch's outputs: a crashed
    # candidate is a crash even when the reference left no outputs.
    for name in ("reference", candidate):
        error = (phases.get(name) or {}).get("error")
        if error:
            return {
                "verdict": "FAIL",
                "kind": "crash",
                "candidate": candidate,
                "problems": [f"{name} run failed: {error}"],
            }
    for name in ("reference", candidate):
        if name not in outputs:
            return {
                "verdict": "NOT-RUN",
                "kind": "not-run",
                "candidate": candidate,
                "problems": [f"no {name} outputs"],
            }
    result = compare(
        outputs["reference"], outputs[candidate], candidate, max_new_tokens
    )
    result["candidate"] = candidate
    if (phases.get(control) or {}).get("error"):
        result["batch_sensitivity_control"] = {
            "verdict": "NOT-RUN",
            "problems": [f"{control} run failed: {phases[control]['error']}"],
        }
    elif control in outputs:
        ctl = compare(outputs["reference"], outputs[control], control, max_new_tokens)
        result["batch_sensitivity_control"] = {
            "description": (
                "the eager reference launched again with every request sent alone; "
                "per-request drift of the eager path under another batch composition"
            ),
            **{
                k: ctl[k]
                for k in ("verdict", "kind", "max_drift", "request_drift", "problems")
            },
        }
        _apply_batch_sensitivity_control(result, ctl)
    return result


# -- server-log evidence -------------------------------------------------------------

_BATCH_LINE = re.compile(r"\b(Prefill|Decode) batch\b.*\bcuda graph: (True|False)")
_RESOLVED_KEYS = (
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "cuda_graph_backend_prefill",
    "cuda_graph_backend_decode",
    "disable_cuda_graph",
    "enable_lora",
    "lora_backend",
    "moe_runner_backend",
    "tp_size",
    "pp_size",
    "dp_size",
    "attn_dp_size",
    "enable_dp_attention",
    "nnodes",
    "node_rank",
    "enable_torch_compile",
    "speculative_algorithm",
)


def server_log_evidence(log_path: str) -> dict:
    """Counts of prefill / decode batches that replayed a graph (the scheduler's
    own log lines) and the resolved server arguments the verdicts depend on."""
    batches = {
        "prefill_graph": 0,
        "prefill_eager": 0,
        "decode_graph": 0,
        "decode_eager": 0,
    }
    resolved: Dict[str, str] = {}
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = _BATCH_LINE.search(line)
                if m:
                    key = f"{m.group(1).lower()}_{'graph' if m.group(2) == 'True' else 'eager'}"
                    batches[key] += 1
                    continue
                if not resolved and "server_args=" in line:
                    for key in _RESOLVED_KEYS:
                        km = re.search(
                            rf"(?:'{key}': |\b{key}=)('([^']*)'|[^,)}}\s]+)", line
                        )
                        if km:
                            resolved[key] = (
                                km.group(2) if km.group(2) is not None else km.group(1)
                            )
    except OSError:
        pass
    return {"graph_batches": batches, "resolved_args": resolved}


# -- tracer reports ---------------------------------------------------------------------


def _read_reports(
    report_dir: str, prefix: str, run_id: Optional[str] = None
) -> List[dict]:
    """Reports of this run only: a report without the run id, or with another
    one, belongs to an earlier launch in the same directory."""
    reports = []
    if not os.path.isdir(report_dir):
        return reports
    for name in sorted(os.listdir(report_dir)):
        if name.startswith(prefix) and name.endswith(".json"):
            try:
                with open(os.path.join(report_dir, name)) as f:
                    report = json.load(f)
            except (OSError, ValueError):
                continue
            if run_id is not None and report.get("run_id") != run_id:
                continue
            reports.append(report)
    return reports


FLUSH_REASONS = ("SIGUSR1", "flush-request")  # reasons of a report written by a flush
TRACER_VERSION = 3  # the report format the verdicts accept (tracer.TRACER_VERSION)


def topology(resolved_args: Dict[str, str]) -> dict:
    """Derive (replica, pipeline rank, tensor rank) workers from resolved args.

    Attention-DP partitions tensor ranks; it adds no workers. Refuse multi-node
    and replica-plus-attention-DP layouts rather than guess their topology.
    """
    result: Dict[str, Any] = {
        "source": "resolved" if resolved_args else "unavailable",
        "expected_workers": None,
        "identities": None,
        "unsupported": None,
    }
    if not resolved_args:
        result["unsupported"] = "resolved server arguments unavailable"
        return result
    sizes: Dict[str, Optional[int]] = {}
    for key in ("tp_size", "pp_size", "dp_size", "attn_dp_size", "nnodes"):
        try:
            sizes[key] = int(resolved_args.get(key, 1))
        except (TypeError, ValueError):
            sizes[key] = None
    result.update(sizes)
    if None in sizes.values():
        result["unsupported"] = "unreadable parallel sizes in the resolved arguments"
        return result
    tp, pp, dp, attn_dp, nnodes = (
        sizes[k] for k in ("tp_size", "pp_size", "dp_size", "attn_dp_size", "nnodes")
    )
    if nnodes != 1:
        result["unsupported"] = (
            f"multi-node launch (nnodes={nnodes}): this node runs only part of the topology"
        )
        return result
    if dp > 1 and attn_dp > 1:
        result["unsupported"] = (
            "data-parallel replicas combined with attention data parallelism "
            f"(dp_size={dp}, attn_dp_size={attn_dp})"
        )
        return result
    result["identities"] = [
        [d, p, t] for d in range(dp) for p in range(pp) for t in range(tp)
    ]
    result["expected_workers"] = dp * pp * tp
    return result


def _worker_identity(report: dict, topo: dict) -> Optional[Tuple[int, int, int]]:
    """(replica, pipeline rank, tensor rank) from the ranks SGLang published
    in the reporting process; None when it published none."""
    parallel = report.get("parallel") or {}
    tp, pp = parallel.get("tp_rank"), parallel.get("pp_rank")
    if not isinstance(tp, int) or not isinstance(pp, int):
        return None
    if topo.get("dp_size", 1) > 1:
        dp = parallel.get("dp_rank")
        if not isinstance(dp, int):
            return None
    else:
        dp = 0
    return dp, pp, tp


def _identity_problems(
    reports: List[dict], topo: Optional[dict], expected_workers: int
) -> Tuple[List[str], str]:
    """Which expected worker each report is, by published ranks and physical
    device. Without a topology nothing is verified, and that is said; reports
    written before the tracer carried identity (an archived run) are judged
    on their replays alone and say so, since nothing in them can be matched to
    a worker."""
    if topo is None:
        return [], "unverified: no resolved topology"
    if reports and all(
        isinstance(r.get("tracer_version"), int)
        and r["tracer_version"] < TRACER_VERSION
        for r in reports
    ):
        # Nothing in such reports could be matched to a worker whatever the
        # topology says, so an unresolved topology is a note here, not a
        # failure; the count check still applies.
        status = (
            "unverified-legacy: reports carry no worker identity (tracer version "
            f"before {TRACER_VERSION})"
        )
        if topo.get("unsupported"):
            status += f"; topology: {topo['unsupported']}"
        return [], status
    if topo.get("unsupported"):
        return [f"unsupported topology: {topo['unsupported']}"], "unsupported"
    expected = {tuple(i) for i in topo["identities"]}
    seen: Dict[Tuple[int, int, int], List[Any]] = {}
    physical: Dict[Tuple[Any, Any], List[Any]] = {}
    problems: List[str] = []
    for report in reports:
        pid = report.get("pid")
        identity = _worker_identity(report, topo)
        if identity is None:
            problems.append(
                f"pid {pid}: a capture process without published scheduler ranks "
                "cannot stand in for a worker"
            )
            continue
        parallel = report.get("parallel") or {}
        for key in ("tp_size", "pp_size", "dp_size"):
            if parallel.get(key) not in (None, topo[key]):
                problems.append(
                    f"pid {pid}: reports {key}={parallel.get(key)}, resolved {topo[key]}"
                )
        if identity not in expected:
            problems.append(
                f"pid {pid}: worker {_identity_name(identity)} is outside the resolved topology"
            )
        seen.setdefault(identity, []).append(pid)
        uuid = report.get("device_uuid")
        if uuid is None:
            if expected_workers > 1:
                problems.append(
                    f"pid {pid}: physical device identity unavailable; distinct "
                    "devices cannot be verified"
                )
        else:
            physical.setdefault((report.get("hostname"), uuid), []).append(pid)
    for identity, pids in sorted(seen.items()):
        if len(pids) > 1:
            problems.append(
                f"worker {_identity_name(identity)} reported by {len(pids)} processes (pids {pids})"
            )
    for identity in sorted(expected - set(seen)):
        problems.append(f"no report from worker {_identity_name(identity)}")
    for (host, uuid), pids in sorted(physical.items(), key=str):
        if len(pids) > 1:
            problems.append(
                f"pids {pids} report the same physical device {uuid} on {host}"
            )
    return problems, ("verified" if not problems else "failed")


def _identity_name(identity: Tuple[int, int, int]) -> str:
    return f"dp={identity[0]} pp={identity[1]} tp={identity[2]}"


def flush_workers(
    report_dir: str,
    run_id: str,
    server: Optional[Server] = None,
    timeout_s: float = 120,
) -> Tuple[List[dict], str]:
    """Ask every worker of this run for a final check and wait for the
    acknowledgement: a ``flush-<pid>.request`` file carrying a fresh nonce,
    which the tracer sees at its next replay (probe requests keep replays
    coming) and echoes in the report it writes. Only workers that reported
    under this run id are asked; no signal is sent to any process."""
    nonce = f"{run_id}-{time.time_ns()}"
    pids = [r["pid"] for r in _read_reports(report_dir, "worker-", run_id)]
    for pid in pids:
        with open(os.path.join(report_dir, f"flush-{pid}.request"), "w") as f:
            f.write(nonce)
    deadline = time.time() + timeout_s
    next_probe = 0.0
    while time.time() < deadline:
        reports = _read_reports(report_dir, "worker-", run_id)
        if reports and all(r.get("flush_nonce") == nonce for r in reports):
            return reports, nonce
        if server is not None and time.time() >= next_probe:
            next_probe = time.time() + 3
            try:
                server.generate(
                    {
                        "text": _BASE_PROMPTS[0],
                        "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
                    }
                )
            except (OSError, urllib.error.URLError, ValueError):
                pass  # a worker that died has already written its report
        time.sleep(1)
    return _read_reports(report_dir, "worker-", run_id), nonce


def lifetime_verdict(
    reports: List[dict],
    expected_workers: int,
    run_id: Optional[str] = None,
    nonce: Optional[str] = None,
    topo: Optional[dict] = None,
) -> dict:
    """Require worker identities, captures, replays, clean checks and a flush nonce.

    Without a resolved topology, identity remains unverified; a status string
    alone never establishes a pass.
    """
    problems: List[str] = []
    if len(reports) < expected_workers:
        problems.append(f"{len(reports)} worker report(s), expected {expected_workers}")
    identity_problems, identity = _identity_problems(reports, topo, expected_workers)
    problems += identity_problems
    devices: Dict[str, int] = {}
    for report in reports:
        pid = report.get("pid")
        if report.get("tracer_version") != TRACER_VERSION:
            problems.append(f"pid {pid}: unexpected tracer version")
            continue
        if run_id is not None and report.get("run_id") != run_id:
            problems.append(
                f"pid {pid}: report of another run ({report.get('run_id')})"
            )
            continue
        device = report.get("device")
        if device is not None:
            devices[device] = devices.get(device, 0) + 1
        counts = report.get("counts") or {}
        findings = report.get("findings") or {}
        if report.get("status") != "clean":
            problems.append(
                f"pid {pid}: status {report.get('status')} "
                f"({findings.get('dangling_total', 0)} dangling operand(s), "
                f"{len(findings.get('incomplete', []))} incomplete, "
                f"{len(findings.get('failed', []))} failed inspection(s))"
            )
            for finding in findings.get("dangling", [])[:3]:
                problems.append(
                    f"pid {pid}: graph {finding.get('graph')!r} reads "
                    f"{finding.get('pointer')} freed while alive; allocated at "
                    f"{finding.get('allocated_at')}; freed at {finding.get('freed_at')}"
                )
            continue
        if not findings or findings.get("status") != "clean":
            problems.append(f"pid {pid}: no clean inspection findings recorded")
        if counts.get("captures", 0) < 1 or findings.get("graphs", 0) < 1:
            problems.append(f"pid {pid}: no captured graph was inspected")
        if counts.get("checks", 0) < 1:
            problems.append(f"pid {pid}: zero inspections")
        if counts.get("replays", 0) < 1:
            problems.append(f"pid {pid}: no replay observed")
        if nonce is not None and (
            report.get("reason") != "flush-request"
            or report.get("flush_nonce") != nonce
        ):
            problems.append(
                f"pid {pid}: no acknowledged final flush "
                f"(reason {report.get('reason')!r}, nonce {report.get('flush_nonce')!r})"
            )
        elif nonce is None and report.get("reason") not in FLUSH_REASONS:
            problems.append(f"pid {pid}: no final flush ({report.get('reason')})")
        for error in report.get("errors", []):
            problems.append(f"pid {pid}: {error}")
    if topo is None:  # logical names only: the best available without a topology
        for device, count in sorted(devices.items()):
            if count > 1:
                problems.append(f"{count} workers report the same device {device}")
    replayed = {}
    for report in reports:
        for label, count in report.get("label_replays", {}).items():
            replayed[label] = replayed.get(label, 0) + count
    return {
        "verdict": "PASS"
        if reports and not problems
        else ("NO-WORKERS" if not reports else "FAIL"),
        "run_id": run_id,
        "flush_nonce": nonce,
        "identity": identity,
        "topology": topo,
        "problems": problems,
        "workers": [
            {
                "pid": r.get("pid"),
                "hostname": r.get("hostname"),
                "rank": r.get("rank"),
                "parallel": r.get("parallel"),
                "device": r.get("device"),
                "device_uuid": r.get("device_uuid"),
                "visible_devices": r.get("visible_devices"),
                "source": r.get("source"),
                "run_id": r.get("run_id"),
                "status": r.get("status"),
                "reason": r.get("reason"),
                "flush_nonce": r.get("flush_nonce"),
                "counts": r.get("counts"),
                "findings": {
                    k: v
                    for k, v in (r.get("findings") or {}).items()
                    if k
                    in (
                        "status",
                        "graphs",
                        "nodes",
                        "pointers",
                        "static",
                        "temporaries",
                        "intermediates",
                        "outlived",
                        "untracked",
                        "dangling_total",
                        "incomplete",
                        "failed",
                    )
                },
                "dangling": (r.get("findings") or {}).get("dangling", [])[:5],
            }
            for r in reports
        ],
        "label_replays": replayed,
    }


# -- coverage -------------------------------------------------------------------------

_SIZE_TOKEN = re.compile(r"\s(?:size|rows)=\d+")
_TRUE = ("1", "true", "yes", "on", "y", "t")
_LEGACY_DUAL_GRAPH_ENV = "SGLANG_ENABLE_LORA_NOLORA_GRAPH"  # ledgers before the rename
# The launch environment the coverage requirements depend on: persisted in
# the ledger (nothing else from the environment is).
_COVERAGE_ENV_KEYS = ("SGLANG_ENABLE_LORA_DUAL_GRAPH", _LEGACY_DUAL_GRAPH_ENV)


def _dual_graph_knob(launch_env: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    """The LoRA dual-graph knob, accepting its pre-rename name when the current
    one is absent (a compatibility rule, reported as such)."""
    if "SGLANG_ENABLE_LORA_DUAL_GRAPH" in launch_env:
        return str(launch_env["SGLANG_ENABLE_LORA_DUAL_GRAPH"]).lower() in _TRUE, None
    if _LEGACY_DUAL_GRAPH_ENV in launch_env:
        return (
            str(launch_env[_LEGACY_DUAL_GRAPH_ENV]).lower() in _TRUE,
            f"dual-graph knob read from its legacy name {_LEGACY_DUAL_GRAPH_ENV}",
        )
    return False, None


def normalize_label(label: str) -> str:
    """Drop only the size: ``DecodeCudaGraphRunner size=4 variant_label=lora``
    becomes ``DecodeCudaGraphRunner variant_label=lora``; LoRA and attention
    variants stay, so replaying one variant cannot stand in for another."""
    return _SIZE_TOKEN.sub("", label)


def required_coverage(
    resolved_args: Dict[str, str], launch_env: Dict[str, str], expect_labels: List[str]
) -> Tuple[List[str], List[str]]:
    """Derive required replay-label substrings and note unresolved configuration.

    Require automatic LoRA variants only where production records them; without
    resolved arguments, only explicit ``--expect-labels`` requirements count.
    """
    required = list(expect_labels)
    notes: List[str] = []
    if not resolved_args:
        notes.append(
            "resolved server arguments unavailable: requirements come from "
            "--expect-labels only"
        )
        return required, notes
    disabled_all = str(resolved_args.get("disable_cuda_graph")).lower() in _TRUE
    decode_on = (
        not disabled_all
        and resolved_args.get("cuda_graph_backend_decode") != "disabled"
    )
    if decode_on:
        required.append("DecodeCudaGraphRunner")
    prefill = resolved_args.get("cuda_graph_backend_prefill")
    if prefill in ("full", "breakable"):
        required.append("PrefillCudaGraphRunner")
    elif prefill not in (None, "None", "disabled"):
        notes.append(f"prefill graph backend {prefill!r}: no requirement derived")
    dual, knob_note = _dual_graph_knob(launch_env)
    if knob_note:
        notes.append(knob_note)
    if dual:
        lora_on = str(resolved_args.get("enable_lora")).lower() in _TRUE
        if decode_on and lora_on and resolved_args.get("lora_backend") == "triton_v2":
            required += ["variant_label=lora", "variant_label=nolora"]
        else:
            notes.append(
                "dual-graph knob set but no LoRA decode variant is recorded in this "
                "configuration (needs decode graphs, --enable-lora and lora_backend "
                "triton_v2): none required"
            )
    seen: List[str] = []
    for item in required:
        if item not in seen:
            seen.append(item)
    return seen, notes


def coverage_verdict(
    reports: List[dict],
    expected_workers: int,
    required: List[str],
    notes: List[str],
    agreement: Optional[dict] = None,
    topo: Optional[dict] = None,
) -> dict:
    """Every required path must have replayed on every expected worker (matched
    on labels with the size normalized out); the inventory of labels captured
    but never replayed keeps the raw labels, size included, so an unexercised
    bucket is reported, not failed, unless a requirement covers it."""
    coverage: Dict[str, Any] = {
        "workers_reported": len(reports),
        "workers_expected": expected_workers,
        "required": required,
        "notes": list(notes),
        "problems": [],
        "per_worker": [],
        "label_captures": {},
        "label_replays": {},
        "captured_not_replayed": [],
    }
    if agreement is not None:
        coverage["instrumented_vs_ordinary"] = {
            k: agreement.get(k)
            for k in (
                "verdict",
                "kind",
                "max_drift",
                "problems",
                "request_drift",
                "original_verdict",
                "historical",
            )
        }
        if agreement.get("kind") == "evidence-rejected":
            coverage["problems"] += agreement["problems"]
        elif agreement.get("kind") == "not-recomputed":
            # An original PASS carried forward is a note; anything else is a problem.
            key = "notes" if agreement["verdict"] == "PASS" else "problems"
            coverage[key] += agreement["problems"]
        elif agreement["verdict"] != "PASS":
            coverage["problems"].append(
                f"instrumented run differs from the ordinary run ({agreement['kind']})"
            )
    if len(reports) < expected_workers:
        coverage["problems"].append(
            f"{len(reports)} worker report(s), expected {expected_workers}"
        )
    identity_problems, coverage["identity"] = _identity_problems(
        reports, topo, expected_workers
    )
    coverage["problems"] += identity_problems
    not_replayed: set = set()
    any_replay = False
    for report in reports:
        replayed: Dict[str, int] = {}
        raw_captures = report.get("label_captures") or {}
        raw_replays = report.get("label_replays") or {}
        for label, count in raw_captures.items():
            coverage["label_captures"][label] = (
                coverage["label_captures"].get(label, 0) + count
            )
        for label, count in raw_replays.items():
            coverage["label_replays"][label] = (
                coverage["label_replays"].get(label, 0) + count
            )
            key = normalize_label(label)
            replayed[key] = replayed.get(key, 0) + count
        any_replay = any_replay or any(c > 0 for c in replayed.values())
        missing = [
            need
            for need in required
            if not any(need in key and count > 0 for key, count in replayed.items())
        ]
        unexercised = sorted(
            label
            for label, count in raw_captures.items()
            if count > 0 and raw_replays.get(label, 0) == 0
        )
        not_replayed.update(unexercised)
        coverage["per_worker"].append(
            {
                "pid": report.get("pid"),
                "device": report.get("device"),
                "replayed": replayed,
                "captured_not_replayed": unexercised,
                "missing": missing,
            }
        )
        for need in missing:
            coverage["problems"].append(
                f"pid {report.get('pid')}: no replay of a graph matching {need!r}"
            )
    coverage["captured_not_replayed"] = sorted(not_replayed)
    if reports and not any_replay:
        coverage["problems"].append("no graph replay was observed")
    if not reports and required:
        coverage["problems"].append("no worker report to verify the required coverage")
    if not required:
        coverage["problems"].append(
            "no coverage requirement could be derived; pass --expect-labels"
        )
    coverage["verdict"] = "PASS" if not coverage["problems"] else "INCOMPLETE"
    return coverage


# -- environment ----------------------------------------------------------------------


def environment(python: str, cwd: str) -> dict:
    env: Dict[str, Any] = {
        "python": python,
        "cwd": cwd,
        "hostname": os.uname().nodename,
    }
    try:
        out = subprocess.run(
            [
                python,
                "-c",
                "import torch,importlib.metadata as m;print(torch.__version__);"
                "print(m.version('sglang'));print(m.version('flashinfer-python') if m.version('flashinfer-python') else '')",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cwd,
        ).stdout.split()
        env["torch"], env["sglang"], env["flashinfer"] = (out + ["", "", ""])[:3]
    except Exception as e:  # noqa: BLE001 - best effort
        env["versions_error"] = repr(e)
    for cmd, key in (
        (
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            "gpus",
        ),
        (["git", "-C", cwd, "rev-parse", "HEAD", "HEAD^{tree}"], "git"),
        (
            ["git", "-C", cwd, "status", "--porcelain", "--untracked-files=no"],
            "git_status",
        ),
    ):
        try:
            env[key] = (
                subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                .stdout.strip()
                .splitlines()
            )
        except Exception as e:  # noqa: BLE001
            env[key] = repr(e)
    status = env.pop("git_status", None)
    env["git_dirty"] = bool(status) if isinstance(status, list) else None
    return env


# -- evidence identity and recomputation -------------------------------------------

_TOOL_FILES = ("harness.py", "tracer.py", "classifier.py", "inject/sitecustomize.py")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(
    cwd_git: Any, reports: List[dict], cwd_dirty: Optional[bool] = None
) -> dict:
    """Match every worker's imported Git identity against the harness checkout.

    Missing or dirty sources remain unverified. Even a match identifies the
    committed tree, not the bytes of every file on disk.
    """
    sources: List[dict] = []
    unidentified: List[Any] = []
    dirty: List[Any] = []
    for report in reports:
        source = report.get("source")
        git = source.get("git") if isinstance(source, dict) else None
        if not (isinstance(git, list) and len(git) == 2):
            unidentified.append(report.get("pid"))
            continue
        if source.get("dirty"):
            dirty.append(report.get("pid"))
        if source not in sources:
            sources.append(source)
    heads = {tuple(s["git"]) for s in sources}
    if not reports:
        status = "unverified: no worker report"
    elif unidentified:
        status = (
            f"unverified: worker pid(s) {unidentified} reported no git identity for "
            "the sglang they imported (not a checkout, or the probe failed)"
        )
    elif not (isinstance(cwd_git, list) and heads == {tuple(cwd_git)}):
        status = (
            f"unverified: workers imported {sorted(heads)}, the harness working "
            f"directory is {cwd_git}"
        )
    elif dirty or cwd_dirty:
        status = (
            "unverified: uncommitted changes in "
            + ("the harness working directory" if cwd_dirty else "")
            + (" and " if cwd_dirty and dirty else "")
            + (f"the checkout of worker pid(s) {dirty}" if dirty else "")
            + "; the git identity names the committed tree, not the files on disk"
        )
    else:
        status = (
            f"verified by git identity: every worker imported commit {cwd_git[0]} "
            "of the harness working directory and no checkout reported uncommitted "
            "changes (an identity of the committed tree, not a byte check of the "
            "files on disk)"
        )
    return {
        "harness_cwd_git": cwd_git,
        "harness_cwd_dirty": cwd_dirty,
        "worker_source": sources,
        "source_identity": status,
    }


def tool_digest() -> dict:
    """Identity of the tool that produced a ledger: one digest over its source."""
    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for name in _TOOL_FILES:
        digest.update(name.encode())
        with open(os.path.join(here, name), "rb") as f:
            digest.update(f.read())
    return {"sha256": digest.hexdigest(), "files": list(_TOOL_FILES)}


def _resolved_args(phases: Dict[str, dict]) -> Dict[str, str]:
    """The resolved server arguments of the first graph launch whose log had
    them (instrumented, then ordinary)."""
    for name in ("instrumented", "ordinary"):
        phase = phases.get(name) or {}
        if phase.get("resolved_args"):
            return phase["resolved_args"]
    return {}


def _write_outputs(
    path: str, run_id: str, phase: str, outputs: Dict[str, dict]
) -> None:
    """An outputs file names the run and phase it belongs to."""
    with open(path, "w") as f:
        json.dump({"run_id": run_id, "phase": phase, "outputs": outputs}, f)


def _locate_ledger(path: str) -> str:
    """``path`` is a run directory (holding ``ledger.json``) or a ``--out``
    directory holding exactly one run directory."""
    if os.path.isfile(os.path.join(path, "ledger.json")):
        return path
    runs = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name, "ledger.json"))
    )
    if len(runs) != 1:
        raise FileNotFoundError(
            f"{path}: no ledger.json, and {len(runs)} run directories with one"
            + (f" ({', '.join(os.path.basename(r) for r in runs)})" if runs else "")
        )
    return runs[0]


def _archived_outputs(
    run_dir: str, ledger: dict, phase: str
) -> Tuple[Optional[Dict[str, dict]], str, str]:
    """Read phase outputs with their provenance and an evidence note.

    Reject failed phases or mismatched digest/count/run/phase evidence. Legacy
    files without recorded digests remain explicitly unverified.
    """
    entry = ledger.get("phases", {}).get(phase) or {}
    if entry.get("error"):
        return (
            None,
            "failed-phase",
            f"{phase}: the phase failed; its outputs are not consumed",
        )
    evidence = (ledger.get("evidence") or {}).get("outputs")
    if evidence is None:
        path = os.path.join(run_dir, f"{phase}.outputs.json")
        if not os.path.exists(path):
            return None, "never-archived", f"{phase}: no outputs file"
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "outputs" in data and "phase" in data:
            data = data["outputs"]
        return (
            data,
            "unverified-legacy",
            f"{phase}: outputs without a recorded digest (legacy ledger), unverified",
        )
    recorded = evidence.get(phase)
    if recorded is None:
        return None, "never-archived", f"{phase}: no outputs recorded for this run"
    path = os.path.join(run_dir, os.path.basename(recorded["path"]))
    if not os.path.exists(path):
        return (
            None,
            "missing",
            f"{phase}: recorded outputs file {os.path.basename(path)} is missing",
        )
    if _sha256_file(path) != recorded.get("sha256"):
        return (
            None,
            "mismatch",
            f"{phase}: outputs file does not match the recorded digest",
        )
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "outputs" in data and "phase" in data:
        if data.get("run_id") != ledger.get("run_id") or data.get("phase") != phase:
            return (
                None,
                "mismatch",
                f"{phase}: outputs file belongs to run {data.get('run_id')!r} phase "
                f"{data.get('phase')!r}",
            )
        data = data["outputs"]
    if len(data) != recorded.get("requests"):
        return (
            None,
            "mismatch",
            f"{phase}: outputs hold {len(data)} requests, {recorded.get('requests')} recorded",
        )
    return data, "verified", f"{phase}: verified against the recorded digest"


def _instrumented_failure(phase: Optional[dict], verdict: dict) -> dict:
    """A launch error recorded for the instrumented phase is a lifetime
    failure whatever the reports say: a worker that died may have left clean
    reports and an acknowledged nonce from before it died. Used live and when
    an archive is re-judged."""
    error = (phase or {}).get("error")
    if not error:
        return verdict
    verdict = dict(verdict)
    verdict["problems"] = [
        f"instrumented launch recorded an error: {error}",
        *verdict.get("problems", []),
    ]
    if verdict.get("verdict") == "PASS":
        verdict["verdict"] = "FAIL"
    return verdict


def _phase_ran(phases: Dict[str, dict], name: str) -> bool:
    entry = phases.get(name) or {}
    return bool(entry) and not entry.get("error") and bool(entry.get("requests"))


def _archived_agreement(
    ledger: dict,
    outputs: Dict[str, Dict[str, dict]],
    provenance: Dict[str, str],
    max_new_tokens: int,
) -> Optional[dict]:
    """Recompute instrumented/plain agreement, or explain why it is unavailable.

    Rejected recorded evidence inherits nothing. Never-archived outputs retain
    only a labelled historical judgment; an unrun launch has no agreement.
    """
    phases = ledger.get("phases", {})
    if "ordinary" in outputs and "instrumented" in outputs:
        return compare(
            outputs["ordinary"], outputs["instrumented"], "instrumented", max_new_tokens
        )
    if not (_phase_ran(phases, "ordinary") and _phase_ran(phases, "instrumented")):
        return None  # no agreement ever existed for this run
    rejected = [
        p
        for p in ("ordinary", "instrumented")
        if provenance.get(p) in ("mismatch", "missing")
    ]
    if rejected:
        return {
            "verdict": "NOT-RECOMPUTED",
            "kind": "evidence-rejected",
            "max_drift": None,
            "problems": [
                "instrumented vs ordinary agreement cannot be re-judged: recorded "
                f"outputs of {', '.join(rejected)} were rejected ({', '.join(provenance[p] for p in rejected)}); "
                "the original judgment is not inherited"
            ],
            "request_drift": {},
            "original_verdict": None,
            "historical": False,
        }
    original = ((ledger.get("verdicts") or {}).get("coverage") or {}).get(
        "instrumented_vs_ordinary"
    ) or {}
    verdict = original.get("verdict")
    return {
        "verdict": "PASS" if verdict == "PASS" else "NOT-RECOMPUTED",
        "kind": "not-recomputed",
        "max_drift": original.get("max_drift"),
        "problems": [
            "instrumented vs ordinary agreement not re-judged (the archive never "
            f"saved the outputs); historical judgment {verdict}, not re-verified"
        ],
        "request_drift": {},
        "original_verdict": verdict,
        "historical": True,
    }


def _preserve(path: str) -> None:
    """Keep an existing recomputed ledger under its own tool digest."""
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            tag = (json.load(f).get("tool") or {}).get("sha256", "")[:12] or "prev"
    except (OSError, ValueError):
        tag = "prev"
    os.replace(path, path.replace(".json", f".{tag}.json"))


def recompute(out_dir: str, max_new_tokens: Optional[int] = None) -> dict:
    """Re-judge saved evidence into a separate ledger, preserving earlier results.

    Lifetime is not recomputable when reports predate the current worker-identity
    protocol. The original ledger, outputs and reports are never modified.
    """
    out_dir = _locate_ledger(out_dir)
    with open(os.path.join(out_dir, "ledger.json")) as f:
        ledger = json.load(f)
    max_new_tokens = max_new_tokens or int(ledger.get("max_new_tokens", 16))
    outputs: Dict[str, Dict[str, dict]] = {}
    provenance: Dict[str, Any] = {}
    notes: List[str] = []
    for phase in ("reference", "reference2", "ordinary", "instrumented"):
        data, status, note = _archived_outputs(out_dir, ledger, phase)
        provenance[phase] = status
        notes.append(note)
        if data is not None:
            outputs[phase] = data
    phases = ledger.get("phases", {})
    candidate = (
        "ordinary" if "ordinary" in outputs or "ordinary" in phases else "instrumented"
    )
    # Recorded evidence that is rejected (digest or identity mismatch) or
    # missing is kept apart from an old archive that never saved outputs:
    # the first never yields a verdict, the second is simply not re-judged.
    # Only the two required sides of the comparison can reject it, and only
    # when both ran at all; a launch that never ran keeps its NOT-RUN
    # whatever happened to other files. The optional control (reference2) is
    # judged below on its own: unusable when rejected, never a replacement
    # for the verdict the required outputs establish.
    required_phases = ("reference", candidate)
    rejected = sorted(
        p
        for p in required_phases
        if _phase_ran(phases, p) and provenance.get(p) in ("mismatch", "missing")
    )
    if rejected and all(_phase_ran(phases, p) for p in required_phases):
        numerical: Dict[str, Any] = {
            "verdict": "INCONCLUSIVE",
            "kind": (
                "evidence-mismatch"
                if any(provenance[p] == "mismatch" for p in rejected)
                else "evidence-missing"
            ),
            "candidate": candidate,
            "problems": [n for n in notes if n.split(":")[0] in rejected],
        }
    else:
        numerical = numerical_verdict(outputs, phases, candidate, max_new_tokens)
    if (
        _phase_ran(phases, "reference2")
        and provenance.get("reference2") in ("mismatch", "missing")
        and "reference2" not in outputs
    ):
        reason = provenance["reference2"]
        numerical["batch_sensitivity_control"] = {
            "verdict": "UNUSABLE",
            "kind": f"evidence-{reason}",
            "problems": [n for n in notes if n.startswith("reference2:")],
        }
        if numerical.get("verdict") == "FAIL":
            numerical["problems"].insert(
                0,
                f"batch-sensitivity control unusable (recorded outputs {reason}); "
                f"the verdict stands on the reference and {candidate} outputs alone",
            )
    consumed = [p for p in ("reference", candidate, "reference2") if p in outputs]
    numerical["evidence_provenance"] = (
        "unverified-legacy"
        if any(provenance[p] == "unverified-legacy" for p in consumed)
        else "verified"
    )
    verdicts: Dict[str, Any] = {"numerical": numerical}
    reports = (phases.get("instrumented") or {}).get("worker_reports", [])
    topo = ledger.get("topology") or topology(_resolved_args(phases))
    expected_workers = int(
        topo.get("expected_workers") or ledger.get("expected_workers", 1)
    )
    versions = sorted({r.get("tracer_version") for r in reports}, key=str)
    if versions and versions != [TRACER_VERSION]:
        verdicts["lifetime"] = {
            "verdict": "NOT-RECOMPUTED",
            "problems": [
                f"reports predate worker identity (tracer version {versions}, "
                f"the verdict needs {TRACER_VERSION})"
            ],
        }
    else:
        verdicts["lifetime"] = lifetime_verdict(
            reports,
            expected_workers,
            ledger.get("run_id"),
            (phases.get("instrumented") or {}).get("flush_nonce"),
            topo,
        )
    verdicts["lifetime"] = _instrumented_failure(
        phases.get("instrumented"), verdicts["lifetime"]
    )
    # The requirements are the union of what the run declared, what its
    # original verdict checked and what the metadata derives now, so a
    # recomputation can only hold a run to more, never to less.
    declaration = ledger.get("coverage_declaration")
    if declaration is not None:
        env = dict(declaration.get("launch_env") or {})
        provenance_note = "declared"
    else:
        env = {}
        for phase in phases.values():
            env.update(phase.get("extra_env") or {})
        provenance_note = "unverified-legacy"
    resolved = _resolved_args(phases)
    derived, notes = required_coverage(resolved, env, ledger.get("expect_labels", []))
    required: List[str] = []
    for item in [
        *((declaration or {}).get("required") or []),
        *(((ledger.get("verdicts") or {}).get("coverage") or {}).get("required") or []),
        *derived,
    ]:
        if item not in required:
            required.append(item)
    if declaration is None:
        notes.append(
            "ledger without a coverage declaration: requirements derived from the "
            "phases' recorded environment differences, which omit launch-wide "
            "settings, and joined with the original verdict's requirement list"
        )
    agreement = _archived_agreement(ledger, outputs, provenance, max_new_tokens)
    verdicts["coverage"] = coverage_verdict(
        reports, expected_workers, required, notes, agreement, topo
    )
    verdicts["coverage"]["requirement_provenance"] = provenance_note
    result = {
        "recomputed_from": os.path.abspath(out_dir),
        "original_run_id": ledger.get("run_id"),
        "original_verdicts": {
            k: (v or {}).get("verdict")
            for k, v in (ledger.get("verdicts") or {}).items()
        },
        "tool": tool_digest(),
        "outputs": provenance,
        "notes": notes,
        "verdicts": verdicts,
    }
    path = os.path.join(out_dir, "ledger.recomputed.json")
    _preserve(path)
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    return result


# -- main --------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--serving-cmd",
        help="the exact serving command, e.g. 'python -m sglang.launch_server ...'",
    )
    parser.add_argument(
        "--recompute",
        metavar="DIR",
        help="re-judge an archived run directory with the current verdict logic "
        "(writes ledger.recomputed.json; nothing is launched)",
    )
    parser.add_argument("--out", help="directory for logs, reports and the ledger")
    parser.add_argument(
        "--lora-names",
        help="comma-separated adapter names to exercise (default: parsed from --lora-paths)",
    )
    parser.add_argument(
        "--expect-workers",
        type=int,
        help="scheduler workers that must report a tracer (default: tp*pp*dp)",
    )
    parser.add_argument(
        "--expect-labels",
        default="",
        help="comma-separated substrings that must appear in a replayed graph label, e.g. 'variant_label=lora,variant_label=nolora'",
    )
    parser.add_argument("--batch-sizes", default="8,4,2,1")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="skip the eager reference (no numerical verdict)",
    )
    parser.add_argument(
        "--skip-ordinary",
        action="store_true",
        help="skip the uninstrumented run (compare instrumented with the reference instead)",
    )
    parser.add_argument(
        "--skip-instrumented",
        action="store_true",
        help="numerical probe only: eager reference and the plain run, no tracer launch",
    )
    parser.add_argument(
        "--repeat-reference",
        action="store_true",
        help="launch the eager reference a second time with every request sent alone: a "
        "batch-sensitivity control; a drift failure that the same request's control explains "
        "(within twice its drift) becomes INCONCLUSIVE instead of FAIL",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="KEY=VALUE for every launch (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.recompute:
        result = recompute(
            args.recompute,
            args.max_new_tokens if argv and "--max-new-tokens" in argv else None,
        )
        for name in ("lifetime", "numerical", "coverage"):
            verdict = result["verdicts"][name]
            flag = ""
            if verdict.get("evidence_provenance") == "unverified-legacy":
                flag = " [evidence unverified: legacy outputs without digests]"
            if str(verdict.get("identity", "")).startswith("unverified-legacy"):
                flag += " [identity unverified: legacy reports without worker identity]"
            print(
                f"{name}: {verdict['verdict']} (was {result['original_verdicts'].get(name)}){flag}"
            )
            for problem in verdict.get("problems", [])[:6]:
                print(f"  {name}: {problem}")
        return 0
    if not args.serving_cmd or not args.out:
        parser.error("--serving-cmd and --out are required unless --recompute is used")

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    # Every artifact of this run (server logs, outputs, worker reports, the
    # ledger) lives in its own directory, so a reused --out never offers an
    # earlier launch's file as this run's evidence.
    run_dir = os.path.join(args.out, run_id)
    os.makedirs(run_dir)
    lora_names = args.lora_names.split(",") if args.lora_names else None
    cmd = ServingCommand.parse(args.serving_cmd, lora_names)
    expected_workers = args.expect_workers or cmd.expected_workers
    expect_labels = [s for s in args.expect_labels.split(",") if s]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    base_env = dict(os.environ)
    for item in args.env:
        key, value = item.split("=", 1)
        base_env[key] = value
    python = cmd.argv[0]
    ledger: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": run_dir,
        "serving_cmd": args.serving_cmd,
        "lora_names": cmd.lora_names,
        "expected_workers": expected_workers,
        "expect_labels": expect_labels,
        "batch_sizes": batch_sizes,
        "max_new_tokens": args.max_new_tokens,
        "check_every": args.check_every,
        "environment": environment(python, os.getcwd()),
        "evidence": {"tool": tool_digest(), "outputs": {}},
        "phases": {},
    }
    groups = build_sweep(batch_sizes, cmd.lora_names)
    ledger["traffic"] = {
        "groups": len(groups),
        "requests": sum(len(g) for g in groups),
        "prompts": len(_BASE_PROMPTS) * 2 + len(_LONG_PROMPT_WORDS),
    }
    outputs: Dict[str, Dict[str, dict]] = {}

    def run_phase(
        name: str,
        extra: List[str],
        env: Dict[str, str],
        phase_groups: Optional[List[List[Request]]] = None,
    ) -> None:
        if "--nccl-port" not in _options(cmd.argv):
            # A fresh port per launch: the launcher's own choice is the first
            # free port below the ephemeral range, which two servers starting
            # together both pick, and a port a server just released may still
            # refuse a new listener.
            extra = [*extra, "--nccl-port", str(_free_port())]
        server = Server(cmd, extra, env, os.path.join(run_dir, f"{name}.server.log"))
        started = time.time()
        phase: Dict[str, Any] = {
            "argv": server.argv,
            "extra_env": {
                k: v for k, v in env.items() if k not in base_env or base_env[k] != v
            },
        }
        ledger["phases"][name] = phase
        try:
            server.start(args.startup_timeout)
            phase["startup_s"] = round(time.time() - started, 1)
            outputs[name] = run_sweep(
                server, phase_groups or groups, args.max_new_tokens
            )
            phase["requests"] = len(outputs[name])
            path = os.path.join(run_dir, f"{name}.outputs.json")
            _write_outputs(path, run_id, name, outputs[name])
            ledger["evidence"]["outputs"][name] = {
                "path": os.path.basename(path),
                "run_id": run_id,
                "phase": name,
                "sha256": _sha256_file(path),
                "requests": len(outputs[name]),
            }
            if name == "instrumented":
                phase["worker_reports"], phase["flush_nonce"] = flush_workers(
                    env["SGLANG_GRAPH_LIFETIME_REPORT_DIR"], run_id, server
                )
        except Exception as e:  # noqa: BLE001 - recorded in the ledger
            phase["error"] = repr(e)
        finally:
            server.stop()
            phase["wall_s"] = round(time.time() - started, 1)
            phase.update(server_log_evidence(server.log_path))
            if name == "instrumented" and "worker_reports" not in phase:
                phase["worker_reports"] = _read_reports(
                    env["SGLANG_GRAPH_LIFETIME_REPORT_DIR"], "worker-", run_id
                )
            if name == "instrumented":
                phase["installed_markers"] = len(
                    _read_reports(
                        env["SGLANG_GRAPH_LIFETIME_REPORT_DIR"], "installed-", run_id
                    )
                )

    # Explicit per-phase selectors win over --disable-cuda-graph, so the
    # reference repeats them as disabled (last occurrence wins in argparse).
    eager = [
        "--disable-cuda-graph",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "disabled",
    ]
    if not args.skip_reference:
        run_phase("reference", eager, base_env)
        if args.repeat_reference:
            run_phase("reference2", eager, base_env, regroup_alone(groups))
    if not args.skip_ordinary:
        run_phase("ordinary", [], base_env)
    report_dir = os.path.join(run_dir, "tracer_reports")
    os.makedirs(report_dir, exist_ok=True)
    instrumented_env = dict(base_env)
    instrumented_env["PYTHONPATH"] = _INJECT_DIR + (
        os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else ""
    )
    instrumented_env["SGLANG_GRAPH_LIFETIME_REPORT_DIR"] = report_dir
    instrumented_env["SGLANG_GRAPH_LIFETIME_RUN_ID"] = run_id
    instrumented_env["SGLANG_GRAPH_LIFETIME_CHECK_EVERY"] = str(args.check_every)
    if not args.skip_instrumented:
        run_phase("instrumented", [], instrumented_env)

    verdicts: Dict[str, Any] = {}
    phase = ledger["phases"].get("instrumented", {})
    resolved_args = _resolved_args(ledger["phases"])
    # The topology the server resolved (aliases, config files and every
    # parallelism mode included) decides how many workers must report and
    # which; the command line only gave an estimate.
    topo = topology(resolved_args)
    topo["cli_estimate"] = cmd.expected_workers
    topo["declared"] = args.expect_workers
    if topo["expected_workers"] is not None:
        expected_workers = topo["expected_workers"]
        if args.expect_workers not in (None, expected_workers):
            topo["unsupported"] = (
                f"--expect-workers {args.expect_workers} disagrees with the resolved "
                f"topology ({expected_workers} workers)"
            )
    ledger["expected_workers"] = expected_workers
    ledger["topology"] = topo
    verdicts["lifetime"] = lifetime_verdict(
        phase.get("worker_reports", []),
        expected_workers,
        run_id,
        phase.get("flush_nonce"),
        topo,
    )
    verdicts["lifetime"] = _instrumented_failure(phase, verdicts["lifetime"])
    if args.skip_instrumented:
        verdicts["lifetime"] = {
            "verdict": "NOT-RUN",
            "problems": ["--skip-instrumented"],
        }
    candidate = "ordinary" if not args.skip_ordinary else "instrumented"
    verdicts["numerical"] = numerical_verdict(
        outputs, ledger["phases"], candidate, args.max_new_tokens, args.skip_reference
    )
    agreement = None
    if "ordinary" in outputs and "instrumented" in outputs:
        agreement = compare(
            outputs["ordinary"],
            outputs["instrumented"],
            "instrumented",
            args.max_new_tokens,
        )
    required, notes = required_coverage(resolved_args, base_env, expect_labels)
    ledger["coverage_declaration"] = {
        "expect_labels": expect_labels,
        "launch_env": {k: base_env[k] for k in _COVERAGE_ENV_KEYS if k in base_env},
        "resolved_args": resolved_args,
        "required": required,
        "notes": notes,
    }
    coverage = coverage_verdict(
        phase.get("worker_reports", []),
        expected_workers,
        required,
        notes,
        agreement,
        topo,
    )
    coverage["requirement_provenance"] = "declared"
    coverage["installed_markers"] = phase.get("installed_markers")
    coverage["graph_batches"] = {
        name: phase_.get("graph_batches") for name, phase_ in ledger["phases"].items()
    }
    coverage["resolved_args"] = resolved_args
    verdicts["coverage"] = coverage
    ledger["evidence"].update(
        source_identity(
            ledger["environment"].get("git"),
            phase.get("worker_reports", []),
            ledger["environment"].get("git_dirty"),
        )
    )
    ledger["evidence"]["resolved_args"] = resolved_args
    ledger["verdicts"] = verdicts
    ledger["claim"] = (
        "Under the recorded configuration and exercised traffic, the tool found no tracked "
        "lifetime violations and the numerical comparisons passed."
        if all(v.get("verdict") == "PASS" for v in verdicts.values())
        else "Not all verdicts passed; see verdicts."
    )
    ledger["limitations"] = [
        "pointers are read from recorded launch arguments only (device-side pointer tables are not followed)",
        "memory outside the PyTorch caching allocator is not tracked",
        "metadata that stays allocated but is no longer refreshed by replay is not detected",
        "only frees that happen before the final flush, and traffic the sweep sends, are covered",
        "one capture device per process; raw graphs mutated after their first inspection, and capture paths without a label wrapper, are unsupported",
        "requests whose first generated token differs contribute no decode position (insufficient evidence, never a pass)",
        "the batch-sensitivity control is a per-request heuristic (eager under another batch composition), not a proof of graph correctness",
    ]
    with open(os.path.join(run_dir, "ledger.json"), "w") as f:
        json.dump(ledger, f, indent=1, default=str)
    summary = [
        f"lifetime: {verdicts['lifetime']['verdict']}",
        f"numerical: {verdicts['numerical']['verdict']}",
        f"coverage: {verdicts['coverage']['verdict']}",
    ]
    for name, verdict in verdicts.items():
        for problem in verdict.get("problems", [])[:10]:
            summary.append(f"  {name}: {problem}")
    summary.append(f"labels replayed: {verdicts['coverage']['label_replays']}")
    summary.append(f"ledger: {os.path.join(run_dir, 'ledger.json')}")
    print("\n".join(summary))
    return 0 if all(v.get("verdict") == "PASS" for v in verdicts.values()) else 1
