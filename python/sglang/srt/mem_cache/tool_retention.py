"""Tool-execution-window KV retention policy for agent workloads.

When an agent session finishes a turn whose output ends in a tool call, its
next access is delayed by the tool execution latency — externally driven and,
unlike classic caching gaps, *predictable* from the tool identity. Plain LRU
evicts the least-recently-touched prefix, which in this workload is the
session *closest to its return* (thrashing fast-tool sessions while long-tool
sessions squat in the pool).

This policy annotates the finished session's radix leaf with an expected
return time (now + E[gap | tool]) and evicts leaves with the *latest*
expected return first (Belady-style with prior/learned gap estimates).
RadixCache implements the eviction-key change; this module owns everything
else: tool-call detection in the generated output, gap priors, and online
refinement of per-tool gap estimates.

E[gap | tool] is tracked as an EMA over *log* elapsed times (i.e., a
geometric-mean estimate, robust to heavy tails), seeded from a small prior
table (medians of common tool latency classes) or a single global default.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Prior medians (seconds) for common tool latency classes. Overridable via a
# JSON file passed through --tool-retention-prior: {"tool_name": seconds}.
DEFAULT_PRIOR_S = {
    "quick_fs": 0.3,
    "web_search": 2.0,
    "code_edit": 4.0,
    "run_tests": 10.0,
    "*": 3.0,  # fallback for unknown tools
}

_TOOL_CALL_RE = re.compile(r'<tool_call>\s*\{\s*"name"\s*:\s*"([^"]+)"')
_EMA_DECAY = 0.3  # weight of a new observation (log domain)
_TAIL_IDS = 96  # generated ids scanned for a trailing tool call


@dataclass
class ToolRetentionPolicy:
    tokenizer_path: str
    prior_s: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIOR_S))
    # tool -> EMA of log(seconds)
    ema_log: Dict[str, float] = field(default_factory=dict)
    tokenizer: Optional[Any] = None
    # observability counters
    n_annotated: int = 0
    n_observed: int = 0

    @classmethod
    def from_server_args(cls, server_args) -> Optional["ToolRetentionPolicy"]:
        if not getattr(server_args, "enable_tool_retention", False):
            return None
        policy = cls(tokenizer_path=server_args.model_path)
        prior_path = getattr(server_args, "tool_retention_prior", None)
        if prior_path:
            with open(prior_path) as f:
                policy.prior_s.update(json.load(f))
        return policy

    # ---- detection & prediction ----

    def detect_tool(self, output_ids) -> Optional[str]:
        """Return the tool name if the generated output contains a tool call."""
        if self.tokenizer is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=True
            )
        text = self.tokenizer.decode(output_ids[-_TAIL_IDS:], skip_special_tokens=False)
        m = _TOOL_CALL_RE.search(text)
        return m.group(1) if m else None

    def mean_gap_s(self, tool: str) -> float:
        if tool in self.ema_log:
            return max(1e-3, _safe_ln_inv(self.ema_log[tool]))
        return self.prior_s.get(tool, self.prior_s["*"])

    def predict_return(self, tool: str, now: Optional[float] = None) -> float:
        return (now or time.monotonic()) + self.mean_gap_s(tool)

    # ---- online refinement ----

    def observe(self, tool: str, elapsed_s: float):
        obs = max(1e-3, elapsed_s)
        prior = self.prior_s.get(tool, self.prior_s["*"])
        self.ema_log[tool] = (
            self.ema_log.get(tool, _safe_ln(prior)) * (1 - _EMA_DECAY)
            + _safe_ln(obs) * _EMA_DECAY
        )
        self.n_observed += 1


def _safe_ln(x: float) -> float:
    import math

    return math.log(max(1e-3, x))


def _safe_ln_inv(log_x: float) -> float:
    import math

    return math.exp(log_x)


def build_tool_retention_policy(server_args) -> Optional[ToolRetentionPolicy]:
    """Factory for the scheduler's tree-cache wiring (None when disabled)."""
    return ToolRetentionPolicy.from_server_args(server_args)
