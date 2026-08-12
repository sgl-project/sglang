"""Unit tests for the Tail-Optimized LRU eviction strategy (arXiv:2510.15152).

T-LRU reports a node as infinitely old once the conversation holding it is above
its TEL-safe budget, so the ordinary eviction driver drains those nodes first
(the paper's phase 1) and then continues in recency order (phase 2). These tests
exercise that ordering against stub nodes, so they need no GPU, model or CUDA
build and run in well under a second.

evict_policy is loaded straight from its file: importing sglang as a package
pulls in the engine's runtime dependencies, which would make a pure-logic test
require a full install.
"""

import importlib.util
import math
import os
from dataclasses import dataclass, field

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # standalone run without an sglang install; CI parses the
    # registration below from the AST, so the stub changes nothing for CI.
    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVICT_POLICY = os.path.normpath(
    os.path.join(_HERE, "../../../../python/sglang/srt/mem_cache/evict_policy.py")
)
_spec = importlib.util.spec_from_file_location("evict_policy", _EVICT_POLICY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LRUStrategy, TLRUStrategy = _mod.LRUStrategy, _mod.TLRUStrategy

PAGE = 256
XI = 4096
Q_HAT = 1024
DELTA = XI - Q_HAT  # tokens of tail the policy may free


@dataclass
class FakeKey:
    n: int

    def __len__(self):
        return self.n


@dataclass
class FakeNode:
    """A radix node on one conversation's path; only the fields T-LRU reads."""

    depth: int
    key_len: int
    convo_length: int
    last_access_time: float = 0.0
    key: FakeKey = field(init=False)

    def __post_init__(self):
        self.key = FakeKey(self.key_len)


def chain(node_lens, convo_length=None, t0=0.0):
    """Build a root->leaf chain, deepest last.

    convo_length defaults to the full depth, which is the state right after the
    conversation's newest turn was inserted.
    """
    convo = sum(node_lens) if convo_length is None else convo_length
    nodes, depth = [], 0
    for i, n in enumerate(node_lens):
        depth += n
        nodes.append(
            FakeNode(
                depth=depth, key_len=n, convo_length=convo, last_access_time=t0 + i
            )
        )
    return nodes


def strategy(threshold=XI, next_prompt_estimate=Q_HAT):
    return TLRUStrategy(threshold=threshold, next_prompt_estimate=next_prompt_estimate)


def is_tel_safe(s, node):
    return s.get_priority(node)[0] < 0


def freed_tokens(s, nodes):
    return sum(n.key_len for n in nodes if is_tel_safe(s, n))


def test_fresh_conversation_frees_exactly_the_tail_budget():
    s = strategy()
    assert freed_tokens(s, chain([PAGE] * 40)) == DELTA


def test_trimming_stops_after_the_budget():
    """The survivors of a trim must be protected.

    This is the regression guard for deriving the history length from what is
    still resident: that would leave the shortened conversation over budget on
    every subsequent pass and walk it down to nothing.
    """
    s = strategy()
    nodes = chain([PAGE] * 40)
    survivors = [n for n in nodes if not is_tel_safe(s, n)]
    assert freed_tokens(s, survivors) == 0


def test_conversation_under_threshold_is_entirely_free():
    s = strategy()
    short = chain([PAGE] * 4)  # 1024 + Q_hat <= xi, so no caching is needed
    assert freed_tokens(s, short) == sum(n.key_len for n in short)


def test_phase_one_spreads_across_conversations_then_falls_back_to_lru():
    s = strategy()
    old = chain([PAGE] * 40, t0=0.0)
    new = chain([PAGE] * 40, t0=100.0)
    order = sorted(old + new, key=s.get_priority)
    n_safe = 2 * (DELTA // PAGE)

    assert all(is_tel_safe(s, n) for n in order[:n_safe])
    # Both conversations donate their tail, which is what the paper's
    # per-conversation loop exists to produce.
    assert any(n in old for n in order[:n_safe])
    assert any(n in new for n in order[:n_safe])
    # Phase 2 is plain recency.
    assert order[n_safe].last_access_time == old[0].last_access_time
    safe_times = [n.last_access_time for n in order[:n_safe]]
    assert safe_times == sorted(safe_times)


def test_degenerates_to_lru_when_estimate_reaches_threshold():
    pool = chain([PAGE] * 40, t0=0.0) + chain([PAGE] * 40, t0=100.0)
    degenerate = strategy(next_prompt_estimate=XI)
    lru = LRUStrategy()
    assert [id(n) for n in sorted(pool, key=degenerate.get_priority)] == [
        id(n) for n in sorted(pool, key=lru.get_priority)
    ]
    assert freed_tokens(degenerate, chain([PAGE] * 40)) == 0


def test_oversized_tail_node_is_protected_rather_than_partially_freed():
    """Node granularity under-trims instead of over-trimming.

    The paper trims one block at a time; a radix tree can only drop whole leaves,
    so a turn larger than the budget stays put and phase 2 decides its fate.
    """
    s = strategy()
    assert freed_tokens(s, chain([PAGE * 40])) == 0


def test_compacted_branch_keeps_shared_prefix_protected():
    """Context compaction shortens a conversation instead of extending it.

    It occurs in about 2% of turns in the agentic traces we benchmark, and forks a
    shallow branch off a shared ancestor that still carries the deeper branch's
    high-water mark. The ancestor is then measured against a history longer than
    what hangs below it, which must stay conservative: over-protect the shared
    prefix, never free it early.
    """
    s = strategy()
    assert freed_tokens(s, chain([PAGE] * 8, convo_length=200_000)) == 0


def test_budget_clamps_at_zero():
    """A budget below zero must mean nothing needs caching, not wrap around."""
    s = strategy()
    node = FakeNode(depth=PAGE, key_len=PAGE, convo_length=0)
    assert is_tel_safe(s, node)


def test_priority_is_finite_and_orderable():
    """The driver pushes (priority, node) onto a heap, so the keys must compare
    without falling through to comparing nodes."""
    s = strategy()
    a, b = chain([PAGE] * 4)[:2]
    for node in (a, b):
        flag, when = s.get_priority(node)
        assert flag in (-1, 0)
        assert math.isfinite(when)
    assert (s.get_priority(a) < s.get_priority(b)) or (
        s.get_priority(b) < s.get_priority(a)
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK   {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(
        "\n" + ("TLRU_TESTS_OK" if not failures else f"TLRU_TESTS_FAILED ({failures})")
    )
    raise SystemExit(1 if failures else 0)
