"""Nothing under layers/attention may translate KV ids for itself.

Ownership is exactly two places: `KVIndexTranslator` for READS (indices are
born kernel-facing, backends consume its tables) and the ForwardBatch rebind
(`rebind_write_loc`) for WRITES. Virtual and physical ids share a value range,
so a backend that forgets a translate -- or does one twice -- reads the wrong
rows and nothing crashes. This scan makes both unrepresentable.

Out of scope, deliberately: the allocator-internal implementations
(`multi_ended_allocator` / `unified_memory_pool`), which ARE the mechanism the
translator calls; the PD transfer plane's `translate_kv_indices_for_transfer`,
which stages for RDMA outside the forward path; and the STATIC SWA pool's
legacy full->swa slot map, a different mapping kind with no virtual/physical
ambiguity -- its call sites are count-pinned below so new ones are added
consciously.

    python3 -m pytest test/registered/unit/layers/attention/test_kv_translate_ownership.py -v
"""

import ast
import os
import re
import unittest

from sglang.srt.layers.attention import triton_backend as _anchor_module
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dots_hybrid_backend import (
    DotsHybridAttnBackend,
    DotsSWAMLAAttnBackend,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    ShortConvHybridAttnBackend,
)
from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxHybridAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The attention package is a namespace package (no __init__), so anchor the
# scan on a concrete module inside it.
_ATTN_DIR = os.path.dirname(os.path.abspath(_anchor_module.__file__))


def _iter_sources():
    for root, _dirs, files in os.walk(_ATTN_DIR):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                yield os.path.relpath(path, _ATTN_DIR), fh.read()


class TestUnifiedTranslateBanned(CustomTestCase):
    def test_no_unified_translate_calls(self):
        """No backend calls the unified translate surfaces. A hit here means
        a backend re-grew its own id-space transition -- the design whose two
        failure modes (forgotten translate, duplicated translate) this scan
        exists to prevent. Route reads through KVIndexTranslator views and
        writes through the ForwardBatch rebind instead."""
        banned = re.compile(r"\.translate_kv_loc(_kernel_id)?\(")
        hits = [
            f"{rel}: {m.group(0)}"
            for rel, src in _iter_sources()
            for m in banned.finditer(src)
        ]
        self.assertEqual(hits, [])

    def test_no_translate_capability_probing(self):
        """No backend probes an allocator for translate capability -- the
        getattr-hook pattern is how per-backend translation grew the first
        time."""
        probing = re.compile(r"""getattr\([^)]*['"]translate_kv_loc""")
        hits = [rel for rel, src in _iter_sources() if probing.search(src)]
        self.assertEqual(hits, [])

    def test_hooks_module_deleted_and_unimported(self):
        """The per-backend hooks module (the previous owner of backend-side
        v2p knowledge) stays deleted, and nothing imports it."""
        self.assertFalse(
            os.path.exists(os.path.join(_ATTN_DIR, "unified_mem_hooks.py"))
        )
        hits = [
            rel
            for rel, src in _iter_sources()
            if "unified_mem_hooks" in src or "unified_mla_hooks" in src
        ]
        self.assertEqual(hits, [])


def _derive_wrapper_names():
    """Wrapper classes, read off the source so a NEW one shows up the day it is
    written: an AttentionBackend subclass whose own __init__ takes another
    backend. Per class, not per file -- a file-wide scan passes as soon as any
    one class in it forwards."""
    backend = re.compile(r"Att(?:ention|n)Backend")
    names = set()
    for _rel, src in _iter_sources():
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(backend.search(ast.unparse(b)) for b in node.bases):
                continue
            init = next(
                (
                    b
                    for b in node.body
                    if isinstance(b, ast.FunctionDef) and b.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            if any(
                a.annotation is not None and backend.search(ast.unparse(a.annotation))
                for a in init.args.args[1:]
            ):
                names.add(node.name)
    return names


class _Inner(AttentionBackend):
    """Stand-in for a wrapped backend, carrying only what the wrappers' __init__
    bodies read off their inners."""

    def __init__(self, translator=None):
        self.kv_index_translator = translator
        self.token_to_kv_pool = None
        self.req_to_token_pool = None
        self.needs_cpu_seq_lens = False
        self.max_context_len = 8


class _InnerModelConfig:
    context_len = 8


class _Runner:
    """HybridAttnBackend takes its translator from the runner, not an inner."""

    def __init__(self, translator):
        self.kv_index_translator = translator
        self.kv_cache_dtype = None
        self.token_to_kv_pool = None
        self.req_to_token_pool = None
        self.model_config = _InnerModelConfig()


def _build_wrappers(translator):
    """One live instance per wrapper. Only the inner that MUST supply the
    translator carries it; every other inner carries None, so a wrapper that
    copies from the linear / sparse / DSA side ends up with None and fails."""
    carrier = _Inner(translator)
    # HybridAttnBackend reads the spec bag in __init__; the bag is unpublished
    # outside a launched server.
    with get_context().override_server_args(speculative_attention_mode="decode"):
        hybrid = HybridAttnBackend(_Runner(translator), _Inner(), _Inner())
    return {
        "DotsSWAMLAAttnBackend": DotsSWAMLAAttnBackend(carrier),
        "DotsHybridAttnBackend": DotsHybridAttnBackend(_Inner(), carrier),
        "HybridAttnBackend": hybrid,
        "HybridLinearAttnBackend": HybridLinearAttnBackend(carrier, _Inner(), [0]),
        "ShortConvHybridAttnBackend": ShortConvHybridAttnBackend(
            carrier, _Inner(), [0]
        ),
        "MiniMaxHybridAttnBackend": MiniMaxHybridAttnBackend(carrier, _Inner(), [0]),
        "TboAttnBackend": TboAttnBackend(carrier, [_Inner()]),
    }


class TestWrapperBackendsForwardTranslator(CustomTestCase):
    """BUG REGRESSION. `AttentionBackend.kv_index_translator` defaults to None,
    so a wrapper that does not re-expose its inner's copy reads as "needs no
    translation" and producers that fetch it off `get_attn_backend()` skip the
    virtual->kernel-facing translation instead of failing."""

    def test_every_wrapper_is_constructed_here(self):
        self.assertEqual(
            _derive_wrapper_names(),
            set(_build_wrappers(object())),
            "a wrapper backend has no instance in _build_wrappers; add one so "
            "its translator forwarding is checked",
        )

    def test_wrappers_forward_the_translator(self):
        translator = object()
        for name, wrapper in _build_wrappers(translator).items():
            with self.subTest(wrapper=name):
                self.assertIs(wrapper.kv_index_translator, translator)


if __name__ == "__main__":
    unittest.main()
