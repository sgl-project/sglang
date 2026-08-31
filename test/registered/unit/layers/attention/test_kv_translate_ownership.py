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

import os
import re
import unittest

from sglang.srt.layers.attention import triton_backend as _anchor_module
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

    def test_wrapper_backends_forward_the_translator(self):
        """BUG REGRESSION. A backend that WRAPS another must re-expose the
        inner backend's `kv_index_translator`.

        `AttentionBackend.kv_index_translator` defaults to None, so a wrapper
        that omits it reads as "needs no translation" rather than failing.
        Producers that fetch the translator off `get_attn_backend()` -- the MLA
        chunked-prefix-cache path does exactly this, under
        `if src is not None` -- then skip translation and hand a kernel raw
        VIRTUAL ids. Nothing raises. On a hybrid MLA model with the unified pool
        and a radix cache (so a shared prefix actually exists to read),
        gsm8k fell from ~0.91 to 0.05-0.31 while the same config on a backend
        that does not take that path stayed correct.

        Derived from the source, not a hand-kept list: a wrapper is any
        AttentionBackend subclass that TAKES another AttentionBackend as a
        constructor parameter, so a NEW wrapper is covered the day it is
        written. Keying on the parameter type rather than the attribute name
        matters -- the wrappers spell their fields `prefill_backend`,
        `decode_backend`, `backend`, `swa_backend` and `dense`, so any
        name-based pattern misses most of them. Plain adapters that do not
        subclass AttentionBackend are out of scope: they never reach
        `get_attn_backend()`, so no producer reads a translator off them."""
        is_backend = re.compile(r"^class \w+\([^)]*Att(?:ention|n)Backend[^)]*\):", re.M)
        wraps = re.compile(r"^\s*\w+\s*:\s*[\"']?\w*Att(?:ention|n)Backend", re.M)
        forwards = re.compile(r"^\s*self\.kv_index_translator\s*=", re.M)
        offenders = [
            rel
            for rel, src in _iter_sources()
            if is_backend.search(src) and wraps.search(src) and not forwards.search(src)
        ]
        self.assertEqual(
            offenders,
            [],
            "wrapper backend does not forward kv_index_translator, so producers "
            "reading it off get_attn_backend() silently skip translation: "
            + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
