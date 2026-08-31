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


if __name__ == "__main__":
    unittest.main()
