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
"""The choke-point enforcement scan: unified-pool id-space translation is
owned by exactly two places, and no attention backend may re-grow its own.

Ownership after the read-path refactor:

* READS  — ``KVIndexSource`` (canonical tables / batch views): indices are
  born kernel-facing; backends consume views and never translate.
* WRITES — the ForwardBatch rebind (``apply_unified_kv_loc_rebind``):
  ``out_cache_loc`` arrives kernel-facing at every backend.

Scattered per-backend translation had two demonstrated failure modes, in
opposite directions, both silent (virtual and physical ids share a value
range, so nothing crashes — kernels just read wrong rows): the eager
TARGET_VERIFY branch that FORGOT its translate, and the captured
verify+SWA path that translated TWICE. This scan makes both classes
unrepresentable: a backend cannot call the unified translate surfaces at
all.

Deliberately out of scope, with reasons:

* allocator-internal implementations (``multi_ended_allocator`` /
  ``unified_memory_pool``) — they ARE the mechanism the choke point calls;
* the PD transfer-plane ``translate_kv_indices_for_transfer`` — translates
  for RDMA staging outside the forward path, not for kernels;
* the STATIC SWA pool's legacy full->swa slot map
  (``translate_loc_from_full_to_swa`` on non-unified pools) — a different
  mapping kind with no virtual/physical ambiguity. Its call sites in the
  choke-point backends are count-pinned below so new ones are added
  consciously, not accidentally.

    python3 -m pytest test/registered/unit/layers/attention/test_kv_translate_choke_point.py -v
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
        a backend re-grew its own id-space transition — the design whose two
        failure modes (forgotten translate, duplicated translate) this scan
        exists to prevent. Route reads through KVIndexSource views and
        writes through the ForwardBatch rebind instead."""
        banned = re.compile(r"\.translate_kv_loc(_dense)?\(")
        hits = [
            f"{rel}: {m.group(0)}"
            for rel, src in _iter_sources()
            for m in banned.finditer(src)
        ]
        self.assertEqual(hits, [])

    def test_no_translate_capability_probing(self):
        """No backend probes an allocator for translate capability — the
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


class TestStaticSwaTranslateSitesPinned(CustomTestCase):
    # The static-pool legacy full->swa map call counts in the backends that
    # ALSO serve the unified pool. Adding a site here must be a conscious
    # decision (is it reachable with a unified pool? then it belongs in the
    # choke point instead) — update the count together with that reasoning.
    _PINNED = {
        "triton_backend.py": 6,
        "flashattention_backend.py": 12,
        "flashinfer_backend.py": 4,
        "trtllm_mha_backend.py": 2,
    }

    def test_choke_point_backend_site_counts(self):
        counts = {}
        for rel, src in _iter_sources():
            if rel in self._PINNED:
                counts[rel] = src.count("translate_loc_from_full_to_swa")
        self.assertEqual(counts, self._PINNED)


if __name__ == "__main__":
    unittest.main()
