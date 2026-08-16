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
"""Source pins for the two id-space bugs the read choke point makes
inexpressible in the Triton backend's TARGET_VERIFY paths.

Under the unified pool every id in ``req_to_token`` is virtual, and both
historical failure modes were per-branch translate bookkeeping:

* the eager TARGET_VERIFY branch filled kv_indices from raw ``req_to_token``
  and never translated (virtual prefix ids reached the verify kernel), while
  decode and extend each carried their own post-fill translate;
* the captured TARGET_VERIFY + hybrid-SWA path translated the window buffer
  TWICE (``update_sliding_window_buffer``'s in-function translate plus the
  per-replay refill hook) because only the decode builder passed the
  ``skip_full_to_swa_translation`` flag.

The fix is structural: metadata builders read ``KVIndexSource`` views whose
canonical tables are born kernel-facing, so there is no fill-then-translate
sequence left to forget or duplicate. These tests pin that structure at the
source level; both are RED before this change.

    python3 -m pytest test/registered/unit/layers/attention/test_triton_unified_verify_sources.py -v
"""

import ast
import inspect
import textwrap
import unittest

from sglang.srt.layers.attention import triton_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagerVerifyBornTranslated(CustomTestCase):
    def test_fill_reads_the_view_not_raw_req_to_token(self):
        """The shared fill helper consumes a KVIndexBatchView (kernel-facing
        table under the unified pool) — not the raw virtual req_to_token, and
        it carries no translate of its own."""
        src = inspect.getsource(
            triton_backend.TritonAttnBackend._fill_kv_indptr_and_indices
        )
        self.assertIn("kv_view", src)
        self.assertNotIn("self.req_to_token", src)
        self.assertNotIn("_translate_kv_loc", src)

    def test_every_eager_fill_call_site_passes_a_view(self):
        """All eager metadata branches (decode, target_verify, extend) hand
        the fill a view — none can bypass the choke point. The historical bug
        was exactly one branch (target_verify) filling from raw req_to_token
        with no translate, so the strongest structural pin is: NO raw
        req_to_token reference remains in the eager builder at all."""
        src = inspect.getsource(triton_backend.TritonAttnBackend.init_forward_metadata)
        self.assertEqual(
            src.count("self.req_to_token"),
            0,
            "an eager metadata branch reads raw (virtual) req_to_token — "
            "fills must come from KVIndexSource views",
        )
        fills = src.count("_fill_kv_indptr_and_indices(")
        self.assertGreaterEqual(fills, 3, "decode/verify/extend fill sites")
        self.assertGreaterEqual(
            src.count("self._kv_index_view("),
            fills,
            "fewer view builds than fill sites — some fill bypasses the view",
        )

    def test_no_post_fill_translate_remains(self):
        """The per-branch post-fill translates are gone: indices are born
        kernel-facing, not translated as a separate (forgettable) step."""
        src = inspect.getsource(triton_backend.TritonAttnBackend.init_forward_metadata)
        self.assertNotIn("_translate_kv_loc", src)


class TestCapturedVerifyTranslatesOnce(CustomTestCase):
    def test_window_builder_translates_only_for_static_pools(self):
        """Under the unified pool the window fill reads the view's swa table
        (already swa-kernel-facing) with NO translate; the one remaining
        translate is the STATIC-pool legacy map, and it must sit inside a
        branch guarded by `not kv_view.kernel_facing`. The historical
        captured-verify double-translate needed a second, unconditional owner
        (the replay hook) plus a skip-flag both callers had to remember — both
        are gone.

        Parses the function instead of matching text: any call to the map that
        is not dominated by that guard fails, whatever the call is spelled or
        wrapped like."""
        src = inspect.getsource(triton_backend.update_sliding_window_buffer)
        self.assertNotIn("skip_full_to_swa_translation", src)
        self.assertIn("swa_read_table", src)

        tree = ast.parse(textwrap.dedent(src))

        def guards_on_static_pool(node):
            return "not kv_view.kernel_facing" in ast.unparse(node.test)

        def calls_the_map(node):
            return any(
                isinstance(n, ast.Attribute)
                and n.attr == "translate_loc_from_full_to_swa"
                for n in ast.walk(node)
            )

        guarded, unguarded = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and guards_on_static_pool(node):
                guarded.extend(
                    n
                    for n in ast.walk(node)
                    if isinstance(n, ast.Attribute)
                    and n.attr == "translate_loc_from_full_to_swa"
                )
        all_calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and n.attr == "translate_loc_from_full_to_swa"
        ]
        unguarded = [n for n in all_calls if n not in guarded]
        self.assertEqual(
            len(all_calls), 1, "expected exactly one static-pool translate"
        )
        self.assertEqual(
            unguarded,
            [],
            "every full->swa translate here must be gated on the view NOT being "
            "kernel-facing — an unconditional translate re-creates the "
            "captured-verify double-translate",
        )

    def test_verify_builder_threads_the_view(self):
        """The captured verify builder passes the view into the window fill —
        symmetric with decode, so the flag asymmetry that caused the verify
        double-translate cannot come back."""
        src = inspect.getsource(
            triton_backend.TritonAttnBackend._update_target_verify_buffers
        )
        self.assertIn("kv_view", src)

    def test_replay_hook_is_write_only(self):
        """The per-replay hook refills WRITE locs only; the read half (the
        second translate owner) is deleted."""
        src = inspect.getsource(
            triton_backend.TritonAttnBackend._refill_cuda_graph_write_locs
        )
        self.assertNotIn("cuda_graph_kv_indices", src)
        self.assertNotIn("cuda_graph_window_kv_indices", src)
        self.assertNotIn("translate_loc_from_full_to_swa(", src)


if __name__ == "__main__":
    unittest.main()
