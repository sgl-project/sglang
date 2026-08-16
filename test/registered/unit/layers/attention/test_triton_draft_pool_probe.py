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
"""The Triton backend's id translate must be gated on POOL ownership.

A speculative DRAFT runner is handed the TARGET's allocator — it shares the
target's virtual id space and its `req_to_token` — but owns a SEPARATE KV
pool, direct-indexed by those virtual ids and sized to that virtual space.

So "is there an allocator with a translate?" is the wrong question. The right
one is "does this runner's pool hold the ids that allocator mints?". Probing
the allocator alone translates the draft's indices into the TARGET's
kernel-facing space and then applies them to a pool expecting the raw virtual
id: out of bounds on both the read and the write rail. That is what makes
DSPARK + --enable-unified-memory fault.

The failure mode this guards is a predicate degrading to always-true — the
probe silently going back to "allocator has the attribute", which every
non-speculative test would still pass. Checked at the source level because
`TritonAttnBackend.__init__` needs a whole ModelRunner to instantiate; the
behavioural twin lives with the read-path choke point
(`test_kv_index_source.py::test_runner_with_its_own_pool_is_disabled`).

CPU-only.

    python -m pytest test/registered/unit/layers/attention/test_triton_draft_pool_probe.py -v
"""

import ast
import inspect
import textwrap
import unittest

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _init_tree():
    src = textwrap.dedent(inspect.getsource(TritonAttnBackend.__init__))
    return ast.parse(src)


class TestTritonDraftPoolProbe(CustomTestCase):
    def test_translate_is_assigned_only_under_an_ownership_test(self):
        """RED before the fix: `__init__` assigned `_translate_kv_loc`
        unconditionally from the allocator, so a draft runner sharing the
        target's allocator got the target's translate."""
        tree = _init_tree()

        # Every assignment to self._translate_kv_loc, and whether some
        # enclosing `if` compares a kvcache against this runner's pool.
        unguarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.dump(node.test)
            owns = "get_kvcache" in test_src and "token_to_kv_pool" in test_src
            if not owns:
                continue
            # Found the ownership branch: both arms must set the attribute.
            arms = []
            for arm in (node.body, node.orelse):
                names = [
                    t.attr
                    for stmt in arm
                    for t in ast.walk(stmt)
                    if isinstance(t, ast.Attribute) and isinstance(t.ctx, ast.Store)
                ]
                arms.append("_translate_kv_loc" in names)
            if all(arms):
                return  # gated, both arms resolved
            unguarded.append(ast.dump(node.test))

        self.fail(
            "TritonAttnBackend.__init__ must resolve `_translate_kv_loc` inside "
            "a branch that compares `allocator.get_kvcache()` against this "
            "runner's `token_to_kv_pool`, with BOTH arms assigning it. "
            f"Ownership branches seen: {unguarded or 'none'}"
        )

    def test_non_owning_runner_gets_no_translate(self):
        """The draft arm must resolve to None, not to a fallback probe: a
        draft runner translating at all is the bug."""
        tree = _init_tree()

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "get_kvcache" not in ast.dump(node.test):
                continue
            self.assertTrue(node.orelse, "the non-owning arm must be explicit")
            dumped = ast.dump(ast.Module(body=node.orelse, type_ignores=[]))
            self.assertIn("_translate_kv_loc", dumped)
            self.assertIn(
                "Constant(value=None)",
                dumped,
                "a non-owning runner must get None, never a fallback translate",
            )
            return
        self.fail("no ownership branch found in TritonAttnBackend.__init__")

    def test_the_translate_is_resolved_by_type_not_by_duck_probe(self):
        """The predicate must narrow on the allocator TYPE and then reach the
        method directly. A `getattr(alloc, "translate_...", None)` chain answers
        "does this object have the attribute?", which is true for a draft
        runner's borrowed allocator and silently true again for any future
        allocator that grows the name — the two ways this probe has already
        gone wrong."""
        tree = _init_tree()

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "get_kvcache" not in ast.dump(node.test):
                continue
            self.assertIn(
                "isinstance",
                ast.dump(node.test),
                "the ownership test must also narrow the allocator by type",
            )
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            self.assertNotIn(
                "getattr",
                body,
                "resolve the translate by direct attribute access, not a duck probe",
            )
            self.assertIn("translate_kv_loc_dense", body)
            return
        self.fail("no ownership branch found in TritonAttnBackend.__init__")


if __name__ == "__main__":
    unittest.main()
