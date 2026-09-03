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
"""A widened verify table must reach the kernel that reads past seq_lens.

BUG REGRESSION, two shapes. A whole-sequence verify reads
`[committed prefix | drafts]` back out of the pool, so a page-table consumer
has to build its read table with `seq_len_delta` set. Both shapes are a
widening that is computed but cannot arrive.

FIRST: the widened table is bound to a local and then the UN-widened per-batch
view is handed on instead -- the widening dropped on the floor. Nothing fails
loudly: a dead store compiles, type-checks, and at page_size 256 the un-widened
`ceil(seq/ps)` columns still happen to cover the few draft tokens. Only at
page_size 1, where every draft token needs its own column, do the drafts read
stale entries -- observed as accept length collapsing (6.20 -> 1.37) and, on a
run that kept its accept length, wrong tokens (gsm8k 0.905 -> 0.730).

SECOND, and invisible to the first guard: the eager verify path widens
correctly while the CAPTURED path -- the one a cuda-graph replay runs -- builds
its table with no `seq_len_delta` at all. There is no dead store to find. The
replayed verify reads a table filled only to `seq_lens` while the CSR builder
widened the lens by `draft_token_num`, so every row's draft tail is stale.

Both guards are structural, because both failures are: a widened table nothing
reads, and a captured build that never widens, each mean the widening cannot
reach a kernel. That is cheap to check exactly, and it stays red for the whole
class of rewiring mistakes rather than the one spelling each bug took.

Scope note: this guards the PAGE-TABLE consumers (fa3 and the MLA family),
which gather from a rectangle whose live prefix can fall short. The CSR
consumers gather through `fill_packed_read_stream` from already-widened lens,
so they materialize no table to under-fill and no widening to drop.

    python -m pytest test/registered/unit/layers/attention/test_verify_widening_is_wired.py -v
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Every backend that builds a read table through the KV-index translator.
_BACKEND_DIR = (
    pathlib.Path(__file__).resolve().parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
)
_WIDENING_CALL = "widened_index_table"


def _dead_widenings(path: pathlib.Path):
    """(function, variable, lineno) for each widened table that is never read.

    A widening lands in one of two shapes. Bound to a LOCAL, it has to be read
    again in the same function or it went nowhere. PUBLISHED to an attribute
    (`self.spec_kv_view = ...`, how the flashinfer backends hand the table to
    updaters that take no table argument) it escapes the function, so the
    consumption to look for is a read of that attribute somewhere in the
    module -- an attribute nothing loads is the same dead store wearing a
    different spelling.
    """
    tree = ast.parse(path.read_text())
    attr_loads = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }
    dead = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        locals_, published = {}, {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == _WIDENING_CALL:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            locals_[target.id] = node.lineno
                        elif isinstance(target, ast.Attribute):
                            published[target.attr] = node.lineno
        if not locals_ and not published:
            continue
        loaded = {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        for name, lineno in locals_.items():
            if name not in loaded:
                dead.append((fn.name, name, lineno))
        for name, lineno in published.items():
            if name not in attr_loads:
                dead.append((fn.name, f"self.{name}", lineno))
    return sorted(dead, key=lambda d: d[2])


# The captured/replay builders. A backend reaches them per cuda-graph replay,
# so a build here that does not widen truncates every replayed verify.
_CAPTURED_BUILDERS = ("out_graph", "cuda_graph_metadata")
_BUILD_CALL = "build_index_table"


def _table_builders(tree: ast.Module):
    """`build_index_table`, plus this module's own helpers that forward a
    `seq_len_delta` to it.

    A captured builder need not call the translator directly -- it may slice
    its batch to `bs` first and go through a helper. That hop must not be a
    place the delta can be dropped, so the helper is held to the same rule as
    the call it wraps.
    """
    builders = {_BUILD_CALL}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == _BUILD_CALL
                and any(kw.arg == "seq_len_delta" for kw in node.keywords)
            ):
                builders.add(fn.name)
                break
    return builders


def _unwidened_captured_builds(path: pathlib.Path):
    """(function, lineno) for each captured-path build that omits the delta."""
    tree = ast.parse(path.read_text())
    builders = _table_builders(tree)
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(marker in fn.name for marker in _CAPTURED_BUILDERS):
            continue
        if fn.name in builders:
            continue  # the helper itself; its own call carries the delta
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in builders:
                if not any(kw.arg == "seq_len_delta" for kw in node.keywords):
                    found.append((fn.name, node.lineno))
    return found


class TestVerifyWideningIsWired(unittest.TestCase):
    def test_widened_table_is_consumed(self):
        """A widened table that is never read cannot reach a kernel, so the
        verify tail silently falls back to the un-widened prefix."""
        offenders = []
        for path in sorted(_BACKEND_DIR.glob("*.py")):
            if _WIDENING_CALL not in path.read_text():
                continue
            for fn, name, lineno in _dead_widenings(path):
                offenders.append(f"{path.name}:{lineno} {fn}() -> '{name}'")
        self.assertEqual(
            offenders,
            [],
            "widened verify table computed but never read (the updater gets the "
            "un-widened view, truncating the draft tail at page_size 1): "
            + "; ".join(offenders),
        )

    def test_captured_build_widens_like_the_eager_one(self):
        """A backend that widens its EAGER verify read must widen the CAPTURED
        one by the same delta, or every cuda-graph replay truncates the draft
        tail while the CSR builder still widens the lens."""
        offenders = []
        for path in sorted(_BACKEND_DIR.glob("*.py")):
            if _WIDENING_CALL not in path.read_text():
                continue
            for fn, lineno in _unwidened_captured_builds(path):
                offenders.append(f"{path.name}:{lineno} {fn}()")
        self.assertEqual(
            offenders,
            [],
            "captured verify build omits seq_len_delta while the eager path "
            "widens (replay reads a stale draft tail): " + "; ".join(offenders),
        )

    def test_guard_detects_an_unwidened_captured_build(self):
        """The detector must catch the shape it guards."""
        src = (
            "class B:\n"
            "    def init_forward_metadata_out_graph(self, fb):\n"
            "        kv_view = self.t.build_index_table(\n"
            "            req_pool_indices=fb.req_pool_indices, seq_lens=fb.seq_lens\n"
            "        )\n"
            "        self.u.update(kv_view=kv_view)\n"
        )
        tmp = pathlib.Path(self.id().replace(".", "_") + ".py")
        tmp.write_text(src)
        try:
            self.assertEqual(
                _unwidened_captured_builds(tmp),
                [("init_forward_metadata_out_graph", 3)],
            )
        finally:
            tmp.unlink()

    def test_guard_accepts_a_widened_captured_build(self):
        """...and must NOT fire once the delta is passed, else it is unfixable."""
        src = (
            "class B:\n"
            "    def init_forward_metadata_out_graph(self, fb):\n"
            "        kv_view = self.t.build_index_table(\n"
            "            seq_lens=fb.seq_lens, seq_len_delta=4\n"
            "        )\n"
        )
        tmp = pathlib.Path(self.id().replace(".", "_") + ".py")
        tmp.write_text(src)
        try:
            self.assertEqual(_unwidened_captured_builds(tmp), [])
        finally:
            tmp.unlink()

    def test_guard_detects_an_unwidened_helper_hop(self):
        """A captured builder that reaches the translator through a helper
        must still carry the delta across the hop."""
        src = (
            "class B:\n"
            "    def _spec_table(self, seq_lens, seq_len_delta):\n"
            "        return self.t.build_index_table(\n"
            "            seq_lens=seq_lens, seq_len_delta=seq_len_delta\n"
            "        )\n"
            "    def init_forward_metadata_out_graph(self, fb):\n"
            "        self.kv_view = self._spec_table(seq_lens=fb.seq_lens)\n"
        )
        tmp = pathlib.Path(self.id().replace(".", "_") + ".py")
        tmp.write_text(src)
        try:
            self.assertEqual(
                _unwidened_captured_builds(tmp),
                [("init_forward_metadata_out_graph", 7)],
            )
        finally:
            tmp.unlink()

    def test_guard_detects_a_published_widening_nothing_reads(self):
        """Publishing to an attribute is only a consumption if something reads
        the attribute back -- otherwise it is a dead store with extra steps."""
        src = (
            "class B:\n"
            "    def init_forward_metadata(self, fb):\n"
            "        self.spec_kv_view = self.t.widened_index_table(fb, 4)\n"
        )
        tmp = pathlib.Path(self.id().replace(".", "_") + ".py")
        tmp.write_text(src)
        try:
            self.assertEqual(
                _dead_widenings(tmp),
                [("init_forward_metadata", "self.spec_kv_view", 3)],
            )
        finally:
            tmp.unlink()

    def test_guard_accepts_a_published_widening_that_is_read(self):
        """...and must NOT fire once a consumer reads it, else it is unfixable."""
        src = (
            "class B:\n"
            "    def init_forward_metadata(self, fb):\n"
            "        self.spec_kv_view = self.t.widened_index_table(fb, 4)\n"
            "class U:\n"
            "    def call_begin_forward(self):\n"
            "        return self.backend.spec_kv_view\n"
        )
        tmp = pathlib.Path(self.id().replace(".", "_") + ".py")
        tmp.write_text(src)
        try:
            self.assertEqual(_dead_widenings(tmp), [])
        finally:
            tmp.unlink()

    def test_guard_detects_a_dead_widening(self):
        """The detector itself must catch the shape it guards -- otherwise it
        would pass green forever after a refactor of the walk above."""
        src = (
            "class B:\n"
            "    def init_forward_metadata(self, fb):\n"
            "        kv_view = self.t.index_table_for_batch(fb)\n"
            "        index_table = self.t.widened_index_table(fb, seq_len_delta=4)\n"
            "        self.u.update(kv_view=kv_view)\n"
        )
        tmp = pathlib.Path(self.id().replace(".", "_") + ".py")
        tmp.write_text(src)
        try:
            self.assertEqual(
                _dead_widenings(tmp), [("init_forward_metadata", "index_table", 4)]
            )
        finally:
            tmp.unlink()


if __name__ == "__main__":
    unittest.main()
