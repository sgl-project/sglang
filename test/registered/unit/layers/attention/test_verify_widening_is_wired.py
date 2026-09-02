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
"""The widened verify table must reach the metadata updater.

BUG REGRESSION. A whole-sequence verify reads `[committed prefix | drafts]` back
out of the pool, so its read table has to be built with `seq_len_delta` set.
Both flashinfer backends computed that widened table into a local and then
handed the updater the UN-widened per-batch view instead -- the widening was
computed and dropped on the floor. Nothing failed loudly: a dead store compiles,
type-checks, and at page_size 256 the un-widened `ceil(seq/ps)` columns still
happen to cover the few draft tokens. Only at page_size 1, where every draft
token needs its own column, did the drafts read stale entries -- DFLASH's accept
length fell from 6.20 to 1.37 and NEXTN kept its accept length but returned
wrong tokens (gsm8k 0.905 -> 0.730).

A SECOND, distinct shape reached GPUs (eval_593): the eager verify path
widened correctly while the CAPTURED path -- the one a cuda-graph replay runs --
built its read table with no `seq_len_delta` at all. There is no dead store to
find there, so the first guard cannot see it. With `cg on`, DFLASH's draft block
forward (which runs as TARGET_VERIFY on the draft runner) replayed against a
table filled only to `seq_lens`, while the CSR builder widened the lens by
`draft_token_num` -- every row's draft tail read stale entries. Accept length
1.45 vs 7.0 on baseline, on both flashinfer backends, whatever the target
backend was.

Both guards are structural because both failures are: a widened table that is
never read, and a captured build that never widens, each mean the widening
cannot reach a kernel. That is cheap to check exactly, and it stays red for the
whole class of rewiring mistakes rather than the one spelling each bug took.

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
    """(function, variable, lineno) for each widened table that is never read."""
    tree = ast.parse(path.read_text())
    dead = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == _WIDENING_CALL:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assigned[target.id] = node.lineno
        if not assigned:
            continue
        loaded = {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        for name, lineno in assigned.items():
            if name not in loaded:
                dead.append((fn.name, name, lineno))
    return dead


# The captured/replay builders. A backend reaches them per cuda-graph replay,
# so a build here that does not widen truncates every replayed verify.
_CAPTURED_BUILDERS = ("out_graph", "cuda_graph_metadata")
_BUILD_CALL = "build_index_table"


def _unwidened_captured_builds(path: pathlib.Path):
    """(function, lineno) for each captured-path build that omits the delta."""
    tree = ast.parse(path.read_text())
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(marker in fn.name for marker in _CAPTURED_BUILDERS):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == _BUILD_CALL:
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
