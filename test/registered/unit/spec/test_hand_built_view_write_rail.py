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
"""A hand-built ForwardBatch view must carry the iteration's plan.

The speculative cuda-graph runners do not hand a real ForwardBatch to
`init_forward_metadata_out_graph`; they build a `SimpleNamespace` view over
their capture-stable buffers. The backends read the iteration's ids through
the view's `kv_loc_plan`: its read table on a translating pool.

A view that omits the plan crashes on a unified pool, and passes on a static
one, whose reads never touch the plan -- which is why the default builds
never catch it. A view that names it None is the same omission in disguise.

Several runners build such views, so this guards the shape rather than the
sites: any `SimpleNamespace` view that reaches
`init_forward_metadata_out_graph` must name `kv_loc_plan`.

CPU-only, no torch.

    python -m pytest test/registered/unit/spec/test_hand_built_view_write_rail.py -v
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SPEC_DIR = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "speculative"
)
_HOOK = "init_forward_metadata_out_graph"
_RAIL = "kv_loc_plan"


def _views_missing_the_rail(path: pathlib.Path):
    """(function, variable, lineno) per hand-built view that drops the rail.

    A view counts when it is a `SimpleNamespace(...)` bound to a name that is
    later passed to the hook in the same function -- that is exactly the shape
    the runners use, and it keeps the walk from flagging namespaces built for
    unrelated purposes.
    """
    tree = ast.parse(path.read_text())
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Names handed to the hook anywhere in this function.
        handed = set()
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == _HOOK
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        handed.add(arg.id)
        if not handed:
            continue
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name != "SimpleNamespace":
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id in handed):
                    continue
                # A literal None is the quiet failure: the hook zeroes the
                # capture buffer, routing every write to the page-0 sink.
                if not any(
                    kw.arg == _RAIL
                    and not (
                        isinstance(kw.value, ast.Constant) and kw.value.value is None
                    )
                    for kw in node.value.keywords
                ):
                    offenders.append((fn.name, target.id, node.lineno))
    return sorted(offenders, key=lambda o: o[2])


class TestHandBuiltViewWriteRail(unittest.TestCase):
    def test_every_hand_built_view_carries_the_rail(self):
        offenders = []
        for path in sorted(_SPEC_DIR.rglob("*.py")):
            for fn, var, lineno in _views_missing_the_rail(path):
                offenders.append(f"{path.name}:{lineno} {fn}() -> {var}")
        self.assertEqual(
            offenders,
            [],
            f"hand-built ForwardBatch view reaches {_HOOK} without {_RAIL}; on a "
            "unified pool its reads have no table to gather from: "
            + "; ".join(offenders),
        )

    def test_detector_catches_a_view_that_drops_the_rail(self):
        """The detector must catch the shape it guards, else it passes green
        forever after a refactor of the walk above."""
        src = (
            "class R:\n"
            "    def replay(self, fb):\n"
            "        v = SimpleNamespace(\n"
            "            batch_size=8, out_cache_loc=self.buffers.out_cache_loc\n"
            "        )\n"
            "        self.backend.init_forward_metadata_out_graph(v)\n"
        )
        self.assertEqual(
            _views_missing_the_rail(self._probe(src)), [("replay", "v", 3)]
        )

    def test_detector_catches_a_view_that_passes_none(self):
        """Naming the plan with None is the omission in disguise."""
        src = (
            "class R:\n"
            "    def replay(self, fb):\n"
            "        v = SimpleNamespace(batch_size=8, kv_loc_plan=None)\n"
            "        self.backend.init_forward_metadata_out_graph(v)\n"
        )
        self.assertEqual(
            _views_missing_the_rail(self._probe(src)), [("replay", "v", 3)]
        )

    def test_detector_accepts_a_view_that_carries_it(self):
        """...and must NOT fire once the rail is named, else it is unfixable."""
        src = (
            "class R:\n"
            "    def replay(self, fb):\n"
            "        v = SimpleNamespace(\n"
            "            batch_size=8,\n"
            "            out_cache_loc=self.buffers.out_cache_loc,\n"
            "            kv_loc_plan=fb.kv_loc_plan,\n"
            "        )\n"
            "        self.backend.init_forward_metadata_out_graph(v)\n"
        )
        self.assertEqual(_views_missing_the_rail(self._probe(src)), [])

    def test_detector_ignores_a_namespace_that_never_reaches_the_hook(self):
        """A SimpleNamespace built for something else is not a view."""
        src = (
            "class R:\n"
            "    def replay(self, fb):\n"
            "        cfg = SimpleNamespace(batch_size=8)\n"
            "        return cfg\n"
        )
        self.assertEqual(_views_missing_the_rail(self._probe(src)), [])

    def _probe(self, src):
        # A temp dir, not the CWD: the checkout may be read-only.
        import shutil
        import tempfile

        scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, scratch, ignore_errors=True)
        tmp = pathlib.Path(scratch) / (self.id().replace(".", "_") + ".py")
        tmp.write_text(src)
        return tmp


if __name__ == "__main__":
    unittest.main()
