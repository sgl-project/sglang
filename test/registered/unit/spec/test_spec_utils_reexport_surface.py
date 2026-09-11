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
"""`spec_utils` is a RE-EXPORT hub: the speculative Triton kernels live in
`sglang.kernels.ops.speculative.*` and are forwarded here as `X as X` so the
srt modules import one name from one place.

BUG REGRESSION. A new kernel was added to `cache_locs` and imported by
`triton_backend` from `spec_utils`, but never forwarded there. Nothing local
caught it: the name is imported and used, so ruff's F401/F821 stay quiet, and
`py_compile` only parses. It surfaced as an ImportError at server boot that
took down EVERY server using the Triton backend, spec-off included, because the
import is at module scope.

This walks the REAL consumers (an AST scan of srt, not a hand-kept list) and
checks each imported name resolves, so a forgotten forward fails here instead
of on a GPU.

    python -m pytest test/registered/unit/spec/test_spec_utils_reexport_surface.py -v
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HUB = "sglang.srt.speculative.spec_utils"


def _srt_root() -> Path:
    """The installed `sglang/srt` tree.

    Via the PARENT package: `sglang.srt` ships no `__init__.py`, so it is a PEP
    420 namespace package whose `__file__` is None. This is the idiom the other
    srt-walking tests use (test_server_args_namespaces.py and friends).
    """
    import sglang

    return Path(sglang.__file__).resolve().parent / "srt"


def _consumers():
    """(module path, imported names) for every srt module importing from the hub."""
    root = _srt_root()
    assert root.is_dir(), f"srt tree not found at {root}"
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            if node.module == _HUB:
                yield path, tuple(a.name for a in node.names if a.name != "*")


class TestSpecUtilsReexportSurface(CustomTestCase):
    def test_every_imported_name_resolves(self):
        import sglang.srt.speculative.spec_utils as hub

        consumers = list(_consumers())
        # A scan that finds nothing would pass vacuously forever.
        self.assertGreater(len(consumers), 5, "AST scan found no spec_utils consumers")

        missing = [
            f"{path.name} imports {name!r}"
            for path, names in consumers
            for name in names
            if not hasattr(hub, name)
        ]
        self.assertEqual(missing, [], f"{_HUB} does not export: {missing}")

    def test_the_forwarded_kernels_are_the_kernel_module_objects(self):
        """A forward must alias the kernel itself; a same-named local shim
        would pass the attribute check above while shadowing the real one."""
        import sglang.kernels.ops.speculative.cache_locs as cache_locs
        import sglang.srt.speculative.spec_utils as hub

        forwarded = [
            a.asname or a.name
            for node in ast.walk(ast.parse(Path(hub.__file__).read_text()))
            if isinstance(node, ast.ImportFrom)
            and node.module == "sglang.kernels.ops.speculative.cache_locs"
            for a in node.names
        ]
        self.assertIn("generate_draft_decode_kv_indices", forwarded)
        for name in forwarded:
            self.assertIs(getattr(hub, name), getattr(cache_locs, name), name)


if __name__ == "__main__":
    unittest.main()
