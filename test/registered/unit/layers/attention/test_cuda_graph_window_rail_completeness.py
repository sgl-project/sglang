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
"""Every cuda-graph forward mode must fill the sliding-window READ rail.

BUG REGRESSION. `_build_cuda_graph_forward_metadata` builds one
`ForwardMetadata` per captured forward mode. The draft-extend branch passed
`window_kv_indices=None` while its buffer-fill helper never built the rail,
so a banded MTP depth (a window layer on the draft) reached the Triton extend
kernel with a null index tensor and died during graph capture as
`AttributeError("'NoneType' object has no attribute 'type'")` inside IR
construction -- a message that names neither the rail nor the mode. The eager
path built it, so only cuda-graph runs failed.

A bare `None` here is the defect: every mode this function serves can carry
window layers, so each must pass the buffer, guarded on the model actually
having a window. This is an AST check over the real function body, so a new
mode branch that forgets the rail fails here rather than on a GPU.

    python -m pytest test/registered/unit/layers/attention/test_cuda_graph_window_rail_completeness.py -v
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BUILDER = "_build_cuda_graph_forward_metadata"
_RAIL = "window_kv_indices"


def _builder_body():
    import sglang.srt.layers.attention.triton_backend as tb

    tree = ast.parse(Path(tb.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _BUILDER:
            return node
    raise AssertionError(f"{_BUILDER} not found in triton_backend.py")


def _rail_args(body):
    """(line, value node) of the window-rail kwarg of each ForwardMetadata."""
    for call in ast.walk(body):
        if not isinstance(call, ast.Call):
            continue
        name = call.func.id if isinstance(call.func, ast.Name) else None
        if name != "ForwardMetadata":
            continue
        for kw in call.keywords:
            if kw.arg == _RAIL:
                yield call.lineno, kw.value
                break
        else:
            yield call.lineno, None  # kwarg omitted entirely


class TestCudaGraphWindowRailCompleteness(CustomTestCase):
    def test_no_captured_mode_passes_a_bare_none_rail(self):
        rails = list(_rail_args(_builder_body()))
        # A scan that finds nothing would pass vacuously forever.
        self.assertGreaterEqual(
            len(rails), 3, "expected one ForwardMetadata per captured forward mode"
        )
        offenders = [
            line
            for line, value in rails
            if value is None
            or (isinstance(value, ast.Constant) and value.value is None)
        ]
        self.assertEqual(
            offenders,
            [],
            f"{_BUILDER}: {_RAIL} is a bare None at line(s) {offenders}; a window "
            "layer in that mode reaches the Triton kernel with a null index "
            "tensor. Pass the cuda-graph buffer, guarded on the sliding-window flag.",
        )


if __name__ == "__main__":
    unittest.main()
