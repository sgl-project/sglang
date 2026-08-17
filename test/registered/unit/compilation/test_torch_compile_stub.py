# Copyright 2026 SGLang Team
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
"""Tests for SGLang-local ``torch.compile`` handling with the Triton stub."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

from sglang import _torch_compile
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestTorchCompileStub(CustomTestCase):
    def test_stub_disables_only_sglang_call_site(self):
        original_compile = _torch_compile.torch.compile
        sentinel = object()
        triton_stub = SimpleNamespace(__sglang_stub__=True)

        with (
            patch.dict(sys.modules, {"triton": triton_stub}),
            patch.object(
                _torch_compile.torch, "compile", return_value=sentinel
            ) as call,
        ):
            self.assertIs(
                _torch_compile.sglang_compile("callable", dynamic=True), sentinel
            )
            call.assert_called_once_with("callable", dynamic=True, disable=True)

        self.assertIs(_torch_compile.torch.compile, original_compile)

    def test_real_triton_preserves_torch_compile_arguments(self):
        sentinel = object()
        real_triton = SimpleNamespace(__sglang_stub__=False)

        with (
            patch.dict(sys.modules, {"triton": real_triton}),
            patch.object(
                _torch_compile.torch, "compile", return_value=sentinel
            ) as call,
        ):
            self.assertIs(
                _torch_compile.sglang_compile("callable", dynamic=True), sentinel
            )
            call.assert_called_once_with("callable", dynamic=True)


if __name__ == "__main__":
    import unittest

    unittest.main()
