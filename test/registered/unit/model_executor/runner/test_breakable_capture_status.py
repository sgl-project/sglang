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

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_musa_uses_portable_stream_capture_status() -> None:
    stream = object()
    device_module = SimpleNamespace(
        stream=lambda value: nullcontext(),
        is_current_stream_capturing=lambda: True,
    )

    with (
        patch.object(breakable_cuda_graph, "is_hip", return_value=False),
        patch.object(breakable_cuda_graph, "is_musa", return_value=True),
        patch.object(breakable_cuda_graph, "_is_xpu", False),
        patch.object(
            breakable_cuda_graph, "get_device_module", return_value=device_module
        ),
    ):
        assert breakable_cuda_graph._is_stream_capturing(stream)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
