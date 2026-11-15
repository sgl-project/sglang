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

import gc
import sys

import pytest

from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Cycle:
    finalized = 0

    def __init__(self) -> None:
        self.self_ref = self

    def __del__(self) -> None:
        _Cycle.finalized += 1


@pytest.fixture(autouse=True)
def _restore_gc_state():
    threshold = gc.get_threshold()
    enabled = gc.isenabled()
    _Cycle.finalized = 0
    yield
    gc.set_threshold(*threshold)
    if enabled:
        gc.enable()
    else:
        gc.disable()


def test_cyclic_garbage_is_not_collected_until_capture_ends() -> None:
    with freeze_gc(enable_cudagraph_gc=False):
        gc.set_threshold(1)
        for _ in range(100):
            _Cycle()
        assert _Cycle.finalized == 0
    assert _Cycle.finalized == 100
    assert gc.isenabled()


def test_gc_disabled_by_the_caller_stays_disabled() -> None:
    gc.disable()
    with freeze_gc(enable_cudagraph_gc=False):
        pass
    assert not gc.isenabled()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
