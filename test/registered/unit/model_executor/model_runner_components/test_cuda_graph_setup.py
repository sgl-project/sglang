import sys

import pytest

from sglang.srt.model_executor.cuda_graph_config import Phase
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    should_skip_auto_prefill_cuda_graph_for_memory,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def test_auto_prefill_cuda_graph_memory_gate():
    assert should_skip_auto_prefill_cuda_graph_for_memory(3.99, set())
    assert not should_skip_auto_prefill_cuda_graph_for_memory(4.0, set())


def test_explicit_prefill_backend_bypasses_memory_gate():
    assert not should_skip_auto_prefill_cuda_graph_for_memory(
        0.0, {(Phase.PREFILL, "backend")}
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
