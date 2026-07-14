import sys

import pytest

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    backend_supports_multimodal_prefill_cuda_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_tc_piecewise_is_the_only_multimodal_prefill_backend():
    # TC PCG keeps the VLM wrapper eager and replays only the decoder. The
    # other backends capture a layer body and cannot merge vision embeddings.
    assert backend_supports_multimodal_prefill_cuda_graph(Backend.TC_PIECEWISE)
    assert not backend_supports_multimodal_prefill_cuda_graph(Backend.BREAKABLE)
    assert not backend_supports_multimodal_prefill_cuda_graph(Backend.FULL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
