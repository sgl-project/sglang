import sys
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import pytest

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=5,
    suite="base-a-test-cpu",
    nightly=False,
    disabled=None,
)


def test_multimodal_wrapper_builds_dcp_extend_metadata():
    """The top-level model must publish DCP metadata before prefix KV gather."""
    kwargs = {
        "seq_lens": sentinel.seq_lens,
        "extend_prefix_lens": sentinel.extend_prefix_lens,
        "extend_prefix_lens_cpu": sentinel.extend_prefix_lens_cpu,
        "extend_seq_lens": sentinel.extend_seq_lens,
        "req_pool_indices": sentinel.req_pool_indices,
        "req_to_token": sentinel.req_to_token,
        "seq_lens_sum": 17,
        "kv_buffer_shape": sentinel.kv_buffer_shape,
        "kv_cache_dtype": sentinel.kv_cache_dtype,
        "kv_cache_device": sentinel.kv_cache_device,
        "create_chunked_prefix_cache_kv_indices_fn": sentinel.create_indices,
    }

    with patch(
        "sglang.srt.models.glm5_next.prepare_decode_context_parallel_metadata",
        return_value=sentinel.metadata,
    ) as planner:
        result = (
            Glm5NextForConditionalGeneration.prepare_context_parallel_metadata_for_dcp(
                SimpleNamespace(), **kwargs
            )
        )

    assert result is sentinel.metadata
    planner.assert_called_once_with(**kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
