import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_dcp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=5,
    suite="base-a-test-cpu",
    nightly=False,
    disabled=None,
)


def test_nope_prefix_kv_gather_accepts_missing_rope_cache():
    """NoPE DCP prefix reuse must not require a rope-cache tensor."""
    prefix_kv_a = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    prefix_kv_lens_cpu = torch.tensor([3], dtype=torch.int32)
    parallel = SimpleNamespace(dcp_enabled=True, dcp_size=1, dcp_rank=0)

    with (
        patch("sglang.srt.layers.dcp.comm.get_parallel", return_value=parallel),
        patch(
            "sglang.srt.layers.dcp.comm._all_gather_dcp_kv_cache",
            side_effect=lambda tensor: tensor,
        ),
    ):
        result = all_gather_kv_cache_for_dcp(
            prefix_kv_a,
            None,
            prefix_kv_lens_cpu,
        )

    assert torch.equal(result, prefix_kv_a)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
