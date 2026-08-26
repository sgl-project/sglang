import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=10,
    suite="base-a-test-cpu",
    nightly=False,
    disabled=None,
)


def _run_nope_concat(backend: str, kv_cache_dtype: str, pool_dtype: torch.dtype):
    fake_self = SimpleNamespace(
        qk_rope_head_dim=0,
        current_attention_backend=backend,
        kv_cache_dtype=kv_cache_dtype,
    )
    fake_pool = SimpleNamespace(dtype=pool_dtype)
    with (
        mock.patch.object(forward_mha, "_is_cuda", True),
        mock.patch.object(forward_mha, "get_token_to_kv_pool", return_value=fake_pool),
    ):
        return forward_mha.DeepseekMHAForwardMixin._concat_and_cast_mha_k(
            fake_self,
            torch.randn(4, 2, 128, dtype=torch.bfloat16),
            None,
            None,
        )


@pytest.mark.parametrize(
    "backend,kv_cache_dtype,pool_dtype,expected_dtype",
    [
        ("fa3", "fp8_e4m3", torch.float8_e4m3fn, torch.float8_e4m3fn),
        ("fa3", "auto", torch.float8_e4m3fn, torch.bfloat16),
        ("trtllm_gen", "fp8_e4m3", torch.float8_e4m3fn, torch.bfloat16),
    ],
)
def test_nope_mha_k_cast(backend, kv_cache_dtype, pool_dtype, expected_dtype):
    out = _run_nope_concat(backend, kv_cache_dtype, pool_dtype)
    assert out.dtype == expected_dtype


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
