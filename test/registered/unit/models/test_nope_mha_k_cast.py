import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mha,
    forward_mha_rocm,
)
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


def _run_nope_concat_rocm(backend: str, k_pe: torch.Tensor | None):
    # qk_head_dim / qk_nope_head_dim are the roped-model values so that the
    # concat branch would be entered (and fail on the zero-width tail) if the
    # qk_rope_head_dim == 0 guard were missing.
    fake_self = SimpleNamespace(
        qk_rope_head_dim=0,
        qk_nope_head_dim=128,
        qk_head_dim=128,
        num_local_heads=2,
        current_attention_backend=backend,
    )
    return forward_mha_rocm.DeepseekMHARocmForwardMixin._concat_and_cast_mha_k_rocm(
        fake_self,
        torch.randn(4, 2, 128, dtype=torch.bfloat16),
        k_pe,
    )


@pytest.mark.parametrize("backend", ["aiter", "triton"])
@pytest.mark.parametrize("zero_width_k_pe", [False, True])
def test_nope_mha_k_cast_rocm(backend, zero_width_k_pe):
    k_pe = torch.randn(4, 1, 0, dtype=torch.bfloat16) if zero_width_k_pe else None
    out = _run_nope_concat_rocm(backend, k_pe)
    assert out.shape == (4, 2, 128)
    assert out.dtype == torch.bfloat16
    assert out.is_contiguous()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))


def test_fp8_dsa_prefix_read_uses_the_hybrid_full_attn_child():
    """A hybrid KDA/DSA wrapper holds no metadata; the FP8 prefix read crashed on it."""
    page_table = torch.tensor([3, 5, 8])
    child = SimpleNamespace(
        forward_metadata=SimpleNamespace(page_table_1_flattened=page_table)
    )
    outer = SimpleNamespace(
        full_attn_backend=child, forward_metadata=None, kv_index_translator=None
    )
    read_indices = []

    def get_mla_kv_buffer(layer, indices, dtype):
        read_indices.append(indices)
        return torch.zeros(len(indices), 1, 4), torch.zeros(len(indices), 1, 2)

    fake_pool = SimpleNamespace(get_mla_kv_buffer=get_mla_kv_buffer)
    with (
        mock.patch.object(forward_mha, "_use_aiter_gfx95", True),
        mock.patch.object(forward_mha, "get_attn_backend", return_value=outer),
        mock.patch.object(forward_mha, "get_token_to_kv_pool", return_value=fake_pool),
        mock.patch.object(
            forward_mha,
            "filter_dcp_local_kv_indices",
            side_effect=lambda kv_indices: kv_indices,
        ),
    ):
        forward_mha.DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
            SimpleNamespace(attn_mha=object()), SimpleNamespace(forward_mode=None)
        )

    assert len(read_indices) == 1 and torch.equal(read_indices[0], page_table)
