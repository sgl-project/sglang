from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend():
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.page_size = 1
    backend.kv_cache_dtype = torch.float16
    backend.kv_cache_dtype_str = "float8_e4m3fn"
    backend.kv_cache_is_mxfp8 = False
    backend.fa_impl_ver = 3
    backend.num_splits = 4
    return backend


def test_get_paged_mha_kv_cache_supports_head_groups():
    backend = _backend()
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=Mock(
            return_value=(
                torch.empty(8, 2, 16),
                torch.empty(8, 2, 16),
            )
        )
    )
    layer = SimpleNamespace(
        layer_id=3,
        tp_k_head_num=2,
        tp_v_head_num=2,
        head_dim=16,
        v_head_dim=16,
    )

    key_cache, value_cache = backend.get_paged_mha_kv_cache(
        layer,
        head_group_num=2,
    )

    assert key_cache.shape == (16, 1, 1, 16)
    assert value_cache.shape == (16, 1, 1, 16)


def test_prepare_paged_mha_query_reuses_fa_scaling_policy():
    backend = _backend()
    layer = SimpleNamespace(
        head_dim=16,
        k_scale=torch.tensor(2.0),
        v_scale=torch.tensor(4.0),
    )
    q = torch.ones(2, 16, dtype=torch.bfloat16)

    q, _, _, k_descale, v_descale = backend.prepare_paged_mha_query(
        q,
        None,
        None,
        layer,
        logical_batch_size=2,
        kv_head_num=1,
        is_prefill=True,
    )

    assert q.dtype == torch.float16
    assert k_descale.shape == (2, 1)
    assert v_descale.shape == (2, 1)
