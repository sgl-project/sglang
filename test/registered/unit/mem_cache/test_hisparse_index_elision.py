"""Shared index-K ownership must agree with pool sizing and keep wire layer IDs."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import kv_cache_configurator as policy
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def glm_config():
    return SimpleNamespace(
        model_type="glm_moe_dsa",
        num_hidden_layers=78,
        indexer_types=[
            "full" if i < 3 or (i - 2) % 4 == 0 else "shared" for i in range(78)
        ],
    )


@pytest.mark.parametrize(
    "change,expected",
    [
        ({}, True),
        ({"enabled": False}, False),
        ({"draft": True}, False),
        ({"cuda": False}, False),
        ({"npu": True}, False),
        ({"external": True}, False),
        ({"pp_size": 2}, False),
        ({"attn_cp_size": 2}, False),
        ({"attn_dcp_size": 2}, False),
        ({"mode": "null"}, False),
        ({"mode": "prefill"}, False),
        ({"model_type": "deepseek_v3"}, False),
        ({"types": ["shared"] * 78}, False),
        ({"types": ["full"]}, False),
        ({"types": ["full"] + ["unknown"] * 77}, False),
        ({"types": None}, False),
        ({"disable_prefetch": True}, True),
        ({"hicache": True}, True),
    ],
)
def test_hisparse_policy_is_narrow_and_independent_of_prefetch(change, expected):
    config = glm_config()
    config.model_type = change.get("model_type", config.model_type)
    config.indexer_types = change.get("types", config.indexer_types)
    override = get_context().override_server_args(
        enable_hisparse=True,
        enable_unified_cache_external_linker=change.get("external", False),
        enable_hierarchical_cache=change.get("hicache", False),
        disaggregation_mode=change.get("mode", "decode"),
    )
    override.install()
    try:
        with (
            envs.SGLANG_ENABLE_HISPARSE_INDEX_K_ELISION.override(
                change.get("enabled", True)
            ),
            envs.SGLANG_DISABLE_HISPARSE_PREFETCH.override(
                change.get("disable_prefetch", False)
            ),
            patch.object(policy, "_is_cuda", change.get("cuda", True)),
            patch.object(policy, "_is_npu", change.get("npu", False)),
            get_parallel().override(
                tp_size=change.get("attn_cp_size", 1) * change.get("attn_dcp_size", 1),
                attn_tp_size=change.get("attn_dcp_size", 1),
                attn_dp_size=1,
                moe_tp_size=change.get("attn_cp_size", 1)
                * change.get("attn_dcp_size", 1),
                **{
                    k: change.get(k, 1)
                    for k in ("pp_size", "attn_cp_size", "attn_dcp_size")
                },
            ),
        ):
            assert (
                policy._should_elide_dsa_index_k(
                    is_draft_worker=change.get("draft", False), hf_config=config
                )
                is expected
            )
    finally:
        override.restore()


@pytest.mark.parametrize(
    "start,end,active", [(0, 78, 21), (0, 21, 7), (21, 40, 5), (40, 59, 5), (59, 78, 4)]
)
def test_index_storage_keeps_layer_positions_and_exact_bytes(start, end, active):
    types = glm_config().indexer_types[start:end]
    pool = SimpleNamespace(
        page_size=64,
        index_page_size=64,
        index_kpool=1,
        index_head_dim=128,
        quant_block_size=128,
        index_k_with_scale_buffer_dtype=torch.uint8,
        skip_topk_layers=[t == "shared" for t in types],
        layer_num=end - start,
        start_layer=start,
        layer_transfer_counter=None,
        custom_mem_pool=None,
        memory_saver_adapter=SimpleNamespace(region=lambda _: nullcontext()),
        device="cpu",
    )
    logical_tokens = 128 * 20
    cache = IndexKeyCache(pool, logical_tokens)
    ptrs, sizes, strides = cache.state_buf_infos()
    assert len(ptrs) == len(sizes) == len(strides) == end - start
    assert sum(s > 0 for s in sizes) == active
    assert sum(sizes) == active * (logical_tokens + 64) * 132
    for i, shared in enumerate(pool.skip_topk_layers):
        if shared:
            assert sizes[i] == strides[i] == 0
            assert cache.get_local_buffer(start + i).shape == (0, 64 * 132)
            with pytest.raises(RuntimeError, match="no index-K storage"):
                cache.get_k_and_scale(start + i, None, None, 0, 0)
            with pytest.raises(RuntimeError, match="no index-K storage"):
                cache.store_quantized(start + i, None, None, None)
        else:
            assert (
                cache.get_local_buffer(start + i).shape[0] == logical_tokens // 64 + 1
            )
            assert strides[i] == 64 * 132


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("accessor", ["local", "ordinary", "page_interleave"])
def test_shared_raw_getters_keep_placeholder_and_advance_prefetch(
    enabled, active, accessor
):
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.page_interleave_pool import (
        PageInterleaveDSATokenToKVPool,
        _PageInterleaveIndexKeyCache,
    )

    counter = SimpleNamespace(wait_until=Mock())
    acquire = Mock(
        return_value=SimpleNamespace(tensors={"index_k": torch.ones(5, 64 * 132)})
    )
    pool = SimpleNamespace(
        page_size=64,
        index_page_size=64,
        index_kpool=1,
        index_head_dim=128,
        quant_block_size=128,
        index_k_with_scale_buffer_dtype=torch.uint8,
        skip_topk_layers=[False, True, False],
        layer_num=3,
        start_layer=3,
        layer_transfer_counter=counter,
        custom_mem_pool=None,
        memory_saver_adapter=SimpleNamespace(region=lambda _: nullcontext()),
        device="cpu",
        _shard_extend_active=active,
        _acquire_slot_for_read=acquire,
    )
    cache = _PageInterleaveIndexKeyCache(pool, 256)
    pool.index_key_cache = cache
    with envs.SGLANG_ENABLE_HISPARSE_INDEX_K_ELISION.override(enabled):
        if accessor == "local":
            result = cache.get_local_buffer(4)
        elif accessor == "ordinary":
            result = DSATokenToKVPool.get_index_k_with_scale_buffer(pool, 4)
        else:
            result = PageInterleaveDSATokenToKVPool.get_index_k_with_scale_buffer(
                pool, 4
            )
    assert result.shape == (0, 64 * 132)
    counter.wait_until.assert_called_once_with(1)
    if active and accessor == "page_interleave":
        acquire.assert_called_once_with(4)
    else:
        acquire.assert_not_called()


@pytest.mark.parametrize("hisparse", [False, True])
@pytest.mark.parametrize("hicache", [False, True])
@pytest.mark.parametrize("mode", ["decode", "null"])
def test_npu_without_hf_config_preserves_existing_policy(hisparse, hicache, mode):
    override = get_context().override_server_args(
        enable_hisparse=hisparse,
        enable_hierarchical_cache=hicache,
        enable_unified_cache_external_linker=False,
        disaggregation_mode=mode,
    )
    override.install()
    try:
        with (
            envs.SGLANG_ENABLE_HISPARSE_INDEX_K_ELISION.override(True),
            patch.object(policy, "_is_npu", True),
            patch.object(policy, "_is_cuda", False),
            patch.object(policy, "getattr", wraps=getattr, create=True) as read_config,
        ):
            expected = mode == "null" and not hicache and not hisparse
            assert (
                policy._should_elide_dsa_index_k(is_draft_worker=False, hf_config=None)
                is expected
            )
            if mode == "null" and not hicache and hisparse:
                # The real NPU caller can omit hf_config. Ensure this case
                # reaches the safe getattr, rather than an earlier PD gate.
                read_config.assert_called_once_with(None, "indexer_types", None)
            else:
                read_config.assert_not_called()
    finally:
        override.restore()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
