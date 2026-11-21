"""Packing eligibility against real current pools and storage registration APIs.

These tests allocate small GPU pools; they do not load MoRI or issue network IO.
The allocator adapter for static pools supplies the same get_kvcache/state API
as serving; every pool, region and wire entry is produced by production code.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.packed_layout import (
    describe_packable_pool,
    describe_packable_storage,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def static_storage(pool):
    return StaticKVStorage(
        None,
        SimpleNamespace(storage=None, state=None, get_kvcache=lambda: pool),
    )


def make_pool(dtype=torch.bfloat16, hybrid=False):
    kwargs = dict(
        size=128,
        page_size=16,
        dtype=dtype,
        head_num=4,
        head_dim=64,
        device=f"cuda:{torch.cuda.current_device()}",
        enable_memory_saver=False,
        enable_alt_stream=False,
        kv_cache_layout="nhd",
    )
    if hybrid:
        return SWAKVPool(
            size_swa=64,
            full_attention_layer_ids=[1, 3],
            swa_attention_layer_ids=[0, 2],
            **kwargs,
        )
    return MHATokenToKVPool(layer_num=2, **kwargs)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
)
def test_real_static_storage_registration_order_and_logical_dtype(hybrid, dtype):
    pool = make_pool(dtype, hybrid)
    storage = static_storage(pool)
    proof = describe_packable_storage(storage)
    assert proof is not None
    entries = list(storage.wire_entries())
    assert len(entries) == int(hybrid)
    for name, key in [(storage.owner, "full")] + [
        (entry.sub_pool, "swa") for entry in entries
    ]:
        view, region = storage.view(name), storage.regions(name)
        if hybrid and name == storage.owner:
            view = pool.full_kv_pool
        buffers = list(view.k_buffer) + list(view.v_buffer)
        geometry = proof[key]
        assert list(region.ptrs) == [tensor.data_ptr() for tensor in buffers]
        assert geometry["logical_dtype"] == str(dtype)
        assert geometry["storage_dtype"] == str(buffers[0].dtype)
        assert geometry["page_capacities"] == [
            span // geometry["page_bytes"] for span in region.lens
        ]
        assert geometry["heads"] == 4
        if region.layer_ids:
            assert tuple(region.layer_ids) == tuple(view.model_layer_ids) * 2


def test_real_hybrid_region_reordered_pointers_and_layer_ids_rejected():
    storage = static_storage(make_pool(hybrid=True))
    entry = storage.wire_entries()[0]
    region, pool = storage.regions(entry.sub_pool), storage.view(entry.sub_pool)
    # Region is a production msgspec.Struct; replace only the claimed metadata.
    import msgspec

    wrong_ptrs = msgspec.structs.replace(region, ptrs=tuple(reversed(region.ptrs)))
    # Static SWA currently uses positional registration and omits layer IDs.
    # Inject a nonempty, wrong order; reversing the real empty tuple is a no-op.
    wrong_layers = msgspec.structs.replace(
        region, layer_ids=tuple(reversed(pool.model_layer_ids)) * 2
    )
    assert describe_packable_pool(pool, wrong_ptrs) is None
    assert describe_packable_pool(pool, wrong_layers) is None


def test_real_unified_page_envelope_is_conservative_fallback():
    from sglang.test.unified_allocator_fixtures import (
        build_test_unified_pools,
        setup_allocator_context,
    )

    setup_allocator_context()
    bundle = build_test_unified_pools(
        device=f"cuda:{torch.cuda.current_device()}",
        kv_cache_dtype=torch.bfloat16,
        page_size=16,
        start_layer=0,
        end_layer=4,
        full_attention_layer_ids=[1, 3],
        swa_attention_layer_ids=[0, 2],
        head_num=4,
        head_dim=64,
        swa_head_num=4,
        swa_head_dim=64,
        full_max_total_num_tokens=128,
        swa_max_total_num_tokens=64,
        sliding_window_size=64,
    )
    storage = bundle.token_to_kv_pool_allocator.storage
    assert storage is not None
    # Unified arena registers page envelopes, not K-layer/V-layer row spans.
    assert describe_packable_storage(storage) is None
