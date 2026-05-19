from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.kv_cache_canary_verify import (
    _MAX_REAL_KV_SOURCES,
    CANARY_SLOT_BYTES,
    RealKvSource,
)
from sglang.srt.kv_cache_canary.buffer_group import PoolKind
from sglang.srt.kv_cache_canary.pool_patch import (
    _make_row_source,
    attach_canary_buffers,
    get_canary_buffer_groups,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


def test_canary_buffer_group_allocate_full_only(make_mha_pool, base_config, device):
    pool = make_mha_pool(num_slots=16, dim=8, layer_num=2)
    attach_canary_buffers(pool=pool, config=base_config, device=device)
    groups = get_canary_buffer_groups(pool)
    assert set(groups.keys()) == {PoolKind.FULL}
    group = groups[PoolKind.FULL]
    assert group.k_head.shape == (16, CANARY_SLOT_BYTES)
    assert group.k_tail.shape == (16, CANARY_SLOT_BYTES)
    assert group.v_head is not None and group.v_tail is not None
    assert group.v_head.shape == (16, CANARY_SLOT_BYTES)


def test_canary_buffer_group_allocate_full_and_swa(make_swa_pool, base_config, device):
    pool = make_swa_pool(full_slots=16, swa_slots=8)
    attach_canary_buffers(pool=pool, config=base_config, device=device)
    groups = get_canary_buffer_groups(pool)
    assert set(groups.keys()) == {PoolKind.FULL, PoolKind.SWA}
    assert groups[PoolKind.FULL].k_head.shape[0] == 16
    assert groups[PoolKind.SWA].k_head.shape[0] == 8
    assert groups[PoolKind.SWA].swa_index_lut is not None
    assert groups[PoolKind.FULL].swa_index_lut is None


def test_mla_pool_no_v_half(make_mla_pool, base_config, device):
    pool = make_mla_pool(num_slots=16, dim=16)
    attach_canary_buffers(pool=pool, config=base_config, device=device)
    group = get_canary_buffer_groups(pool)[PoolKind.FULL]
    assert group.v_head is None
    assert group.v_tail is None
    assert group.real_kv_sources_v == ()
    assert group.has_v_half is False


def test_dsv4_pool_real_kv_sources_count(make_dsv4_pool, base_config, device):
    pool = make_dsv4_pool(full_slots=16, swa_slots=8, page_size=128)
    attach_canary_buffers(pool=pool, config=base_config, device=device)
    groups = get_canary_buffer_groups(pool)
    for group in groups.values():
        assert len(group.real_kv_sources_k) <= _MAX_REAL_KV_SOURCES
        assert len(group.real_kv_sources_v) <= _MAX_REAL_KV_SOURCES


def test_real_kv_sources_below_4(device):
    layer = torch.zeros(8, 16, dtype=torch.float16, device=device)
    sources = _make_row_source(layer_buffer=layer, read_bytes=4)
    assert 0 < len(sources) <= _MAX_REAL_KV_SOURCES


def test_real_kv_sources_above_4_raises(device):
    from sglang.jit_kernel.kv_cache_canary_verify import (
        CanaryLaunchTag,
        RealKvHashMode,
        VerifyPlan,
        canary_verify_step,
    )

    tensor = torch.zeros(4, 4, dtype=torch.uint8, device=device)
    sources = tuple(
        RealKvSource(tensor=tensor, page_size=1, num_bytes_per_token=4, read_bytes=2)
        for _ in range(_MAX_REAL_KV_SOURCES + 1)
    )
    canary_buf = torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    plan = VerifyPlan.allocate(verify_capacity=1, device=device)
    violation_ring = torch.zeros(2, 8, dtype=torch.int64, device=device)
    violation_write_index = torch.zeros(1, dtype=torch.int32, device=device)
    slot_run_counter = torch.zeros(1, dtype=torch.int64, device=device)
    kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=device)

    with pytest.raises(ValueError):
        canary_verify_step(
            canary_buf=canary_buf,
            plan=plan,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=violation_ring,
            violation_write_index=violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=sources,
            real_kv_hash_mode=RealKvHashMode.OFF,
        )


@pytest.mark.parametrize("patched", [False, True])
def test_get_contiguous_buf_infos_inserts_canary_entries(
    make_mha_pool, base_config, device, patched
):
    pool = make_mha_pool(num_slots=16, dim=8, layer_num=2)
    ptrs_before, _, _ = pool.get_contiguous_buf_infos()
    n_before = len(ptrs_before)

    if patched:
        attach_canary_buffers(pool=pool, config=base_config, device=device)
        ptrs_after, _, _ = pool.get_contiguous_buf_infos()
        assert len(ptrs_after) == n_before + 4
    else:
        ptrs_after, _, _ = pool.get_contiguous_buf_infos()
        assert ptrs_after == ptrs_before


@pytest.mark.parametrize("patched", [False, True])
def test_get_state_buf_infos_inserts_canary_entries(
    make_swa_pool, base_config, device, patched
):
    pool = make_swa_pool(full_slots=16, swa_slots=8)
    ptrs_before, _, _ = pool.get_state_buf_infos()
    n_before = len(ptrs_before)

    if patched:
        attach_canary_buffers(pool=pool, config=base_config, device=device)
        ptrs_after, _, _ = pool.get_state_buf_infos()
        assert len(ptrs_after) == n_before + 8
    else:
        ptrs_after, _, _ = pool.get_state_buf_infos()
        assert ptrs_after == ptrs_before


def test_pd_layout_canary_inserted_correctly(make_mha_pool, base_config, device):
    pool = make_mha_pool(num_slots=16, dim=8, layer_num=2)
    k_ptrs_orig = [b.data_ptr() for b in pool.k_buffer]
    v_ptrs_orig = [b.data_ptr() for b in pool.v_buffer]

    attach_canary_buffers(pool=pool, config=base_config, device=device)
    group = get_canary_buffer_groups(pool)[PoolKind.FULL]
    ptrs_after, _, _ = pool.get_contiguous_buf_infos()

    canary_k_ptrs = [group.k_head.data_ptr(), group.k_tail.data_ptr()]
    canary_v_ptrs = [group.v_head.data_ptr(), group.v_tail.data_ptr()]
    assert ptrs_after[0] == canary_k_ptrs[0]
    assert ptrs_after[1 : 1 + len(k_ptrs_orig)] == k_ptrs_orig
    k_tail_idx = 1 + len(k_ptrs_orig)
    assert ptrs_after[k_tail_idx] == canary_k_ptrs[1]
    assert ptrs_after[k_tail_idx + 1] == canary_v_ptrs[0]
    v_start = k_tail_idx + 2
    assert ptrs_after[v_start : v_start + len(v_ptrs_orig)] == v_ptrs_orig
    assert ptrs_after[-1] == canary_v_ptrs[1]


def test_pd_naive_first_last_insert_fails_layout_check(
    make_mha_pool, base_config, device
):
    pool = make_mha_pool(num_slots=16, dim=8, layer_num=2)
    k_ptrs_orig = [b.data_ptr() for b in pool.k_buffer]
    v_ptrs_orig = [b.data_ptr() for b in pool.v_buffer]

    attach_canary_buffers(pool=pool, config=base_config, device=device)
    group = get_canary_buffer_groups(pool)[PoolKind.FULL]
    ptrs_after, _, _ = pool.get_contiguous_buf_infos()

    naive_layout = (
        [group.k_head.data_ptr(), group.k_tail.data_ptr()]
        + k_ptrs_orig
        + v_ptrs_orig
        + [group.v_head.data_ptr(), group.v_tail.data_ptr()]
    )
    assert ptrs_after != naive_layout


def test_canary_buf_per_token_bytes_within_budget(make_mha_pool, base_config, device):
    pool = make_mha_pool(num_slots=16, dim=64, layer_num=2)
    attach_canary_buffers(pool=pool, config=base_config, device=device)
    group = get_canary_buffer_groups(pool)[PoolKind.FULL]

    slot_stride_bytes = group.k_head.stride(0) * group.k_head.element_size()
    assert slot_stride_bytes <= CANARY_SLOT_BYTES

    real_kv_per_token_bytes = (
        pool.k_buffer[0].stride(0) * pool.k_buffer[0].element_size()
    )
    assert slot_stride_bytes < real_kv_per_token_bytes
