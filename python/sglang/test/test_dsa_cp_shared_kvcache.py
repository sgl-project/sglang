import argparse

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.dsa_cp_shared import (
    maybe_create_dsa_cp_shared_l2_allocator,
    should_enable_dsa_cp_shared_kvcache,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.shared_kv import SharedKVLayout
from sglang.srt.server_args import ServerArgs


def test_dsa_cp_shared_kvcache_cli_flag_is_registered():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    parsed = parser.parse_args(
        ["--model-path", "dummy", "--enable-dsa-cp-shared-kv-cache"]
    )

    assert parsed.enable_dsa_cp_shared_kv_cache


def test_dsa_cp_shared_kvcache_feature_gate():
    assert not should_enable_dsa_cp_shared_kvcache(
        enable_hisparse=False,
        enabled=False,
        is_hip_platform=False,
    )
    assert should_enable_dsa_cp_shared_kvcache(
        enable_hisparse=False,
        enabled=True,
        is_hip_platform=False,
    )
    assert not should_enable_dsa_cp_shared_kvcache(
        enable_hisparse=True,
        enabled=True,
        is_hip_platform=False,
    )
    assert not should_enable_dsa_cp_shared_kvcache(
        enable_hisparse=False,
        enabled=True,
        is_hip_platform=True,
    )


def test_dsa_cp_shared_slot_mapping_cpu_fallback():
    slots_per_page = 64
    layout = SharedKVLayout(cp_size=4, slots_per_page=slots_per_page, pages_per_rank=8)
    slot_indices = torch.tensor(
        [0, 63, 64, 65, 2 * 64, 3 * 64 + 5, 8 * 64, -1],
        dtype=torch.int64,
    )

    rank_major = layout.translate_read_slots(slot_indices)
    local = layout.translate_write_slots(slot_indices)
    owned_rank2 = layout.owned_slot_mask(slot_indices, owner_rank=2)

    assert torch.equal(
        rank_major,
        torch.tensor([0, 63, 512, 513, 1024, 1541, 128, -1], dtype=torch.int64),
    )
    assert torch.equal(
        local,
        torch.tensor([0, 63, 0, 1, 0, 5, 128, -1], dtype=torch.int64),
    )
    assert torch.equal(
        owned_rank2,
        torch.tensor([False, False, False, False, True, False, False, False]),
    )


def test_dsa_cp_shared_layout_dtype_helpers():
    layout = SharedKVLayout(cp_size=4, slots_per_page=64, pages_per_rank=8)
    slot_indices = torch.tensor([0, 64, 2 * 64 + 3, -1], dtype=torch.int64)

    assert torch.equal(
        layout.translate_read_slots(slot_indices, output_dtype=torch.int32),
        torch.tensor([0, 512, 1027, -1], dtype=torch.int32),
    )
    assert torch.equal(
        layout.translate_write_slots(slot_indices, output_dtype=torch.int32),
        torch.tensor([0, 0, 3, -1], dtype=torch.int32),
    )


def test_shared_kv_layout_selects_hicache_transfer_indices():
    layout = SharedKVLayout(cp_size=4, slots_per_page=64, pages_per_rank=8)
    host_indices = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    device_indices = torch.tensor([0, 64, 2 * 64 + 3, 3 * 64 + 5], dtype=torch.int64)

    load_host, load_device = layout.select_host_to_device_indices(
        host_indices, device_indices, owner_rank=2
    )
    assert torch.equal(load_host, torch.tensor([12], dtype=torch.int64))
    assert torch.equal(load_device, torch.tensor([3], dtype=torch.int64))

    backup_host, backup_device = layout.select_device_to_host_indices(
        host_indices, device_indices, current_rank=0, io_rank=0
    )
    assert torch.equal(backup_host, host_indices)
    assert torch.equal(
        backup_device,
        torch.tensor([0, 512, 1027, 1541], dtype=torch.int64),
    )

    non_io_host, non_io_device = layout.select_device_to_host_indices(
        host_indices, device_indices, current_rank=1, io_rank=0
    )
    assert non_io_host.numel() == 0
    assert non_io_device.numel() == 0


def test_dsa_cp_shared_page_table_translation_prefers_int32_fast_path():
    backend = object.__new__(DeepseekSparseAttnBackend)
    page_table = torch.tensor([[0, 64, -1]], dtype=torch.int64)

    class TokenToKVPool:
        def __init__(self):
            self.calls = []

        def translate_loc_to_cp_shared_device_int32(self, loc):
            self.calls.append(("int32", loc))
            return loc.to(torch.int32) + 7

        def translate_loc_to_cp_shared_device(self, loc):
            self.calls.append(("generic", loc))
            return loc + 11

    token_to_kv_pool = TokenToKVPool()
    backend.token_to_kv_pool = token_to_kv_pool

    out = backend._translate_page_table_for_main_kv(page_table)

    assert out.dtype == torch.int32
    assert torch.equal(out, torch.tensor([[7, 71, 6]], dtype=torch.int32))
    assert [name for name, _ in token_to_kv_pool.calls] == ["int32"]


def test_dsa_cp_shared_write_fence_hook_calls_pool_sync():
    backend = object.__new__(DeepseekSparseAttnBackend)

    class TokenToKVPool:
        def __init__(self):
            self.sync_calls = 0

        def synchronize_cp_shared_kv_write(self):
            self.sync_calls += 1

    token_to_kv_pool = TokenToKVPool()
    backend.token_to_kv_pool = token_to_kv_pool

    backend._synchronize_cp_shared_kv_write_before_read()

    assert token_to_kv_pool.sync_calls == 1
    assert (
        "_synchronize_cp_shared_kv_write_before_read"
        in DeepseekSparseAttnBackend.forward_decode.__code__.co_names
    )
    assert (
        "_synchronize_cp_shared_kv_write_before_read"
        in DeepseekSparseAttnBackend.forward_extend.__code__.co_names
    )


def test_dsa_cp_shared_mla_write_path_uses_shared_kernel():
    names = MLATokenToKVPool.set_mla_kv_buffer.__code__.co_names

    assert "_get_mla_write_buffer" in names
    assert "set_mla_kv_buffer_triton_cp_shared" in names


def test_dsa_cp_shared_l2_allocator_defaults_off():
    params = type("Params", (), {"attn_cp_cache_group": None, "tp_cache_group": None})()
    server_args = type("ServerArgs", (), {"enable_dsa_cp_shared_kv_cache": False})()
    kv_pool = type("KVPool", (), {"enable_cp_shared_kvcache": True})()

    assert (
        maybe_create_dsa_cp_shared_l2_allocator(
            params=params,
            server_args=server_args,
            kv_pool=kv_pool,
            kind="dsa_l2",
        )
        is None
    )
