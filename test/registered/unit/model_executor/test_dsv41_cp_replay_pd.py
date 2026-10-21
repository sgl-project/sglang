"""CPU contracts for CP owner fan-in and the upstream paged PD ABI."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_dsv41_cp_replay_layout import definitions


@pytest.mark.parametrize("end", [1, 127, 128, 129, 256, 257, 4097, 8193])
def test_window_handoff_checks_history_and_excludes_page_padding(end):
    cls = definitions("srt/mem_cache/dsv41_request_window.py", ["RequestWindow"])[
        "RequestWindow"
    ]
    window = object.__new__(cls)
    window.capacity, window.num_slots, window.page_size = 256, 8, 256
    window.tags = torch.full((3, 8 * 256 + 256), -1, dtype=torch.int64)
    start = max(0, end - 128) // 256 * 256
    positions = torch.arange(max(start, end - 128), end)
    rows = window.row_loc(2, positions)
    window.tags[:, rows] = positions
    np.testing.assert_array_equal(
        window.validate_transfer(2, start, end), window.page_indices(2, start, end)
    )
    window.tags[-1, rows[-1]] = -1
    with pytest.raises(RuntimeError, match="missing SWA history"):
        window.validate_transfer(2, start, end)


@pytest.mark.parametrize("cp", [1, 8])
def test_decode_derives_owner_fan_in_from_remote_cp(cp):
    functions = definitions(
        "srt/disaggregation/common/conn.py",
        ["_resolve_rank_mapping"],
        klass="CommonKVManager",
    )
    manager = NS(
        attn_tp_size=8,
        attn_cp_size=1,
        attn_cp_rank=0,
        pp_size=1,
        pp_rank=0,
        is_mla_backend=True,
        is_hybrid_mla_backend=False,
        kv_args=NS(engine_rank=3),
        enable_all_cp_ranks_for_transfer=False,
    )
    info = NS(
        attn_tp_size=8 // cp,
        attn_cp_size=cp,
        pp_size=1,
        enable_kv_cache_sharding=False,
        enable_dsa_cache_layer_split=False,
        dsv41_spec_layout={"target": "v4", "draft": "v4"},
    )
    functions["_resolve_rank_mapping"](manager, info)
    assert info.target_cp_ranks == list(range(cp))
    assert info.required_prefill_response_num == cp
    assert info.required_dst_info_num == cp


@pytest.mark.parametrize("pages", [1, 7, 11, 64])
def test_kv_pages_have_exactly_one_cp_sender(pages):
    functions = definitions(
        "srt/disaggregation/utils.py",
        ["_get_cp_rank_page_bounds", "filter_kv_indices_for_cp_rank"],
    )
    owners = []
    for rank in range(8):
        manager = NS(attn_cp_rank=rank, attn_cp_size=8)
        selected, index = functions["filter_kv_indices_for_cp_rank"](
            manager,
            np.arange(pages, dtype=np.int32),
            slice(0, pages),
            total_pages=pages,
        )
        owners.extend(selected.tolist())
    assert sorted(owners) == list(range(pages))
    assert len(owners) == len(set(owners))


def test_replicated_state_uses_one_sender_and_keeps_layer_split_exception():
    functions = definitions(
        "srt/disaggregation/common/conn.py",
        ["_should_skip_cp_replicated_state_transfer"],
        klass="CommonKVManager",
        env={
            "get_parallel": lambda: NS(
                enable_dsa_cache_layer_split=False, enable_cp_tp_group_sharing=False
            )
        },
    )
    selected = []
    for rank in range(8):
        manager = NS(
            attn_cp_size=8, attn_cp_rank=rank, enable_all_cp_ranks_for_transfer=True
        )
        if not functions["_should_skip_cp_replicated_state_transfer"](manager):
            selected.append(rank)
    assert selected == [0]
