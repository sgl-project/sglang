from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dflash_compact_physical_layout import (
    CompactDFlashPhysicalLayout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _worker_module():
    from sglang.srt.speculative import dflash_worker_v2

    return dflash_worker_v2


def _bare_worker(*, target_table: torch.Tensor, draft_table: torch.Tensor):
    worker_mod = _worker_module()
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._target_worker = SimpleNamespace()
    worker.device = torch.device("cpu")
    worker.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=target_table)
    )
    worker.draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=draft_table)
    )
    worker._draft_block_end_buf = torch.empty(2, dtype=torch.int32)
    return worker


def _assign_req_to_token_cpu(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    batch_size,
):
    """Small CPU stand-in for the runtime kernel, including ragged packing."""
    packed_offset = 0
    for batch_idx in range(batch_size):
        req_idx = int(req_pool_indices[batch_idx])
        start = int(start_offset[batch_idx])
        end = int(end_offset[batch_idx])
        length = end - start
        req_to_token[req_idx, start:end].copy_(
            out_cache_loc[packed_offset : packed_offset + length].to(req_to_token.dtype)
        )
        packed_offset += length
    assert packed_offset == out_cache_loc.numel()


@pytest.mark.parametrize(
    ("prefix_lens", "extend_lens", "window_size", "expected"),
    [
        ([0], [6], 8, [(0, 0, 6, 0)]),
        ([0], [12], 8, [(0, 4, 12, 4)]),
        (
            [0, 0],
            [4, 12],
            8,
            [(0, 0, 4, 0), (1, 8, 16, 4)],
        ),
    ],
)
def test_compact_prefill_visible_segments_slice_packed_prompt_suffix(
    prefix_lens, extend_lens, window_size, expected
):
    worker_mod = _worker_module()

    assert (
        worker_mod._compact_prefill_visible_segments(
            prefix_lens, extend_lens, window_size
        )
        == expected
    )


def test_compact_prefill_chunk_crosses_ring_wrap_with_absolute_positions():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1, window_size=8, block_size=4, page_size=1
    )

    first = worker_mod._compact_prefill_visible_segments([0], [6], 8)
    second = worker_mod._compact_prefill_visible_segments([6], [6], 8)
    assert first == [(0, 0, 6, 0)]
    assert second == [(0, 0, 6, 6)]

    _, packed_start, packed_end, absolute_start = second[0]
    positions = torch.arange(
        absolute_start,
        absolute_start + packed_end - packed_start,
        dtype=torch.int64,
    )
    owners = torch.ones_like(positions)
    torch.testing.assert_close(
        layout.committed_locs(owners, positions),
        torch.tensor([14, 15, 8, 9, 10, 11]),
        rtol=0,
        atol=0,
    )


def test_commit_accept_covers_zero_accept_and_full_commit_block():
    worker_mod = _worker_module()
    candidates = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])

    out_tokens, commit_lens = worker_mod._commit_accept(
        candidates,
        accept_len=torch.tensor([0, 3]),
        bonus_tokens=torch.tensor([90, 99]),
    )

    torch.testing.assert_close(commit_lens, torch.tensor([1, 4], dtype=torch.int32))
    torch.testing.assert_close(
        out_tokens,
        torch.tensor([[90, 12, 13, 0], [21, 22, 23, 99]]),
    )


def test_compact_verify_locs_and_prefix_valid_writer_keep_zero_and_full_lens():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=2, window_size=8, block_size=4, page_size=1
    )
    owners = torch.tensor([1, 2], dtype=torch.int64)
    positions_2d = torch.tensor([[6, 7, 8, 9], [14, 15, 16, 17]])
    committed_locs = worker_mod._compact_verify_committed_locs(
        layout, owners, positions_2d
    )
    torch.testing.assert_close(
        committed_locs,
        torch.tensor([[14, 15, 8, 9], [38, 39, 32, 33]]),
        rtol=0,
        atol=0,
    )

    class PrefixPool:
        def __init__(self):
            self.calls = []
            self.written_locs = []

        def set_kv_buffer_prefix_valid(
            self, attn, cache_loc_2d, commit_lens, k, v, k_scale, v_scale
        ):
            self.calls.append((cache_loc_2d.clone(), commit_lens.clone(), k, v))
            for row, commit_len in enumerate(commit_lens.tolist()):
                self.written_locs.extend(cache_loc_2d[row, :commit_len].tolist())

    pool = PrefixPool()
    inner_attn = SimpleNamespace(k_scale=1.0, v_scale=1.0)
    attn = SimpleNamespace(
        attn=inner_attn,
        num_kv_heads=1,
        head_dim=2,
        kv_proj_only=lambda hidden: (hidden, hidden + 100),
        apply_k_norm=lambda k: k,
        apply_k_rope=lambda positions, k: k,
    )
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker.model_runner = SimpleNamespace(device=torch.device("cpu"))
    worker.draft_model_runner = SimpleNamespace(token_to_kv_pool=pool)
    worker.draft_model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=attn)],
        project_target_hidden=lambda hidden: hidden,
        prepare_context_hidden_for_kv=lambda layer, hidden: hidden,
    )
    worker._use_fused_kv_materialize = False
    worker._fused_kv_helper = None
    commit_lens = torch.tensor([0, 4], dtype=torch.int32)

    worker_mod.DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc(
        worker,
        target_hidden=torch.arange(16, dtype=torch.float32).view(8, 2),
        cache_loc=committed_locs.reshape(-1),
        cache_loc_2d=committed_locs,
        positions=positions_2d.reshape(-1),
        commit_lens=commit_lens,
    )

    assert len(pool.calls) == 1
    torch.testing.assert_close(pool.calls[0][0], committed_locs)
    torch.testing.assert_close(pool.calls[0][1], commit_lens)
    assert pool.written_locs == committed_locs[1].tolist()


def test_compact_rebuild_returns_scratch_and_writes_physical_mapping(monkeypatch):
    worker_mod = _worker_module()
    monkeypatch.setattr(
        worker_mod, "assign_req_to_token_pool_func", _assign_req_to_token_cpu
    )

    target_table = torch.full((3, 16), -1, dtype=torch.int64)
    draft_table = torch.full_like(target_table, -1)
    worker = _bare_worker(target_table=target_table, draft_table=draft_table)
    worker._compact_physical_layout = CompactDFlashPhysicalLayout.build(
        owner_count=2, window_size=8, block_size=2, page_size=1
    )
    worker._use_triton_compact_rebuild = False

    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    prefix_lens = torch.tensor([10, 9], dtype=torch.int32)
    draft_prefix_lens = torch.tensor([4, 3], dtype=torch.int32)
    verify_locs = torch.tensor([[901, 902], [903, 904]], dtype=torch.int64)

    actual = worker_mod.DFlashWorkerV2._rebuild_compact_draft_cache(
        worker,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        draft_prefix_lens=draft_prefix_lens,
        verify_out_cache_loc_2d=verify_locs,
        bs=2,
        block_size=2,
    )

    expected_scratch = worker._compact_physical_layout.scratch_locs(req_pool_indices, 2)
    torch.testing.assert_close(actual, expected_scratch, rtol=0, atol=0)
    torch.testing.assert_close(draft_table[1, :6], torch.tensor([10, 11, 4, 5, 12, 13]))
    torch.testing.assert_close(draft_table[2, :5], torch.tensor([26, 27, 20, 28, 29]))
    assert not torch.isin(verify_locs, draft_table).any()


def test_compact_triton_rebuild_branch_is_sync_free_and_uses_static_geometry(
    monkeypatch,
):
    worker_mod = _worker_module()
    target_table = torch.full((3, 16), -1, dtype=torch.int64)
    draft_table = torch.full_like(target_table, -1)
    worker = _bare_worker(target_table=target_table, draft_table=draft_table)
    worker._compact_physical_layout = CompactDFlashPhysicalLayout.build(
        owner_count=2, window_size=8, block_size=2, page_size=1
    )
    worker._use_triton_compact_rebuild = True
    worker._draft_physical_out_cache_loc_buf = torch.empty((2, 2), dtype=torch.int64)
    calls = []

    def fake_rebuild(**kwargs):
        calls.append(kwargs)
        layout = worker._compact_physical_layout
        for row in range(kwargs["batch_size"]):
            owner = int(kwargs["req_pool_indices"][row])
            start = int(kwargs["suffix_start"][row])
            length = int(kwargs["draft_prefix_lens"][row])
            positions = torch.arange(start, start + length)
            owners = torch.full_like(positions, owner)
            committed = layout.committed_locs(owners, positions)
            scratch = layout.scratch_locs(torch.tensor([owner]), 2).view(-1)
            draft_table[owner, : length + 2] = torch.cat((committed, scratch))
            kwargs["physical_out_cache_loc_2d"][row].copy_(scratch)

    monkeypatch.setattr(
        worker_mod,
        "rebuild_compact_physical_draft_req_to_token_func",
        fake_rebuild,
    )
    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    prefix_lens = torch.tensor([10, 9], dtype=torch.int32)
    draft_prefix_lens = torch.tensor([4, 3], dtype=torch.int32)

    actual = worker_mod.DFlashWorkerV2._rebuild_compact_draft_cache(
        worker,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        draft_prefix_lens=draft_prefix_lens,
        verify_out_cache_loc_2d=torch.full((2, 2), -1, dtype=torch.int64),
        bs=2,
        block_size=2,
    )

    assert actual.data_ptr() == worker._draft_physical_out_cache_loc_buf.data_ptr()
    assert len(calls) == 1
    assert calls[0]["owner_span"] == worker._compact_physical_layout.owner_span
    assert calls[0]["guard_rows"] == worker._compact_physical_layout.guard_rows
    assert calls[0]["window_size"] == 8
    torch.testing.assert_close(
        actual,
        worker._compact_physical_layout.scratch_locs(req_pool_indices, 2),
    )


def test_legacy_rebuild_returns_verify_locs_and_preserves_target_suffix(monkeypatch):
    worker_mod = _worker_module()
    monkeypatch.setattr(
        worker_mod, "assign_req_to_token_pool_func", _assign_req_to_token_cpu
    )

    target_table = torch.stack(
        (
            torch.arange(16),
            torch.arange(100, 116),
            torch.arange(200, 216),
        )
    )
    draft_table = torch.full((3, 16), -1, dtype=torch.int64)
    worker = _bare_worker(target_table=target_table, draft_table=draft_table)
    worker._compact_physical_layout = None
    worker._use_triton_compact_rebuild = False

    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    prefix_lens = torch.tensor([10, 9], dtype=torch.int32)
    draft_prefix_lens = torch.tensor([4, 3], dtype=torch.int32)
    verify_locs = torch.tensor([[901, 902], [903, 904]], dtype=torch.int64)

    actual = worker_mod.DFlashWorkerV2._rebuild_compact_draft_cache(
        worker,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        draft_prefix_lens=draft_prefix_lens,
        verify_out_cache_loc_2d=verify_locs,
        bs=2,
        block_size=2,
    )

    assert actual is verify_locs
    torch.testing.assert_close(
        draft_table[1, :6], torch.tensor([106, 107, 108, 109, 901, 902])
    )
    torch.testing.assert_close(
        draft_table[2, :5], torch.tensor([206, 207, 208, 903, 904])
    )


@pytest.mark.parametrize(
    ("page_size", "disable_radix_cache", "message"),
    [
        (16, True, "requires page_size=1"),
        (1, False, "requires --disable-radix-cache"),
    ],
)
def test_compact_worker_rejects_unsafe_cache_geometry_before_draft_load(
    monkeypatch, page_size, disable_radix_cache, message
):
    worker_mod = _worker_module()
    monkeypatch.setattr(
        worker_mod,
        "get_schedule",
        lambda: SimpleNamespace(page_size=page_size),
    )
    monkeypatch.setattr(
        worker_mod,
        "get_spec",
        lambda: SimpleNamespace(
            speculative_draft_window_size=2048,
            speculative_dflash_compact_cache=True,
        ),
    )
    monkeypatch.setattr(
        worker_mod,
        "get_memory",
        lambda: SimpleNamespace(disable_radix_cache=disable_radix_cache),
    )
    target_worker = SimpleNamespace(model_runner=SimpleNamespace(), device="cpu")

    with pytest.raises(RuntimeError, match=message):
        worker_mod.DFlashWorkerV2(
            server_args=SimpleNamespace(),
            gpu_id=0,
            ps=SimpleNamespace(),
            nccl_port=1,
            target_worker=target_worker,
        )
