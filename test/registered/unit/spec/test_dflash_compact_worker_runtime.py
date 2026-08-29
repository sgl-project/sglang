from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
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


def _bare_sidecar_worker(layout, *, project=None, move=None):
    worker_mod = _worker_module()
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._compact_physical_layout = layout
    worker.draft_window_size = layout.window_size
    worker.device = torch.device("cpu")
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            move_kv_cache=move or (lambda destination, source: None)
        )
    )
    worker._append_target_hidden_to_draft_kv_by_loc = project or (lambda **kwargs: None)
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


def test_compact_radix_restore_copies_content_to_owner_and_releases_pin():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1,
        window_size=4,
        block_size=2,
        page_size=1,
        content_tokens=8,
    )
    moves = []
    released = []
    source = torch.arange(layout.content_start, layout.content_start + 4)
    plan = SimpleNamespace(source_rows=source, matched_tokens=4)
    tree_cache = SimpleNamespace(
        get_dflash_draft_match_plan=lambda rid: plan if rid == "warm" else None,
        release_dflash_draft_match_pin=lambda rid: released.append(rid),
    )
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._compact_physical_layout = layout
    worker.draft_window_size = 4
    worker.device = torch.device("cpu")
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            move_kv_cache=lambda dst, src: moves.append((dst.clone(), src.clone()))
        )
    )
    batch = SimpleNamespace(
        tree_cache=tree_cache,
        reqs=[SimpleNamespace(rid="warm")],
        prefix_lens=[6],
        req_pool_indices=torch.tensor([1]),
        acquire_owner_mask=torch.tensor([True]),
    )

    restored = worker_mod.DFlashWorkerV2._restore_compact_radix_prefix(worker, batch)

    assert restored == 4
    assert released == ["warm"]
    assert len(moves) == 1
    torch.testing.assert_close(moves[0][1], source)
    torch.testing.assert_close(
        moves[0][0],
        layout.committed_locs(torch.ones(4, dtype=torch.int64), torch.arange(2, 6)),
    )


@pytest.mark.parametrize("bad_source", [False, True])
def test_compact_radix_restore_coverage_or_range_failure_releases_pin(bad_source):
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1,
        window_size=4,
        block_size=2,
        page_size=1,
        content_tokens=4,
    )
    source = (
        torch.arange(4)
        if bad_source
        else torch.arange(layout.content_start, layout.content_start + 3)
    )
    plan = SimpleNamespace(
        source_rows=source,
        matched_tokens=4 if bad_source else 3,
    )
    released = []
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._compact_physical_layout = layout
    worker.draft_window_size = 4
    worker.device = torch.device("cpu")
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(move_kv_cache=lambda dst, src: None)
    )
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            get_dflash_draft_match_plan=lambda rid: plan,
            release_dflash_draft_match_pin=lambda rid: released.append(rid),
        ),
        reqs=[SimpleNamespace(rid="bad")],
        prefix_lens=[6],
        req_pool_indices=torch.tensor([1]),
        acquire_owner_mask=torch.tensor([True]),
    )

    message = "content loc OOB" if bad_source else "coverage mismatch"
    with pytest.raises(RuntimeError, match=message):
        worker_mod.DFlashWorkerV2._restore_compact_radix_prefix(worker, batch)
    assert released == ["bad"]


def test_compact_radix_restore_skips_continuing_owner_without_plan():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1,
        window_size=4,
        block_size=2,
        page_size=1,
        content_tokens=4,
    )
    moves = []
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._compact_physical_layout = layout
    worker.draft_window_size = 4
    worker.device = torch.device("cpu")
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            move_kv_cache=lambda dst, src: moves.append((dst, src))
        )
    )
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            get_dflash_draft_match_plan=lambda rid: None,
            release_dflash_draft_match_pin=lambda rid: None,
        ),
        reqs=[SimpleNamespace(rid="next-chunk")],
        prefix_lens=[6],
        req_pool_indices=torch.tensor([1]),
        acquire_owner_mask=torch.tensor([False]),
    )

    assert worker_mod.DFlashWorkerV2._restore_compact_radix_prefix(worker, batch) == 0
    assert not moves


def test_compact_radix_restore_get_plan_failure_releases_every_batch_rid():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=3,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=6,
    )
    source = torch.arange(layout.content_start, layout.content_start + 2)
    plan = SimpleNamespace(source_rows=source, matched_tokens=2)
    get_calls = []
    released = []

    def get_plan(rid):
        get_calls.append(rid)
        if rid == "second":
            raise ValueError("get-plan boom")
        return plan

    worker = _bare_sidecar_worker(layout)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            get_dflash_draft_match_plan=get_plan,
            release_dflash_draft_match_pin=lambda rid: released.append(rid) or True,
        ),
        reqs=[
            SimpleNamespace(rid="first"),
            SimpleNamespace(rid="second"),
            SimpleNamespace(rid="third"),
        ],
        prefix_lens=[2, 2, 2],
        req_pool_indices=torch.tensor([1, 2, 3]),
        acquire_owner_mask=torch.tensor([True, True, True]),
    )

    with pytest.raises(ValueError, match="get-plan boom"):
        _worker_module().DFlashWorkerV2._restore_compact_radix_prefix(worker, batch)

    assert get_calls == ["first", "second"]
    assert released == ["first", "second", "third"]


def test_compact_radix_restore_mask_failure_releases_every_batch_rid():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=2,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=4,
    )
    released = []
    worker = _bare_sidecar_worker(layout)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            get_dflash_draft_match_plan=lambda rid: pytest.fail(
                "mask validation must precede plan lookup"
            ),
            release_dflash_draft_match_pin=lambda rid: released.append(rid) or True,
        ),
        reqs=[SimpleNamespace(rid="first"), SimpleNamespace(rid="second")],
        prefix_lens=[2, 2],
        req_pool_indices=torch.tensor([1, 2]),
        acquire_owner_mask=torch.tensor([True]),
    )

    with pytest.raises(RuntimeError, match="acquire-mask mismatch"):
        _worker_module().DFlashWorkerV2._restore_compact_radix_prefix(worker, batch)

    assert released == ["first", "second"]


def test_compact_radix_restore_preserves_copy_error_and_attempts_all_releases():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=2,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=4,
    )
    source = torch.arange(layout.content_start, layout.content_start + 2)
    plan = SimpleNamespace(source_rows=source, matched_tokens=2)
    released = []

    def release(rid):
        released.append(rid)
        if rid == "first":
            raise RuntimeError("release boom")
        return True

    def move(destination, source):
        raise ValueError("copy boom")

    worker = _bare_sidecar_worker(layout, move=move)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            get_dflash_draft_match_plan=lambda rid: plan,
            release_dflash_draft_match_pin=release,
        ),
        reqs=[SimpleNamespace(rid="first"), SimpleNamespace(rid="second")],
        prefix_lens=[2, 2],
        req_pool_indices=torch.tensor([1, 2]),
        acquire_owner_mask=torch.tensor([True, True]),
    )

    with pytest.raises(ValueError, match="copy boom"):
        _worker_module().DFlashWorkerV2._restore_compact_radix_prefix(worker, batch)

    assert released == ["first", "second"]


def test_compact_prefill_sidecar_projects_each_visible_token_once_and_falls_back():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=2,
        window_size=4,
        block_size=2,
        page_size=1,
        content_tokens=4,
    )
    content_rows = torch.arange(layout.content_start, layout.content_end)
    allocations = [content_rows, None]
    staged = []
    moves = []
    projections = []
    tree_cache = SimpleNamespace(
        alloc_dflash_draft_content=lambda count: allocations.pop(0),
        stage_dflash_draft_publish=lambda rid, start, rows: staged.append(
            (rid, start, rows.clone())
        ),
        discard_dflash_draft_publish=lambda rid: None,
        free_unstaged_dflash_draft_content=lambda rows: None,
    )
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._compact_physical_layout = layout
    worker.draft_window_size = 4
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            move_kv_cache=lambda dst, src: moves.append((dst.clone(), src.clone()))
        )
    )
    worker._append_target_hidden_to_draft_kv_by_loc = (
        lambda **kwargs: projections.append(kwargs)
    )
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 4, 5], dtype=torch.int64)
    hidden = torch.arange(16, dtype=torch.float32).view(8, 2)
    batch = SimpleNamespace(
        tree_cache=tree_cache,
        reqs=[SimpleNamespace(rid="published"), SimpleNamespace(rid="fallback")],
        prefix_lens=[0, 4],
        extend_lens=[6, 2],
        req_pool_indices=torch.tensor([1, 2]),
    )

    handled = worker_mod.DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
        worker, batch=batch, target_hidden=hidden, positions=positions
    )

    assert handled
    assert len(projections) == 1
    torch.testing.assert_close(
        projections[0]["target_hidden"], torch.cat((hidden[2:6], hidden[6:8]))
    )
    assert projections[0]["target_hidden"].shape[0] == 6
    torch.testing.assert_close(projections[0]["cache_loc"][:4], content_rows)
    fallback_locs = layout.committed_locs(torch.full((2,), 2), torch.tensor([4, 5]))
    torch.testing.assert_close(projections[0]["cache_loc"][4:], fallback_locs)
    assert len(moves) == 1
    torch.testing.assert_close(moves[0][1], content_rows)
    assert staged[0][0:2] == ("published", 2)
    torch.testing.assert_close(staged[0][2], content_rows)


def test_compact_prefill_sidecar_publishes_cap_history_but_copies_only_owner_window():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1,
        window_size=4,
        block_size=2,
        page_size=1,
        content_tokens=8,
    )
    content_rows = torch.arange(layout.content_start, layout.content_end)
    staged = []
    moves = []
    projections = []
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._compact_physical_layout = layout
    worker.draft_window_size = 4
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            move_kv_cache=lambda dst, src: moves.append((dst.clone(), src.clone()))
        )
    )
    worker._append_target_hidden_to_draft_kv_by_loc = (
        lambda **kwargs: projections.append(kwargs)
    )
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            alloc_dflash_draft_content=lambda count: content_rows,
            stage_dflash_draft_publish=lambda rid, start, rows: staged.append(
                (rid, start, rows.clone())
            ),
            discard_dflash_draft_publish=lambda rid: None,
            free_unstaged_dflash_draft_content=lambda rows: None,
        ),
        reqs=[SimpleNamespace(rid="long-system")],
        prefix_lens=[0],
        extend_lens=[10],
        req_pool_indices=torch.tensor([1]),
    )
    hidden = torch.arange(20, dtype=torch.float32).view(10, 2)
    positions = torch.arange(10, dtype=torch.int64)

    assert worker_mod.DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
        worker, batch=batch, target_hidden=hidden, positions=positions
    )

    assert len(projections) == 1
    torch.testing.assert_close(projections[0]["target_hidden"], hidden[2:10])
    torch.testing.assert_close(projections[0]["cache_loc"], content_rows)
    assert staged[0][0:2] == ("long-system", 2)
    torch.testing.assert_close(staged[0][2], content_rows)
    assert len(moves) == 1
    torch.testing.assert_close(moves[0][1], content_rows[4:8])
    torch.testing.assert_close(
        moves[0][0],
        layout.committed_locs(torch.ones(4, dtype=torch.int64), torch.arange(6, 10)),
    )


def test_compact_prefill_sidecar_cap_below_window_projects_uncovered_owner_rows():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1,
        window_size=4,
        block_size=2,
        page_size=1,
        content_tokens=2,
    )
    content_rows = torch.arange(layout.content_start, layout.content_end)
    moves = []
    projections = []
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker._compact_physical_layout = layout
    worker.draft_window_size = 4
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            move_kv_cache=lambda dst, src: moves.append((dst.clone(), src.clone()))
        )
    )
    worker._append_target_hidden_to_draft_kv_by_loc = (
        lambda **kwargs: projections.append(kwargs)
    )
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            alloc_dflash_draft_content=lambda count: content_rows,
            stage_dflash_draft_publish=lambda rid, start, rows: None,
            discard_dflash_draft_publish=lambda rid: None,
            free_unstaged_dflash_draft_content=lambda rows: None,
        ),
        reqs=[SimpleNamespace(rid="small-cap")],
        prefix_lens=[0],
        extend_lens=[6],
        req_pool_indices=torch.tensor([1]),
    )
    hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
    positions = torch.arange(6, dtype=torch.int64)

    assert worker_mod.DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
        worker, batch=batch, target_hidden=hidden, positions=positions
    )

    assert len(projections) == 1
    torch.testing.assert_close(
        projections[0]["target_hidden"], torch.cat((hidden[4:6], hidden[2:4]))
    )
    torch.testing.assert_close(projections[0]["cache_loc"][:2], content_rows)
    torch.testing.assert_close(
        projections[0]["cache_loc"][2:],
        layout.committed_locs(torch.ones(2, dtype=torch.int64), torch.arange(2, 4)),
    )
    torch.testing.assert_close(moves[0][1], content_rows)
    torch.testing.assert_close(
        moves[0][0],
        layout.committed_locs(torch.ones(2, dtype=torch.int64), torch.arange(4, 6)),
    )


def test_compact_prefill_projection_failure_frees_every_unstaged_lease():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=2,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=4,
    )
    allocations = [
        torch.arange(layout.content_start, layout.content_start + 2),
        torch.arange(layout.content_start + 2, layout.content_end),
    ]
    allocated = [rows.clone() for rows in allocations]
    freed = []
    staged = []

    def project(**kwargs):
        raise ValueError("projection boom")

    worker = _bare_sidecar_worker(layout, project=project)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            alloc_dflash_draft_content=lambda count: allocations.pop(0),
            stage_dflash_draft_publish=lambda rid, start, rows: staged.append(rid),
            discard_dflash_draft_publish=lambda rid: True,
            free_unstaged_dflash_draft_content=lambda rows: freed.append(rows.clone()),
        ),
        reqs=[SimpleNamespace(rid="first"), SimpleNamespace(rid="second")],
        decoding_reqs=None,
        prefix_lens=[0, 0],
        extend_lens=[2, 2],
        req_pool_indices=torch.tensor([1, 2]),
    )

    with pytest.raises(ValueError, match="projection boom"):
        _worker_module().DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
            worker,
            batch=batch,
            target_hidden=torch.arange(8, dtype=torch.float32).view(4, 2),
            positions=torch.tensor([0, 1, 0, 1]),
        )

    assert not staged
    assert len(freed) == 2
    for actual, expected in zip(freed, allocated):
        torch.testing.assert_close(actual, expected)


def test_compact_prefill_stage_failure_partitions_staged_and_unstaged_rows():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=3,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=6,
    )
    allocations = [
        torch.arange(layout.content_start + offset, layout.content_start + offset + 2)
        for offset in (0, 2, 4)
    ]
    expected = [rows.clone() for rows in allocations]
    staged = []
    discarded = []
    freed = []

    def stage(rid, start, rows):
        if rid == "second":
            raise ValueError("stage boom")
        staged.append(rid)

    worker = _bare_sidecar_worker(layout)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            alloc_dflash_draft_content=lambda count: allocations.pop(0),
            stage_dflash_draft_publish=stage,
            discard_dflash_draft_publish=lambda rid: discarded.append(rid) or True,
            free_unstaged_dflash_draft_content=lambda rows: freed.append(rows.clone()),
        ),
        reqs=[
            SimpleNamespace(rid="first"),
            SimpleNamespace(rid="second"),
            SimpleNamespace(rid="third"),
        ],
        decoding_reqs=None,
        prefix_lens=[0, 0, 0],
        extend_lens=[2, 2, 2],
        req_pool_indices=torch.tensor([1, 2, 3]),
    )

    with pytest.raises(ValueError, match="stage boom"):
        _worker_module().DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
            worker,
            batch=batch,
            target_hidden=torch.arange(12, dtype=torch.float32).view(6, 2),
            positions=torch.tensor([0, 1, 0, 1, 0, 1]),
        )

    assert staged == ["first"]
    assert discarded == ["first"]
    assert len(freed) == 2
    torch.testing.assert_close(freed[0], expected[1])
    torch.testing.assert_close(freed[1], expected[2])


def test_compact_prefill_rollback_is_best_effort_and_preserves_primary_error():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=3,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=6,
    )
    allocations = [
        torch.arange(layout.content_start + offset, layout.content_start + offset + 2)
        for offset in (0, 2, 4)
    ]
    cleanup_calls = []

    def stage(rid, start, rows):
        if rid == "second":
            raise ValueError("primary stage boom")

    def discard(rid):
        cleanup_calls.append(("discard", rid))
        raise RuntimeError("discard boom")

    def free(rows):
        cleanup_calls.append(("free", int(rows[0])))
        if len([call for call in cleanup_calls if call[0] == "free"]) == 1:
            raise RuntimeError("free boom")

    worker = _bare_sidecar_worker(layout)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            alloc_dflash_draft_content=lambda count: allocations.pop(0),
            stage_dflash_draft_publish=stage,
            discard_dflash_draft_publish=discard,
            free_unstaged_dflash_draft_content=free,
        ),
        reqs=[
            SimpleNamespace(rid="first"),
            SimpleNamespace(rid="second"),
            SimpleNamespace(rid="third"),
        ],
        decoding_reqs=None,
        prefix_lens=[0, 0, 0],
        extend_lens=[2, 2, 2],
        req_pool_indices=torch.tensor([1, 2, 3]),
    )

    with pytest.raises(ValueError, match="primary stage boom") as exc_info:
        _worker_module().DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
            worker,
            batch=batch,
            target_hidden=torch.arange(12, dtype=torch.float32).view(6, 2),
            positions=torch.tensor([0, 1, 0, 1, 0, 1]),
        )

    assert cleanup_calls == [
        ("discard", "first"),
        ("free", layout.content_start + 2),
        ("free", layout.content_start + 4),
    ]
    assert len(exc_info.value.__notes__) == 2


def test_compact_prefill_content_range_failure_frees_fresh_allocation():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=2,
    )
    invalid_rows = torch.arange(layout.content_start - 1, layout.content_start + 1)
    freed = []
    worker = _bare_sidecar_worker(layout)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            alloc_dflash_draft_content=lambda count: invalid_rows,
            stage_dflash_draft_publish=lambda rid, start, rows: None,
            discard_dflash_draft_publish=lambda rid: True,
            free_unstaged_dflash_draft_content=lambda rows: freed.append(rows.clone()),
        ),
        reqs=[SimpleNamespace(rid="bad-range")],
        decoding_reqs=None,
        prefix_lens=[0],
        extend_lens=[2],
        req_pool_indices=torch.tensor([1]),
    )

    with pytest.raises(RuntimeError, match="content loc OOB"):
        _worker_module().DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
            worker,
            batch=batch,
            target_hidden=torch.arange(4, dtype=torch.float32).view(2, 2),
            positions=torch.tensor([0, 1]),
        )

    assert len(freed) == 1
    torch.testing.assert_close(freed[0], invalid_rows)


def test_compact_prefill_mixed_decode_and_skip_rows_only_update_owner_ring():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=3,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=2,
    )
    content_rows = torch.arange(layout.content_start, layout.content_end)
    allocations = []
    staged = []
    projections = []
    prefill = SimpleNamespace(rid="prefill")
    decode = SimpleNamespace(rid="decode")
    skipped = SimpleNamespace(rid="skip", skip_radix_cache_insert=True)
    worker = _bare_sidecar_worker(
        layout, project=lambda **kwargs: projections.append(kwargs)
    )
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            alloc_dflash_draft_content=lambda count: allocations.append(count)
            or content_rows,
            stage_dflash_draft_publish=lambda rid, start, rows: staged.append(rid),
            discard_dflash_draft_publish=lambda rid: True,
            free_unstaged_dflash_draft_content=lambda rows: None,
        ),
        reqs=[prefill, decode, skipped],
        decoding_reqs=[decode],
        prefix_lens=[0, 4, 0],
        extend_lens=[2, 1, 2],
        req_pool_indices=torch.tensor([1, 2, 3]),
    )
    positions = torch.tensor([0, 1, 4, 0, 1])

    assert _worker_module().DFlashWorkerV2._materialize_compact_prefill_with_sidecar(
        worker,
        batch=batch,
        target_hidden=torch.arange(10, dtype=torch.float32).view(5, 2),
        positions=positions,
    )

    assert allocations == [2]
    assert staged == ["prefill"]
    assert len(projections) == 1
    expected_owner_locs = torch.cat(
        (
            layout.committed_locs(torch.tensor([2]), torch.tensor([4])),
            layout.committed_locs(torch.tensor([3, 3]), torch.tensor([0, 1])),
        )
    )
    torch.testing.assert_close(projections[0]["cache_loc"][:2], content_rows)
    torch.testing.assert_close(projections[0]["cache_loc"][2:], expected_owner_locs)


def test_compact_generation_bind_failure_releases_radix_pin():
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=1,
        window_size=2,
        block_size=2,
        page_size=1,
        content_tokens=2,
    )
    released = []
    worker = _bare_sidecar_worker(layout)
    worker._compact_owner_generation = torch.tensor([0, 4], dtype=torch.int64)
    worker.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_generation=torch.tensor([0, 5], dtype=torch.int64)
        )
    )
    batch = SimpleNamespace(
        sampling_info=None,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        is_extend_in_batch=False,
        expected_req_generations_cpu=torch.tensor([4]),
        acquire_owner_mask=torch.tensor([True]),
        req_pool_indices_cpu=torch.tensor([1]),
        req_pool_indices=torch.tensor([1]),
        tree_cache=SimpleNamespace(
            release_dflash_draft_match_pin=lambda rid: released.append(rid) or True
        ),
        reqs=[SimpleNamespace(rid="stale")],
    )

    with pytest.raises(RuntimeError, match="generation mismatch"):
        _worker_module().DFlashWorkerV2.forward_batch_generation(worker, batch)

    assert released == ["stale"]


def test_compact_pool_uses_frozen_owner_and_sidecar_geometry_for_exact_bytes():
    worker_mod = _worker_module()
    layout = CompactDFlashPhysicalLayout.build(
        owner_count=2,
        window_size=4,
        block_size=2,
        page_size=1,
        content_tokens=5,
    )
    cell_bytes = 32
    allocated_bytes = (layout.physical_tokens + 1) * cell_bytes
    calls = []
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker.use_compact_draft_cache = True
    worker.draft_window_size = 4
    worker.block_size = 2
    worker.page_size = 1
    worker.dflash_radix_sidecar_tokens = 5
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec_aux_config=SimpleNamespace(
                dflash_compact_owner_count=2,
                dflash_draft_fixed_bytes=allocated_bytes,
            )
        )
    )
    worker._draft_worker = SimpleNamespace(
        alloc_memory_pool=lambda **kwargs: calls.append(kwargs)
    )
    worker.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(get_kv_size_bytes=lambda: allocated_bytes)
    )
    req_pool = SimpleNamespace(req_to_token=torch.empty((2, 16), dtype=torch.int64))

    worker_mod.DFlashWorkerV2.alloc_memory_pool(
        worker,
        memory_pool_config=MemoryPoolConfig(max_total_num_tokens=999),
        req_to_token_pool=req_pool,
    )

    assert len(calls) == 1
    assert calls[0]["memory_pool_config"].max_total_num_tokens == layout.physical_tokens
    assert worker._compact_physical_layout == layout
    assert worker._compact_owner_generation.shape[0] == 3


def test_compact_pool_rejects_actual_request_slots_above_frozen_owner_budget():
    worker_mod = _worker_module()
    worker = object.__new__(worker_mod.DFlashWorkerV2)
    worker.use_compact_draft_cache = True
    worker.draft_window_size = 4
    worker.block_size = 2
    worker.page_size = 1
    worker.dflash_radix_sidecar_tokens = 0
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec_aux_config=SimpleNamespace(dflash_compact_owner_count=1)
        )
    )
    worker._draft_worker = SimpleNamespace(alloc_memory_pool=lambda **kwargs: None)
    req_pool = SimpleNamespace(req_to_token=torch.empty((3, 16), dtype=torch.int64))

    with pytest.raises(RuntimeError, match="exceeds its frozen budget"):
        worker_mod.DFlashWorkerV2.alloc_memory_pool(
            worker,
            memory_pool_config=MemoryPoolConfig(max_total_num_tokens=999),
            req_to_token_pool=req_pool,
        )


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
