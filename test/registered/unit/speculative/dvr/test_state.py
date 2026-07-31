from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.speculative.dvr.sampling import dvr_proposal_buffer_bytes
from sglang.srt.speculative.dvr.state import DVRStateLifecycle
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakePool:
    def __init__(self, track_slots=(10, 11)):
        self.mamba_ping_pong_track_buffer_size = len(track_slots)
        self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            8, len(track_slots), dtype=torch.int64
        )
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[1] = torch.tensor(
            track_slots
        )

    def get_mamba_ping_pong_keep_idx(self, req):
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - req.mamba_next_track_idx
        return req.mamba_next_track_idx


class FakeAdapter:
    chunk_size = 64

    def __init__(self):
        self.recurrent_workspace = torch.empty(1, 8, 1)
        self.zeroed = []
        self.published = []
        self.staged = []
        self.initialized = []
        self.commits = []
        self.crosses_boundary = torch.tensor([False])

    @staticmethod
    def resolve_request_slots(*, batch):
        return batch.req_pool_indices.to(torch.long), batch.live_slots

    def zero_boundary_state(self, *, indices):
        self.zeroed.extend(indices.tolist())

    def publish_boundary_state(self, **kwargs):
        self.published.append({name: value.tolist() for name, value in kwargs.items()})

    def stage_boundary_state(self, **kwargs):
        valid = kwargs["boundary_conv_steps"] >= 0
        if valid.any():
            self.staged.append(
                {name: value[valid].tolist() for name, value in kwargs.items()}
            )

    def initialize_self_draft_state(self, **kwargs):
        self.initialized.append(
            {name: value.tolist() for name, value in kwargs.items()}
        )

    def commit_accepted_state(self, **kwargs):
        self.commits.append(kwargs)
        boundary_conv_steps = torch.where(
            self.crosses_boundary,
            self.chunk_size - 1 - kwargs["tail_lens_before"],
            torch.full_like(kwargs["tail_lens_before"], -1),
        )
        return self.crosses_boundary, boundary_conv_steps


def make_lifecycle(*, disable_radix=False, track_slots=(10, 11)):
    pool = FakePool(track_slots)
    runner = SimpleNamespace(req_to_token_pool=pool, mambaish_config=object())
    server_args = SimpleNamespace(
        disable_radix_cache=disable_radix,
        mamba_track_interval=64,
        mamba_cache_chunk_size=64,
    )
    lifecycle = DVRStateLifecycle(server_args=server_args, model_runner=runner)
    adapter = FakeAdapter()
    lifecycle.bind_state_adapter(adapter)
    return lifecycle, adapter, pool


def make_batch(*, seq_len, prefix_len=0, next_track=1, track_slots=(10, 11)):
    req = SimpleNamespace(
        rid="r0",
        req_pool_idx=1,
        mamba_pool_idx=torch.tensor([20]),
        mamba_next_track_idx=next_track,
        mamba_ping_pong_track_buffer=torch.tensor(track_slots),
        skip_radix_cache_insert=False,
    )
    batch = SimpleNamespace(
        reqs=[req],
        req_pool_indices=torch.tensor([1]),
        req_to_token_pool=None,
        live_slots=torch.tensor([20]),
        seq_lens=torch.tensor([seq_len]),
        seq_lens_cpu=torch.tensor([seq_len]),
        prefix_lens=[prefix_len],
        batch_size=lambda: 1,
    )
    return req, batch


def attach_pool(batch, pool):
    batch.req_to_token_pool = pool
    return batch


def make_release_case(*, seq_len, accepted_tokens=None):
    lifecycle, adapter, pool = make_lifecycle()
    req, batch = make_batch(seq_len=seq_len, next_track=1)
    attach_pool(batch, pool)
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    if accepted_tokens is not None:
        plan = lifecycle.prepare_rollback(batch)
        adapter.crosses_boundary = torch.tensor([True])
        lifecycle.rollback(
            batch=batch,
            plan=plan,
            accept_lens=torch.tensor([accepted_tokens]),
        )
    return lifecycle, req


def test_dvr_memory_budget_includes_cuda_graph_dummy_row(monkeypatch):
    workspace_state_slots = 2
    helper_args = {}

    def state_slots(*args, **kwargs):
        helper_args.update(kwargs)
        return workspace_state_slots

    monkeypatch.setattr(
        "sglang.srt.layers.attention.dvr.gdn_backend.dvr_gdn_workspace_state_slots",
        state_slots,
    )
    server_args = SimpleNamespace(
        speculative_num_draft_tokens=16,
        speculative_eagle_topk=1,
        max_running_requests=2,
        max_mamba_cache_size=8,
        disable_radix_cache=False,
        dp_size=1,
        enable_dp_attention=False,
    )
    server_args.override = lambda _, **values: vars(server_args).update(values)
    runner = SimpleNamespace(
        server_args=server_args,
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                mamba_cache_per_req=1024,
                layers=(0, 1, 2, 3),
            )
        ),
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        ps=SimpleNamespace(attn_dp_size=1),
        layer_info=SimpleNamespace(start_layer=1, end_layer=3),
        _calculate_mamba_ratio=lambda: 3,
    )

    remaining = KVCacheConfigurator._handle_max_mamba_cache(
        runner, total_rest_memory=1.0
    )

    expected_bytes = (2 + 1) * workspace_state_slots * 1024 + 8 * 1024
    assert helper_args == {"num_layers": 2}
    assert remaining == pytest.approx(1.0 - expected_bytes / (1 << 30))


@pytest.mark.parametrize(
    ("graph_max_bs", "max_running_requests", "expected_capacity"),
    [(4, 48, 48), (64, 48, 64)],
)
def test_self_dvr_proposal_budget_matches_chain_capacity(
    monkeypatch, graph_max_bs, max_running_requests, expected_capacity
):
    available_gb = 2
    monkeypatch.setattr(
        "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
        lambda *args, **kwargs: available_gb,
    )
    monkeypatch.setattr(
        "sglang.srt.mem_cache.kv_cache_configurator.get_world_group",
        lambda: SimpleNamespace(world_size=1, cpu_group=None),
    )
    num_draft_steps = 3
    vocab_size = 10
    runner = SimpleNamespace(
        device="cuda",
        gpu_id=0,
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        server_args=SimpleNamespace(
            mem_fraction_static=1.0,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(max_bs=graph_max_bs)
            ),
            max_running_requests=max_running_requests,
            speculative_num_steps=num_draft_steps,
        ),
        model_config=SimpleNamespace(vocab_size=vocab_size),
        mambaish_config=None,
        post_capture_kv_active=False,
    )

    available_bytes = KVCacheConfigurator._profile_available_bytes(
        runner, pre_model_load_memory=available_gb
    )

    proposal_bytes = (
        expected_capacity * num_draft_steps * vocab_size * torch.float32.itemsize
    )
    assert available_bytes == available_gb * (1 << 30) - proposal_bytes


@pytest.mark.parametrize("disable_overlap", [False, True])
def test_eagle_dvr_proposal_budget_is_request_local(disable_overlap):
    max_running_requests = 48
    dp_size = 2
    vocab_size = 10
    runner = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE,
        server_args=SimpleNamespace(
            speculative_use_rejection_sampling=True,
            disable_overlap_schedule=disable_overlap,
            max_running_requests=max_running_requests,
            dp_size=dp_size,
        ),
        model_config=SimpleNamespace(vocab_size=vocab_size),
    )

    proposal_bytes = dvr_proposal_buffer_bytes(
        spec_algorithm=runner.spec_algorithm,
        server_args=runner.server_args,
        vocab_size=runner.model_config.vocab_size,
    )

    expected_rows = 0 if disable_overlap else max_running_requests // dp_size + 1
    assert proposal_bytes == expected_rows * vocab_size * torch.float32.itemsize


@pytest.mark.parametrize(
    (
        "spec_algorithm",
        "disable_overlap",
        "disable_radix",
        "expected_dual_lanes",
        "expected_ratio",
    ),
    [
        (SpeculativeAlgorithm.NONE, True, False, False, 4),
        (SpeculativeAlgorithm.NONE, False, False, True, 5),
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK, True, False, False, 4),
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK, False, False, True, 5),
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK, True, True, False, 1),
    ],
)
def test_radix_ping_pong_capacity(
    spec_algorithm,
    disable_overlap,
    disable_radix,
    expected_dual_lanes,
    expected_ratio,
):
    server_args = SimpleNamespace(
        disable_radix_cache=disable_radix,
        disable_overlap_schedule=disable_overlap,
        enable_mamba_extra_buffer=lambda: not disable_radix,
        enable_mamba_extra_buffer_lazy=lambda: False,
    )
    runner = SimpleNamespace(
        server_args=server_args,
        spec_algorithm=spec_algorithm,
    )
    uses_dual_lanes = (
        server_args.enable_mamba_extra_buffer()
        and not server_args.disable_overlap_schedule
    )

    assert uses_dual_lanes is expected_dual_lanes
    assert KVCacheConfigurator._calculate_mamba_ratio(runner) == expected_ratio


def test_sync_dvr_pool_disables_generic_snapshots_and_uses_one_radix_lane(
    monkeypatch,
):
    captured = {}

    class FakeHybridReqToTokenPool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "sglang.srt.mem_cache.kv_cache_configurator.HybridReqToTokenPool",
        FakeHybridReqToTokenPool,
    )
    server_args = SimpleNamespace(
        max_mamba_cache_size=32,
        enable_memory_saver=False,
        enable_mamba_extra_buffer=lambda: True,
        enable_mamba_extra_buffer_lazy=lambda: False,
        max_speculative_num_draft_tokens=16,
        speculative_eagle_topk=1,
        disable_overlap_schedule=True,
        enable_linear_replayssm=False,
        linear_replayssm_cache_len=16,
        enable_page_major_kv_layout=False,
        enable_gdn_replayssm_spec=False,
    )
    runner = SimpleNamespace(
        server_args=server_args,
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
        model_config=SimpleNamespace(context_len=4096),
        device="cuda",
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(layers=(0, 1))
        ),
        layer_info=SimpleNamespace(start_layer=0, end_layer=2),
        hybrid_gdn_config=object(),
    )
    KVCacheConfigurator._build_hybrid_req_pool(
        runner,
        max_num_reqs=4,
        extra_max_context_len=16,
    )

    assert captured["speculative_num_draft_tokens"] is None
    assert captured["enable_overlap_schedule"] is False


@pytest.mark.parametrize(
    ("seq_len", "expected_boundary", "expected_tail"),
    [(1, 0, 1), (63, 0, 63), (64, 64, 0), (65, 64, 1)],
)
def test_target_extend_records_live_boundary(seq_len, expected_boundary, expected_tail):
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=seq_len)
    attach_pool(batch, pool)
    batch.seq_lens_cpu = None

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_rollback(batch)

    assert lifecycle.target_boundary_lens[1].item() == expected_boundary
    assert plan.target_cache_slots.tolist() == [20]
    assert plan.tail_lens.tolist() == [expected_tail]
    assert adapter.initialized[-1]["tail_lens"] == [expected_tail]
    assert adapter.zeroed == ([20] if expected_boundary == 0 else [])


def test_new_prefill_boundary_is_copied_to_radix_tracking_slot():
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65, next_track=1)
    attach_pool(batch, pool)

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_rollback(batch)

    assert adapter.published == [
        {
            "source_slots": [20],
            "destination_slots": [10],
            "publish_mask": [True],
        }
    ]
    assert lifecycle.radix_boundary_lens[1].tolist() == [64, -1]
    assert lifecycle.radix_boundary_slots[1].tolist() == [10, -1]
    assert plan.target_cache_slots.tolist() == [20]


def test_sync_single_lane_publishes_first_accepted_boundary():
    lifecycle, adapter, pool = make_lifecycle(track_slots=(10,))
    req, batch = make_batch(seq_len=63, next_track=0, track_slots=(10,))
    attach_pool(batch, pool)

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_rollback(batch)
    adapter.crosses_boundary = torch.tensor([True])
    lifecycle.rollback(batch=batch, plan=plan, accept_lens=torch.tensor([1]))

    assert lifecycle.target_boundary_lens[1].item() == 64
    assert lifecycle.radix_boundary_lens[1].tolist() == [-1]
    assert lifecycle.pending_boundary_conv_steps[1].item() == 0

    req.kv_committed_len = 64
    req.finished_reason = None
    req.effective_kv_committed_len = lambda: req.kv_committed_len
    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 64
    assert req.mamba_next_track_idx == 0
    assert adapter.staged == [
        {
            "request_rows": [1],
            "source_slots": [20],
            "destination_slots": [10],
            "boundary_conv_steps": [0],
        }
    ]


def test_sync_single_lane_uses_replacement_slot_after_radix_donation():
    lifecycle, adapter, pool = make_lifecycle(track_slots=(10,))
    req, batch = make_batch(seq_len=65, next_track=0, track_slots=(10,))
    attach_pool(batch, pool)

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    assert lifecycle.radix_boundary_lens[1].tolist() == [64]

    # Radix owns the original physical checkpoint after donation. The request
    # keeps the same logical lane backed by a fresh slot for its next boundary.
    req.mamba_ping_pong_track_buffer[0] = 12
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping[1, 0] = 12
    plan = lifecycle.prepare_rollback(batch)
    assert lifecycle.radix_boundary_lens[1].tolist() == [-1]
    adapter.crosses_boundary = torch.tensor([True])
    lifecycle.rollback(batch=batch, plan=plan, accept_lens=torch.tensor([63]))

    assert lifecycle.target_boundary_lens[1].item() == 128
    assert lifecycle.radix_boundary_lens[1].tolist() == [-1]

    batch.seq_lens = torch.tensor([128])
    lifecycle.prepare_rollback(batch)

    assert lifecycle.radix_boundary_lens[1].tolist() == [128]
    assert lifecycle.radix_boundary_slots[1].tolist() == [12]
    assert adapter.staged[-1]["destination_slots"] == [12]


def test_warm_partial_extend_needs_no_new_radix_checkpoint():
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65, prefix_len=64, next_track=0)
    attach_pool(batch, pool)

    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_rollback(batch)

    assert adapter.published[-1]["publish_mask"] == [False]
    assert lifecycle.radix_boundary_lens[1].tolist() == [-1, -1]
    assert plan.target_cache_slots.tolist() == [20]
    assert adapter.staged == []


def test_no_radix_boundary_crossing_updates_only_live_state():
    lifecycle, adapter, pool = make_lifecycle(disable_radix=True, track_slots=(10,))
    req, batch = make_batch(seq_len=63, track_slots=(10,))
    attach_pool(batch, pool)
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_rollback(batch)
    adapter.crosses_boundary = torch.tensor([True])

    lifecycle.rollback(batch=batch, plan=plan, accept_lens=torch.tensor([2]))

    assert lifecycle.target_boundary_lens[1].item() == 64
    assert adapter.staged == []

    lifecycle.prepare_for_cache_release(req)

    assert lifecycle.target_boundary_lens[1].item() == -1


def test_radix_boundary_crossing_rotates_publication_lane():
    lifecycle, adapter, pool = make_lifecycle()
    _, batch = make_batch(seq_len=65, next_track=1)
    attach_pool(batch, pool)
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)
    plan = lifecycle.prepare_rollback(batch)
    adapter.crosses_boundary = torch.tensor([True])

    lifecycle.rollback(batch=batch, plan=plan, accept_lens=torch.tensor([63]))

    assert adapter.commits[-1]["target_cache_slots"].tolist() == [20]
    assert lifecycle.target_boundary_lens[1].item() == 128
    assert lifecycle.radix_boundary_lens[1].tolist() == [64, -1]

    batch.seq_lens = torch.tensor([128])
    lifecycle.prepare_rollback(batch)

    assert lifecycle.radix_boundary_lens[1].tolist() == [64, 128]
    assert adapter.staged[-1]["destination_slots"] == [11]


def test_release_selects_latest_visible_radix_boundary():
    lifecycle, req = make_release_case(seq_len=65, accepted_tokens=63)
    req.kv_committed_len = 128
    req.finished_reason = None
    req.effective_kv_committed_len = lambda: req.kv_committed_len

    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 0
    assert lifecycle.target_boundary_lens[1].item() == -1
    assert lifecycle.radix_boundary_lens[1].tolist() == [-1, -1]


def test_prefill_only_release_keeps_fully_processed_prompt_boundary():
    lifecycle, req = make_release_case(seq_len=128)
    req.kv_committed_len = 128
    req.finished_reason = object()
    req.origin_input_ids = list(range(128))
    req.output_ids_through_stop = []
    req.sampling_params = SimpleNamespace(max_new_tokens=0)
    req.effective_kv_committed_len = lambda: req.kv_committed_len

    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 128
    assert req.mamba_next_track_idx == 1
    assert not req.skip_radix_cache_insert


def test_generated_release_excludes_unprocessed_final_token():
    lifecycle, req = make_release_case(seq_len=65, accepted_tokens=63)
    req.kv_committed_len = 128
    req.finished_reason = object()
    req.origin_input_ids = list(range(65))
    req.output_ids_through_stop = list(range(63))
    req.sampling_params = SimpleNamespace(max_new_tokens=63)
    req.effective_kv_committed_len = lambda: req.kv_committed_len

    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 64
    assert req.mamba_next_track_idx == 1


@pytest.mark.parametrize("track_slots", [(10,), (10, 11)])
def test_aligned_generation_keeps_last_visible_boundary(track_slots):
    lifecycle, adapter, pool = make_lifecycle(track_slots=track_slots)
    next_track = 0 if len(track_slots) == 1 else 1
    req, batch = make_batch(
        seq_len=512,
        next_track=next_track,
        track_slots=track_slots,
    )
    attach_pool(batch, pool)
    lifecycle.prepare_target_extend(batch)
    lifecycle.finish_target_extend(batch)

    for seq_len in (512, 528, 544, 560):
        batch.seq_lens = torch.tensor([seq_len])
        plan = lifecycle.prepare_rollback(batch)
        adapter.crosses_boundary = torch.tensor([seq_len == 560])
        lifecycle.rollback(
            batch=batch,
            plan=plan,
            accept_lens=torch.tensor([16]),
        )

    assert lifecycle.target_boundary_lens[1].item() == 576
    assert lifecycle.pending_boundary_conv_steps[1].item() == 15
    assert 512 in lifecycle.radix_boundary_lens[1].tolist()

    req.kv_committed_len = 576
    req.finished_reason = object()
    req.origin_input_ids = list(range(64))
    req.output_ids_through_stop = list(range(512))
    req.sampling_params = SimpleNamespace(max_new_tokens=512)
    req.effective_kv_committed_len = lambda: req.kv_committed_len
    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 512
    assert adapter.staged == []


def test_overlap_extra_round_preserves_last_visible_boundary():
    lifecycle, adapter, pool = make_lifecycle(track_slots=(10, 11))
    req, _ = make_batch(seq_len=640, next_track=0, track_slots=(10, 11))
    lifecycle.target_boundary_lens[1] = 640
    lifecycle.pending_boundary_conv_steps[1] = 15
    lifecycle.radix_boundary_lens[1] = torch.tensor([512, 576])
    lifecycle.radix_boundary_slots[1] = torch.tensor([10, 11])

    req.kv_committed_len = 640
    req.finished_reason = object()
    req.origin_input_ids = list(range(64))
    req.output_ids_through_stop = list(range(512))
    req.sampling_params = SimpleNamespace(max_new_tokens=512)
    req.effective_kv_committed_len = lambda: req.kv_committed_len
    lifecycle.prepare_for_cache_release(req)

    assert req.mamba_last_track_seqlen == 512
    assert req.mamba_next_track_idx == 1
    assert adapter.staged == []
