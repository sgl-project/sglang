"""DFLASH on DSpark's confidence-scheduled verify: the planner takes the worker's
confidence, acceptance stops at the planned verify length, and the confidence
feature reads the selector row each draft token came from."""

import types

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.models.dflash import CandidateSelector
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.srt.speculative.dspark_components import dspark_planner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _greedy_accept(candidates, target_predict, cutoff_verify_lens):
    worker = types.SimpleNamespace(
        _selector_sample=None,
        _tp_sync=types.SimpleNamespace(sync=lambda site, tensor: tensor),
        _use_triton_accept_bonus=False,
        block_size=candidates.shape[1],
    )
    return worker_mod.DFlashWorkerV2._accept_block(
        worker,
        candidates=candidates,
        next_token_logits=None,
        sampling_info=None,
        draft_input=None,
        prefix_lens=torch.tensor([10, 20]),
        bs=candidates.shape[0],
        target_predict=target_predict,
        cutoff_verify_lens=cutoff_verify_lens,
    )


def test_greedy_accept_stops_at_planned_verify_len():
    # Row 0's drafts all match; row 1 mismatches at its third draft.
    candidates = torch.tensor([[5, 1, 2, 3, 4], [6, 7, 8, 0, 9]])
    target_predict = torch.tensor([[1, 2, 3, 4, 11], [7, 8, 12, 13, 14]])

    full = _greedy_accept(candidates, target_predict, None)
    assert full[0].tolist() == [4, 2]
    assert full[5] is None

    accept_len, commit_lens, bonus, out_tokens, _, cap_trim_lens = _greedy_accept(
        candidates, target_predict, torch.tensor([3, 5], dtype=torch.int32)
    )
    # Row 0 verified 2 drafts: it keeps them and takes the target token after
    # the last verified row as its bonus. Row 1 rejected inside its window.
    assert accept_len.tolist() == [2, 2]
    assert cap_trim_lens.tolist() == [2, 0]
    assert bonus.tolist() == [3, 12]
    assert commit_lens.tolist() == [3, 3]
    assert out_tokens[:, :3].tolist() == [[1, 2, 3], [7, 8, 12]]


def test_greedy_accept_with_full_verify_len_matches_static():
    torch.manual_seed(0)
    candidates = torch.randint(0, 4, (16, 8))
    target_predict = torch.randint(0, 4, (16, 8))
    static = _greedy_accept(candidates, target_predict, None)
    capped = _greedy_accept(
        candidates, target_predict, torch.full((16,), 8, dtype=torch.int32)
    )
    for got, ref in zip(capped[:4], static[:4]):
        torch.testing.assert_close(got.long(), ref.long())
    assert capped[5].tolist() == [0] * 16


class _RecordingEstimator:
    def feature_and_predict(self, scores, temperatures):
        self.scores, self.temperatures = scores, temperatures
        return scores.amax(dim=-1), scores.amax(dim=-1).sigmoid()


def test_feature_scores_the_selector_row_each_token_was_drawn_from():
    selector = CandidateSelector(hidden_size=4, vocab_size=64, state_rank=2, top_k=4)
    torch.manual_seed(3)
    bs, slots, top_k = 3, 5, 4
    candidate_ids = torch.stack(
        [torch.randperm(64)[: slots * top_k].view(slots, top_k) for _ in range(bs)]
    )
    scores = torch.randn(bs, slots, top_k, top_k)
    temperatures = torch.tensor([0.7, 1.3, 1e-5])
    greedy_mask = torch.tensor([False, False, True])
    tokens, _ = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=torch.rand(bs, slots),
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )

    estimator = _RecordingEstimator()
    worker_mod._walked_confidence(
        estimator,
        candidate_ids=candidate_ids,
        scores=scores,
        tokens=tokens,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )

    for b in range(bs):
        previous = 0
        for slot in range(slots):
            torch.testing.assert_close(
                estimator.scores[b, slot], scores[b, slot, previous]
            )
            previous = int((candidate_ids[b, slot] == tokens[b, slot]).nonzero())
    # A greedy row is scored at temperature 1, a sampled row at its own.
    torch.testing.assert_close(estimator.temperatures, torch.tensor([0.7, 1.3, 1.0]))


def _patch_planner_context(monkeypatch):
    spec = types.SimpleNamespace(
        speculative_dspark_align_verify_tokens_to_graph_tier=False,
        speculative_dspark_confidence_sts_path=None,
        speculative_dspark_sps_table_path=None,
    )
    monkeypatch.setattr(dspark_planner, "get_spec", lambda: spec)
    monkeypatch.setattr(
        dspark_planner,
        "get_schedule",
        lambda: types.SimpleNamespace(
            disable_overlap_schedule=False, max_running_requests=4
        ),
    )
    monkeypatch.setattr(dspark_planner, "is_dp_attention_enabled", lambda: False)


def _planner(*, external_confidence):
    return dspark_planner.DSparkVerifyPlanner(
        draft_model=types.SimpleNamespace(),
        gamma=3,
        model_runner=types.SimpleNamespace(decode_cuda_graph_runner=None),
        device=torch.device("cpu"),
        tp_rank=1,
        verify_num_draft_tokens=4,
        tp_sync=types.SimpleNamespace(sync=lambda site, tensor: tensor),
        external_confidence=external_confidence,
    )


def test_planner_schedules_external_confidence_without_a_head(monkeypatch):
    _patch_planner_context(monkeypatch)
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("cap-accept"):
        with pytest.raises(ValueError, match="confidence head"):
            _planner(external_confidence=False)
        planner = _planner(external_confidence=True)
        assert planner.schedules_verify_budget and planner.carries_confidence

        confidence = torch.tensor(
            [[0.9, 0.9, 0.9], [0.2, 0.9, 0.9], [0.9, 0.8, 0.1]], dtype=torch.float32
        )
        layout = planner.schedule_layout(
            req_pool_indices=torch.tensor([1, 2, 3]),
            prefix_lens=torch.tensor([5, 6, 7]),
            device=torch.device("cpu"),
            confidence=confidence,
            budget=4,
        )
    # The four most likely drafts by survival: request 0's three and request
    # 2's first; each request also verifies its anchor.
    assert layout.verify_lens.tolist() == [4, 1, 2]


def test_compact_plan_on_the_full_block_graph_verifies_every_draft(monkeypatch):
    _patch_planner_context(monkeypatch)
    monkeypatch.setattr(
        dspark_planner,
        "build_sps_cost_table",
        lambda **_: dspark_planner.SpsCostTable(
            sample_batch_tokens=[1, 16],
            sample_steps_per_sec=[2.0, 1.0],
            max_batch_tokens=16,
        ),
    )
    # Compact graphs of 4, 8, 12 and 16 tokens for blocks of 4 tokens.
    runner = types.SimpleNamespace(
        ragged_verify_mode=True, capture_num_tokens=[4, 8, 12, 16], max_bs=4
    )
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
        planner = dspark_planner.DSparkVerifyPlanner(
            draft_model=types.SimpleNamespace(),
            gamma=3,
            model_runner=types.SimpleNamespace(decode_cuda_graph_runner=runner),
            device=torch.device("cpu"),
            tp_rank=1,
            verify_num_draft_tokens=4,
            tp_sync=types.SimpleNamespace(sync=lambda site, tensor: tensor),
            external_confidence=True,
        )

        def plan(bs, budget):
            return planner.schedule_layout(
                req_pool_indices=torch.arange(1, bs + 1),
                prefix_lens=torch.full((bs,), 5),
                device=torch.device("cpu"),
                confidence=torch.tensor([[0.9, 0.5, 0.1]] * bs),
                budget=budget,
            )

        # Any plan of a single request replays the 4-token graph of its block.
        assert not planner.plans_verify_budget(num_reqs=1)
        single = plan(1, 0)
        assert single.is_full_block(4) and plan(1, 2) is single

        # Three requests: 9 planned tokens replay the 12-token graph of the
        # full blocks, so every draft is verified; 7 fit the 8-token graph.
        assert planner.plans_verify_budget(num_reqs=3)
        full = plan(3, 6)
        assert full.is_full_block(4) and full.verify_lens.tolist() == [4, 4, 4]
        trimmed = plan(3, 4)
        assert not trimmed.is_full_block(4)
        assert trimmed.graph_num_tokens == 8 and int(trimmed.verify_lens.sum()) == 7


def test_observe_counts_the_drafts_the_target_verified():
    class Recorder:
        def observe(self, x, num_verified, num_accepted):
            self.calls.append((x, num_verified.tolist(), num_accepted.tolist()))

    estimator = Recorder()
    estimator.calls = []
    feature = torch.zeros(3, 4)
    worker = types.SimpleNamespace(
        _acceptance_estimator=estimator, _draft_feature=feature, block_size=5
    )
    accept_len = torch.tensor([2, 4, 0], dtype=torch.int32)
    worker_mod.DFlashWorkerV2._observe_acceptance(
        worker, accept_len=accept_len, layout=None
    )
    layout = types.SimpleNamespace(
        verify_lens=torch.tensor([3, 5, 1], dtype=torch.int32)
    )
    worker_mod.DFlashWorkerV2._observe_acceptance(
        worker, accept_len=accept_len, layout=layout
    )
    # Without a layout every draft was verified; with one, verify_lens - 1.
    assert [call[1:] for call in estimator.calls] == [
        ([4, 4, 4], [2, 4, 0]),
        ([2, 4, 0], [2, 4, 0]),
    ]
    assert all(call[0] is feature for call in estimator.calls)


def test_ragged_verify_gates_open_for_dflash_outside_static(monkeypatch):
    from sglang.srt.managers import overlap_utils
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    assert SpeculativeAlgorithm.DFLASH.supports_ragged_verify()
    assert not SpeculativeAlgorithm.EAGLE.supports_ragged_verify()

    def needs_relay(algorithm, mode):
        monkeypatch.setattr(
            overlap_utils,
            "get_spec",
            lambda: types.SimpleNamespace(speculative_algorithm=algorithm),
        )
        with envs.SGLANG_RAGGED_VERIFY_MODE.override(mode):
            return overlap_utils.decide_needs_confidence_relay()

    assert needs_relay("DFLASH", "cap-accept") and needs_relay("DFLASH", "compact")
    assert not needs_relay("DFLASH", "static")
    assert not needs_relay("EAGLE", "compact")


def test_forced_budget_frac_trims_a_compact_plan_without_a_table(monkeypatch):
    from sglang.srt.managers.scheduler_components import internal_state

    _patch_planner_context(monkeypatch)
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
        planner = _planner(external_confidence=True)
    # Compact without a profiled table verifies every draft, unless the
    # profiler pins a budget.
    assert planner.is_verify_all
    planner.set_forced_budget_frac(0.5)
    assert not planner.is_verify_all
    planner.set_forced_budget_frac(None)
    assert planner.is_verify_all

    # A simulated acceptance above 0 could run past the pinned lengths.
    controller = types.SimpleNamespace(
        scheduler=types.SimpleNamespace(
            spec_algorithm=worker_mod.SpeculativeAlgorithm.DFLASH,
            draft_worker=types.SimpleNamespace(set_dspark_forced_budget_frac=None),
        )
    )
    controller._require_dspark_worker = types.MethodType(
        internal_state.SchedulerInternalStateController._require_dspark_worker,
        controller,
    )
    validate = (
        internal_state.SchedulerInternalStateController._validate_dspark_budget_frac
    )
    validate(controller, "dspark_force_budget_frac", 0.5)
    with envs.SGLANG_SIMULATE_ACC_LEN.override(3.0):
        validate(controller, "dspark_force_budget_frac", None)
        with pytest.raises(internal_state.UpdateRejected, match="SIMULATE_ACC_LEN"):
            validate(controller, "dspark_force_budget_frac", 0.5)


def test_compact_is_refused_on_a_target_backend_without_ragged_verify(monkeypatch):
    import contextlib

    monkeypatch.setattr(worker_mod, "draft_pp_context", contextlib.nullcontext)
    monkeypatch.setattr(
        worker_mod, "draft_tp_context", lambda *_: contextlib.nullcontext()
    )
    monkeypatch.setattr(worker_mod, "mambaish_config", lambda _: None)

    def init_backends(supports_ragged_verify_graph):
        worker = types.SimpleNamespace(
            _draft_worker=types.SimpleNamespace(init_attention_backends=lambda: None),
            draft_owns_attention=False,
            model_runner=types.SimpleNamespace(
                model_config=None,
                attn_backend=types.SimpleNamespace(
                    supports_ragged_verify_graph=supports_ragged_verify_graph
                ),
            ),
        )
        worker_mod.DFlashWorkerV2.init_attention_backends(worker)

    with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
        init_backends(True)
        with pytest.raises(ValueError, match="compact"):
            init_backends(False)
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("cap-accept"):
        init_backends(False)
