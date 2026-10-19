from types import SimpleNamespace
from unittest.mock import Mock, call, create_autospec

import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.shared_read_event import (
    maybe_publish_prefill_shared_read_done,
    refresh_shared_read_done_event,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True


def _model_runner(*, spec_algorithm=SpeculativeAlgorithm.NONE, compliant=True):
    declared = SharedReadEnds.PRE_REPLAY if compliant else SharedReadEnds.UNKNOWN
    # Spec'd against the real ABC so a rename fails here, not at runtime.
    attn_backend = create_autospec(AttentionBackend, instance=True)
    attn_backend.shared_read_ends.return_value = declared
    return SimpleNamespace(
        spec_algorithm=spec_algorithm,
        attn_backend=attn_backend,
        shared_read_done_event=None,
        prefill_shared_read_stager=None,
    )


_DEVICE_MODULE = SimpleNamespace(Event=_Event)


def _batch(mode=ForwardMode.EXTEND):
    return SimpleNamespace(forward_mode=mode)


def test_publishes_recorded_event_when_enabled():
    runner = _model_runner()
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        maybe_publish_prefill_shared_read_done(runner, _batch(), _DEVICE_MODULE)
    published = runner.shared_read_done_event
    assert isinstance(published, _Event) and published.recorded


def test_disabled_when_flag_is_false():
    runner = _model_runner()
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(False):
        maybe_publish_prefill_shared_read_done(runner, _batch(), _DEVICE_MODULE)
    assert runner.shared_read_done_event is None


@pytest.mark.parametrize(
    "algorithm", (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK)
)
def test_dflash_family_target_prefill_publishes(algorithm):
    runner = _model_runner(spec_algorithm=algorithm)
    runner.prefill_shared_read_stager = Mock(return_value=False)
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        maybe_publish_prefill_shared_read_done(runner, _batch(), _DEVICE_MODULE)
    published = runner.shared_read_done_event
    assert isinstance(published, _Event) and published.recorded
    runner.prefill_shared_read_stager.assert_not_called()


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize(
    "algorithm",
    [
        SpeculativeAlgorithm.EAGLE,
        SpeculativeAlgorithm.EAGLE3,
        SpeculativeAlgorithm.FROZEN_KV_MTP,
        SpeculativeAlgorithm.STANDALONE,
        SpeculativeAlgorithm.NGRAM,
        SpeculativeAlgorithm.UNO,
        CustomSpecAlgo("TEST_PREFILL", lambda _: object),
    ],
)
def test_speculative_prefill_publishes_only_after_staging(algorithm, staged):
    runner, batch = _model_runner(spec_algorithm=algorithm), _batch()
    calls = Mock()
    runner.prefill_shared_read_stager = calls.stage
    calls.stage.return_value = staged
    calls.Event.side_effect = _Event
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        maybe_publish_prefill_shared_read_done(
            runner, batch, SimpleNamespace(Event=calls.Event)
        )
    assert calls.mock_calls == [call.stage(batch)] + ([call.Event()] if staged else [])
    published = runner.shared_read_done_event
    if staged:
        assert isinstance(published, _Event) and published.recorded
    else:
        assert published is None


@pytest.mark.parametrize(
    "enabled,mode,compliant",
    [
        (False, ForwardMode.EXTEND, True),
        (True, ForwardMode.TARGET_VERIFY, True),
        (True, ForwardMode.MIXED, True),
        (True, ForwardMode.DECODE, True),
        (True, ForwardMode.EXTEND, False),
    ],
)
def test_prefill_gates_skip_staging(enabled, mode, compliant):
    runner = _model_runner(
        spec_algorithm=SpeculativeAlgorithm.EAGLE, compliant=compliant
    )
    runner.prefill_shared_read_stager = Mock(return_value=True)
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(enabled):
        maybe_publish_prefill_shared_read_done(runner, _batch(mode), _DEVICE_MODULE)
    runner.prefill_shared_read_stager.assert_not_called()
    assert runner.shared_read_done_event is None


def test_gates_exclude_non_prefill_unsupported_algorithm_and_noncompliant_backend():
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        for runner, batch in (
            # Verify/mixed/decode publish through the decode graph runner.
            (_model_runner(), _batch(ForwardMode.TARGET_VERIFY)),
            (_model_runner(), _batch(ForwardMode.MIXED)),
            (_model_runner(), _batch(ForwardMode.DECODE)),
            # The algorithm has a later prefill reader or unverified ownership.
            (_model_runner(spec_algorithm=SpeculativeAlgorithm.EAGLE), _batch()),
            (
                _model_runner(
                    spec_algorithm=CustomSpecAlgo("TEST_PREFILL", lambda _: object)
                ),
                _batch(),
            ),
            # Backend has not declared a pre-replay prefill read end.
            (_model_runner(compliant=False), _batch()),
        ):
            maybe_publish_prefill_shared_read_done(runner, batch, _DEVICE_MODULE)
            assert runner.shared_read_done_event is None


class _TargetOnlyPrefillAlgo(CustomSpecAlgo):
    def supports_prefill_shared_read_done(self) -> bool:
        return True


class TestPrefillReadDoneCapability(CustomTestCase):
    def test_builtin_allowlist_is_unchanged(self):
        allowed = {
            SpeculativeAlgorithm.NONE,
            SpeculativeAlgorithm.DFLASH,
            SpeculativeAlgorithm.DSPARK,
        }
        for algorithm in SpeculativeAlgorithm:
            with self.subTest(algorithm=algorithm):
                runner = _model_runner(spec_algorithm=algorithm)
                with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
                    maybe_publish_prefill_shared_read_done(
                        runner, _batch(), _DEVICE_MODULE
                    )
                self.assertEqual(
                    runner.shared_read_done_event is not None, algorithm in allowed
                )

    def test_custom_algorithm_requires_explicit_opt_in(self):
        for algorithm_type, expected in (
            (CustomSpecAlgo, False),
            (_TargetOnlyPrefillAlgo, True),
        ):
            with self.subTest(algorithm_type=algorithm_type):
                algorithm = algorithm_type(
                    "TEST_PREFILL", lambda _: object, supports_overlap=True
                )
                runner, batch = _model_runner(spec_algorithm=algorithm), _batch()
                runner.prefill_shared_read_stager = Mock(return_value=False)
                with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
                    maybe_publish_prefill_shared_read_done(
                        runner, batch, _DEVICE_MODULE
                    )
                event = runner.shared_read_done_event
                self.assertEqual(event is not None, expected)
                if expected:
                    self.assertTrue(event.recorded)
                    runner.prefill_shared_read_stager.assert_not_called()
                else:
                    runner.prefill_shared_read_stager.assert_called_once_with(batch)

    def test_opt_in_does_not_bypass_mode_backend_or_flag(self):
        algorithm = _TargetOnlyPrefillAlgo("TEST_PREFILL", lambda _: object)
        cases = [(False, ForwardMode.EXTEND, SharedReadEnds.PRE_REPLAY)]
        cases.extend(
            (True, mode, SharedReadEnds.PRE_REPLAY)
            for mode in ForwardMode
            if mode != ForwardMode.EXTEND
        )
        cases.extend(
            (True, ForwardMode.EXTEND, declared)
            for declared in SharedReadEnds
            if declared != SharedReadEnds.PRE_REPLAY
        )
        for enabled, mode, declared in cases:
            with self.subTest(enabled=enabled, mode=mode, declared=declared):
                runner = _model_runner(spec_algorithm=algorithm)
                runner.attn_backend.shared_read_ends.return_value = declared
                with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(enabled):
                    maybe_publish_prefill_shared_read_done(
                        runner, _batch(mode), _DEVICE_MODULE
                    )
                self.assertIsNone(runner.shared_read_done_event)


def test_refresh_shared_read_done_event_noop_when_unset():
    runner = _model_runner()
    runner.device_module = _DEVICE_MODULE
    refresh_shared_read_done_event(runner)
    assert runner.shared_read_done_event is None


def test_refresh_shared_read_done_event_replaces_existing_event():
    initial = _Event()
    initial.record()
    runner = _model_runner()
    runner.shared_read_done_event = initial
    runner.decode_cuda_graph_runner = SimpleNamespace(device_module=_DEVICE_MODULE)

    refresh_shared_read_done_event(runner)

    refreshed = runner.shared_read_done_event
    assert isinstance(refreshed, _Event)
    assert refreshed is not initial
    assert refreshed.recorded


def test_ngram_target_verify_refreshes_shared_read_done_after_kv_move(monkeypatch):
    import torch
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    import sglang.srt.speculative.ngram_worker as ngram_worker_mod
    from sglang.srt.speculative.ngram_worker import NGRAMWorker

    order = []
    initial_event = _Event()
    initial_event.record()

    def _make_event():
        ev = _Event()
        order.append(("event_created", ev))
        return ev

    runner = SimpleNamespace(
        shared_read_done_event=None,
        watermark_state=None,
        is_mamba_model=False,
        model_config=SimpleNamespace(hybrid_linear_attn_config=None),
        device_module=SimpleNamespace(Event=_make_event),
    )

    def _target_forward(batch, pp_proxy_tensors=None, is_verify=False):
        assert is_verify
        runner.shared_read_done_event = initial_event
        order.append(("target_forward", initial_event))
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.zeros((1, 4))),
            can_run_cuda_graph=True,
        )

    worker = NGRAMWorker.__new__(NGRAMWorker)
    worker.device = "cpu"
    worker.draft_token_num = 2
    worker.speculative_num_draft_tokens = 2
    worker.model_runner = runner
    worker._target_worker = SimpleNamespace(
        model_runner=runner,
        forward_batch_generation=_target_forward,
    )
    worker.token_to_kv_pool_allocator = SimpleNamespace()
    worker.ngram_corpus = SimpleNamespace(erase_match_state=lambda rids: None)
    worker._prev_decode_rids = set()
    worker._prepare_for_speculative_decoding = lambda batch: None
    worker._update_ngram_corpus = lambda batch: order.append(("update_corpus", None))

    monkeypatch.setattr(
        ngram_worker_mod,
        "commit_mamba_states_after_verify",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ngram_worker_mod,
        "eagle_sample",
        lambda *args, **kwargs: (
            torch.tensor([10, 11], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([[0, 1]], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        ngram_worker_mod,
        "move_accept_tokens_to_target_kvcache",
        lambda *args, **kwargs: order.append(("move_kv", runner.shared_read_done_event)),
    )

    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0")],
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(),
        has_grammar=False,
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        out_cache_loc=torch.tensor([1, 2], dtype=torch.int64),
        return_logprob=False,
    )

    worker.forward_batch_generation(batch)

    # During move_accept_tokens_to_target_kvcache, the old graph event was still active;
    # immediately after move_accept_tokens_to_target_kvcache, a fresh event was recorded
    # and published before corpus update / return.
    op_names = [name for name, _ in order]
    assert op_names == ["target_forward", "move_kv", "event_created", "update_corpus"]
    assert dict(order)["move_kv"] is initial_event
    assert runner.shared_read_done_event is not initial_event
    assert runner.shared_read_done_event.recorded


@pytest.mark.parametrize("topk", [1, 2])
def test_eagle_verify_refreshes_shared_read_done_when_tree_compaction_moves_kv(
    monkeypatch, topk
):
    import contextlib
    import torch
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    import sglang.srt.speculative.eagle_worker_common as eagle_common_mod
    from sglang.srt.speculative.eagle_worker_common import run_eagle_verify

    order = []
    initial_event = _Event()
    initial_event.record()

    runner = SimpleNamespace(
        shared_read_done_event=None,
        watermark_state=None,
        is_mamba_model=False,
        model_config=SimpleNamespace(hybrid_linear_attn_config=None),
        attn_backend=SimpleNamespace(verify_mask=None),
        device_module=SimpleNamespace(Event=_Event),
    )

    def _target_forward(
        batch=None,
        forward_batch=None,
        is_verify=False,
        pp_proxy_tensors=None,
        skip_sample=False,
    ):
        assert is_verify
        runner.shared_read_done_event = initial_event
        order.append("target_forward")
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.zeros((2, 4))),
            can_run_cuda_graph=True,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

    target_worker = SimpleNamespace(
        model_runner=runner,
        forward_batch_generation=_target_forward,
    )

    monkeypatch.setattr(
        eagle_common_mod,
        "commit_mamba_states_after_verify",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        eagle_common_mod,
        "eagle_prepare_for_verify",
        lambda verify_input, req_to_token_pool, batch, target_worker: (
            SimpleNamespace(mark_forward_metadata_ready=lambda: None),
            True,
        ),
    )
    monkeypatch.setattr(
        eagle_common_mod,
        "eagle_sample",
        lambda *args, **kwargs: (
            torch.tensor([10, 11], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([[0, 1]], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        eagle_common_mod,
        "fill_bonus_tokens_func",
        lambda *args, **kwargs: None,
    )

    def _fake_finalize(batch, accept_index, accept_lens, predict, *args, **kwargs):
        order.append("finalize_tree_path")
        assert runner.shared_read_done_event is initial_event
        return predict

    monkeypatch.setattr(
        eagle_common_mod,
        "_finalize_accept_tree_path",
        _fake_finalize,
    )

    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0")],
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(
            draft_token_num=2,
            positions=torch.tensor([4, 5], dtype=torch.int64),
        ),
        has_grammar=False,
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        out_cache_loc=torch.tensor([1, 2], dtype=torch.int64),
        return_logprob=False,
        hisparse_coordinator=None,
        mamba_track_indices=None,
    )
    allocator = SimpleNamespace(get_kvcache=lambda: SimpleNamespace())

    run_eagle_verify(
        batch=batch,
        target_worker=target_worker,
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=allocator,
        plan_stream=None,
        plan_stream_ctx=contextlib.nullcontext(),
        topk=topk,
        num_draft_tokens=2,
        device="cpu",
        metadata_ready_pre_pad=False,
        finalize_tree_path=True,
    )

    if topk > 1:
        assert order == ["target_forward", "finalize_tree_path"]
        assert runner.shared_read_done_event is not initial_event
        assert runner.shared_read_done_event.recorded
    else:
        assert order == ["target_forward"]
        assert runner.shared_read_done_event is initial_event


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
