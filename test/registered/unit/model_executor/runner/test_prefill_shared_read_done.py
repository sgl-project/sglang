from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils import (
    maybe_publish_prefill_shared_read_done,
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


def _model_runner(
    *, spec_algorithm=SpeculativeAlgorithm.NONE, compliant=True, declared=None
):
    if declared is None:
        declared = SharedReadEnds.PRE_REPLAY if compliant else SharedReadEnds.UNKNOWN
    attn_backend = AttentionBackend()
    attn_backend.shared_read_ends = lambda _forward_mode: declared
    runner = SimpleNamespace(
        spec_algorithm=spec_algorithm,
        attn_backend=attn_backend,
        shared_read_done_event=None,
        prefill_shared_read_stager=None,
    )
    return runner


_DEVICE_MODULE = SimpleNamespace(Event=_Event)


def _batch(mode=ForwardMode.EXTEND):
    return SimpleNamespace(forward_mode=mode)


def _prepare_and_publish(runner, batch, device_module=_DEVICE_MODULE):
    shared_read_ends = runner.attn_backend.resolve_prefill_shared_read_ends(
        batch, num_qo_tokens=8
    )
    maybe_publish_prefill_shared_read_done(
        runner, batch, shared_read_ends, device_module
    )


def test_publishes_recorded_event_by_default():
    runner = _model_runner()
    _prepare_and_publish(runner, _batch())
    published = runner.shared_read_done_event
    assert isinstance(published, _Event) and published.recorded


def test_forced_prefill_coarse_barrier_skips_event():
    runner = _model_runner()
    with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(True):
        _prepare_and_publish(runner, _batch())
    assert runner.shared_read_done_event is None


@pytest.mark.parametrize(
    "algorithm", (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK)
)
def test_dflash_family_target_prefill_publishes(algorithm):
    runner = _model_runner(spec_algorithm=algorithm)
    runner.prefill_shared_read_stager = Mock(return_value=False)
    with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(False):
        _prepare_and_publish(runner, _batch())
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
    with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(False):
        _prepare_and_publish(runner, batch, SimpleNamespace(Event=calls.Event))
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
    with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(not enabled):
        _prepare_and_publish(runner, _batch(mode))
    runner.prefill_shared_read_stager.assert_not_called()
    assert runner.shared_read_done_event is None


def test_gates_exclude_non_prefill_unsupported_algorithm_and_noncompliant_backend():
    with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(False):
        for runner, batch in (
            # Verify/mixed/decode publish through the decode graph runner.
            (_model_runner(), _batch(ForwardMode.TARGET_VERIFY)),
            (_model_runner(), _batch(ForwardMode.MIXED)),
            (_model_runner(), _batch(ForwardMode.DECODE)),
            # The algorithm has a later prefill reader and no stager.
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
            _prepare_and_publish(runner, batch)
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
                with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(False):
                    _prepare_and_publish(runner, _batch())
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
                with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(False):
                    _prepare_and_publish(runner, batch)
                event = runner.shared_read_done_event
                self.assertEqual(event is not None, expected)
                if expected:
                    self.assertTrue(event.recorded)
                    runner.prefill_shared_read_stager.assert_not_called()
                else:
                    runner.prefill_shared_read_stager.assert_called_once_with(batch)

    def test_opt_in_does_not_bypass_mode_backend_or_opt_out(self):
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
                runner = _model_runner(spec_algorithm=algorithm, declared=declared)
                with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(not enabled):
                    _prepare_and_publish(runner, _batch(mode))
                self.assertIsNone(runner.shared_read_done_event)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
