from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.shared_read_event import (
    maybe_publish_prefill_shared_read_done,
)
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _published_server_args():
    with get_context().override_server_args():
        yield


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
    )
    return runner


_DEVICE_MODULE = SimpleNamespace(Event=_Event)


def _batch(mode=ForwardMode.EXTEND):
    return SimpleNamespace(forward_mode=mode)


def _prepare_and_publish(runner, batch):
    shared_read_ends = runner.attn_backend.resolve_prefill_shared_read_ends(
        batch, num_qo_tokens=8
    )
    maybe_publish_prefill_shared_read_done(runner, shared_read_ends, _DEVICE_MODULE)


def test_publishes_recorded_event_by_default():
    runner = _model_runner()
    _prepare_and_publish(runner, _batch())
    published = runner.shared_read_done_event
    assert isinstance(published, _Event) and published.recorded


def test_disabled_when_flag_is_false():
    runner = _model_runner()
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(False):
        _prepare_and_publish(runner, _batch())
    assert runner.shared_read_done_event is None


@pytest.mark.parametrize(
    "memory",
    [
        {"enable_hierarchical_cache": True},
        {"hicache_storage_backend": "mooncake"},
        {"enable_lmcache": True},
    ],
)
def test_kv_offload_keeps_the_coarse_barrier(memory):
    runner = _model_runner()
    with get_memory().override(**memory):
        _prepare_and_publish(runner, _batch())
    assert runner.shared_read_done_event is None


def test_dflash_target_prefill_publishes():
    runner = _model_runner(spec_algorithm=SpeculativeAlgorithm.DFLASH)
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        _prepare_and_publish(runner, _batch())
    published = runner.shared_read_done_event
    assert isinstance(published, _Event) and published.recorded


def test_gates_exclude_non_prefill_unsupported_algorithm_and_noncompliant_backend():
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        for runner, batch in (
            # Verify/mixed/decode publish through the decode graph runner.
            (_model_runner(), _batch(ForwardMode.TARGET_VERIFY)),
            (_model_runner(), _batch(ForwardMode.MIXED)),
            (_model_runner(), _batch(ForwardMode.DECODE)),
            # The algorithm has a later prefill reader.
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
                with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
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
                runner = _model_runner(spec_algorithm=algorithm)
                with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
                    _prepare_and_publish(runner, _batch())
                event = runner.shared_read_done_event
                self.assertEqual(event is not None, expected)
                if expected:
                    self.assertTrue(event.recorded)

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
                runner = _model_runner(spec_algorithm=algorithm, declared=declared)
                with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(enabled):
                    _prepare_and_publish(runner, _batch(mode))
                self.assertIsNone(runner.shared_read_done_event)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
