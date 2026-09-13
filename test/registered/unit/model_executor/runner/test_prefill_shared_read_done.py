from types import SimpleNamespace

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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True


def _model_runner(*, spec_algorithm=SpeculativeAlgorithm.NONE, compliant=True):
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
        batch, num_qo_tokens=8, spec_algorithm=runner.spec_algorithm
    )
    maybe_publish_prefill_shared_read_done(runner, shared_read_ends, _DEVICE_MODULE)


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
    with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(False):
        _prepare_and_publish(runner, _batch())
    published = runner.shared_read_done_event
    assert isinstance(published, _Event) and published.recorded


def test_gates_exclude_non_prefill_unsupported_algorithm_and_noncompliant_backend():
    with envs.SGLANG_FORCE_PREFILL_COARSE_WAR_BARRIER.override(False):
        for runner, batch in (
            # Verify/mixed/decode publish through the decode graph runner.
            (_model_runner(), _batch(ForwardMode.TARGET_VERIFY)),
            (_model_runner(), _batch(ForwardMode.MIXED)),
            (_model_runner(), _batch(ForwardMode.DECODE)),
            # The algorithm has a later prefill reader or unverified ownership.
            (_model_runner(spec_algorithm=SpeculativeAlgorithm.EAGLE), _batch()),
            # Backend has not declared a pre-replay prefill read end.
            (_model_runner(compliant=False), _batch()),
        ):
            _prepare_and_publish(runner, batch)
            assert runner.shared_read_done_event is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
