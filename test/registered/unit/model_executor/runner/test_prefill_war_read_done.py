from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils import maybe_publish_prefill_war_read_done
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True


def _model_runner(*, spec_algorithm=SpeculativeAlgorithm.NONE, compliant=True):
    return SimpleNamespace(
        spec_algorithm=spec_algorithm,
        attn_backend=SimpleNamespace(
            prefill_shared_reads_end_at_metadata_init=compliant
        ),
        war_fastpath_read_done_event=None,
    )


_DEVICE_MODULE = SimpleNamespace(Event=_Event)


def _batch(mode=ForwardMode.EXTEND):
    return SimpleNamespace(forward_mode=mode)


def test_publishes_recorded_event_when_enabled():
    runner = _model_runner()
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        maybe_publish_prefill_war_read_done(runner, _batch(), _DEVICE_MODULE)
    published = runner.war_fastpath_read_done_event
    assert isinstance(published, _Event) and published.recorded


def test_disabled_when_flag_is_false():
    runner = _model_runner()
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(False):
        maybe_publish_prefill_war_read_done(runner, _batch(), _DEVICE_MODULE)
    assert runner.war_fastpath_read_done_event is None


def test_gates_exclude_non_prefill_unsupported_algorithm_and_noncompliant_backend():
    with envs.SGLANG_ENABLE_PREFILL_WAR_READ_DONE.override(True):
        for runner, batch in (
            # Verify/mixed/decode publish through the decode graph runner.
            (_model_runner(), _batch(ForwardMode.TARGET_VERIFY)),
            (_model_runner(), _batch(ForwardMode.MIXED)),
            (_model_runner(), _batch(ForwardMode.DECODE)),
            # The algorithm has a later prefill reader or unverified ownership.
            (_model_runner(spec_algorithm=SpeculativeAlgorithm.EAGLE), _batch()),
            # Backend has not declared metadata-init compliance.
            (_model_runner(compliant=False), _batch()),
        ):
            maybe_publish_prefill_war_read_done(runner, batch, _DEVICE_MODULE)
            assert runner.war_fastpath_read_done_event is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
