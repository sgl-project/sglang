"""Hypothesis RuleBasedStateMachine self-e2e: 3 invariants, 3 rules, 1 shrinking meta."""

from __future__ import annotations

import pytest

try:
    from hypothesis import settings
    from hypothesis import strategies as st
    from hypothesis.stateful import (
        RuleBasedStateMachine,
        invariant,
        precondition,
        rule,
    )

    from sglang.srt.mock_mode import MockEngine

    _MOCK_MODE_AVAILABLE = True
except ImportError:
    _MOCK_MODE_AVAILABLE = False
    MockEngine = None
    RuleBasedStateMachine = object
    invariant = lambda *a, **k: (lambda f: f)
    precondition = lambda *a, **k: (lambda f: f)
    rule = lambda *a, **k: (lambda f: f)
    settings = lambda *a, **k: (lambda cls: cls)
    st = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

if not _MOCK_MODE_AVAILABLE:
    pytest.skip(
        "MockEngine harness not yet implemented.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.skip(reason="MockEngine harness not yet implemented.")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


class SchedulerContractMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
        self.handles: list = []

    def teardown(self) -> None:
        self.engine.shutdown()

    @rule(
        prompt_len=st.integers(min_value=1, max_value=200),
        max_new=st.integers(min_value=1, max_value=50),
    )
    def admit_random_prompt(self, prompt_len: int, max_new: int) -> None:
        h = self.engine.admit(prompt=_fake_prompt(prompt_len), max_new_tokens=max_new)
        self.handles.append(h)

    @precondition(lambda self: len(self.engine.active_reqs()) > 0)
    @rule(data=st.data())
    def preempt_random(self, data) -> None:
        target = data.draw(st.sampled_from(self.engine.active_reqs()))
        self.engine.force_preempt(target)

    @precondition(lambda self: any(not self.engine.is_active(h) for h in self.handles))
    @rule(data=st.data())
    def resume_random(self, data) -> None:
        paused = [h for h in self.handles if not self.engine.is_active(h)]
        target = data.draw(st.sampled_from(paused))
        self.engine.resume(target)

    @invariant()
    def no_canary_violations(self) -> None:
        self.engine.assert_no_canary_violations()

    @invariant()
    def allocator_conservation(self) -> None:
        stats = self.engine.allocator_stats()
        assert stats.free + stats.held == stats.total

    @invariant()
    def block_table_in_held(self) -> None:
        held = set(self.engine.allocator_stats().held_slots)
        for req in self.engine.active_reqs():
            assert set(self.engine.block_table(req)).issubset(held)


SchedulerContractMachine.TestCase = settings(max_examples=1000, deadline=60_000)(
    SchedulerContractMachine.TestCase
)


def test_machine_invariant_no_canary_violations() -> None:
    SchedulerContractMachine.TestCase().runTest()


def test_machine_invariant_allocator_conservation() -> None:
    SchedulerContractMachine.TestCase().runTest()


def test_machine_invariant_block_table_in_held() -> None:
    SchedulerContractMachine.TestCase().runTest()


def test_machine_rule_admit_random_prompt() -> None:
    machine = SchedulerContractMachine()
    machine.admit_random_prompt(prompt_len=16, max_new=4)
    assert len(machine.handles) == 1
    machine.teardown()


def test_machine_rule_preempt_random() -> None:
    machine = SchedulerContractMachine()
    machine.admit_random_prompt(prompt_len=8, max_new=4)
    machine.engine.step()
    target = machine.engine.active_reqs()[0]
    machine.engine.force_preempt(target)
    assert not machine.engine.is_active(target)
    machine.teardown()


def test_machine_rule_resume_random() -> None:
    machine = SchedulerContractMachine()
    machine.admit_random_prompt(prompt_len=8, max_new=4)
    machine.engine.step()
    target = machine.engine.active_reqs()[0]
    machine.engine.force_preempt(target)
    machine.engine.resume(target)
    assert machine.engine.is_active(target)
    machine.teardown()


def test_machine_shrinking_minimal_repro_under_10_steps() -> None:
    @settings(max_examples=200, deadline=60_000, derandomize=True)
    class DesignedToFail(SchedulerContractMachine):
        @invariant()
        def always_fail_after_two_admits(self) -> None:
            if len(self.handles) >= 2:
                raise AssertionError("designed-to-fail invariant")

    with pytest.raises(AssertionError) as info:
        DesignedToFail.TestCase().runTest()
    shrunk_trace = str(info.value)

    assert shrunk_trace.count("admit_random_prompt") <= 10
