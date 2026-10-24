"""Native-style deferred resolution contract; actual image smoke is separate."""

from types import SimpleNamespace

import pytest
from sglang_simulator.compat import (
    SGLangCompatibilityError,
    resolve_simulator_server_args,
)


class DeferredArgs:
    """Preserve native raw-versus-declared lifecycle without device imports."""

    def __init__(self):
        self.speculative_num_steps = None
        self.speculative_eagle_topk = None
        self._resolved_overrides = []
        self.resolutions = 0
        self._resolution_finished = False

    def resolve_once(self):
        if self._resolution_finished:
            return
        self.resolutions += 1
        self._resolved_overrides.append(
            (
                "speculative_handler",
                {"speculative_num_steps": 1, "speculative_eagle_topk": 1},
            )
        )
        self._resolution_finished = True


def test_explicit_resolution_runs_handler_and_preserves_raw_inputs():
    args = DeferredArgs()
    assert not args._resolution_finished and args._resolved_overrides == []
    resolve_simulator_server_args(args)
    assert args._resolution_finished and args.resolutions == 1
    assert args.speculative_num_steps is None and args.speculative_eagle_topk is None
    assert args._resolved_overrides[-1][1] == {
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
    }
    resolve_simulator_server_args(args)
    assert args.resolutions == 1


def test_native_resolution_failure_is_not_retried_or_replaced():
    failure = ValueError("native model/argument incompatibility")
    calls = []

    def resolve():
        calls.append(1)
        raise failure

    with pytest.raises(ValueError) as error:
        resolve_simulator_server_args(SimpleNamespace(resolve_once=resolve))
    assert error.value is failure and calls == [1]


def test_native_resolution_api_is_required_instead_of_assuming_constructor_resolved():
    with pytest.raises(SGLangCompatibilityError, match="resolve_once"):
        resolve_simulator_server_args(SimpleNamespace())
