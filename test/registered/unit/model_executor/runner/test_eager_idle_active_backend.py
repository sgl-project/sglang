"""EagerRunner._execute_idle must init the active forward-context backend.

The EAGLE draft idle loop runs each step under a ForwardContext whose
attn_backend is the per-step decode backend, while the draft runner's
model_runner.attn_backend is the draft-extend backend. Layers read the active
context backend, so it has to get forward_metadata for the idle batch too.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_runner(runner_backend):
    seen = {}

    def _model_forward(input_ids, positions, forward_batch, **kwargs):
        # Record the backend the layers would read during the forward.
        seen["backend"] = get_attn_backend()
        return "out"

    model_runner = SimpleNamespace(
        attn_backend=runner_backend,
        device_timer=None,
        model=SimpleNamespace(forward=_model_forward),
        _pp_kwargs=lambda pp_proxy_tensors: {},
    )
    runner = EagerRunner.__new__(EagerRunner)
    runner.model_runner = model_runner
    runner.enable_pdmux = False
    runner.load_batch = lambda forward_batch, pp_proxy_tensors=None: forward_batch
    return runner, seen


def _idle_batch(batch_size):
    return SimpleNamespace(
        forward_mode=ForwardMode.IDLE,
        batch_size=batch_size,
        input_ids=None,
        positions=None,
        moe_num_token_non_padded=None,
    )


class TestEagerIdleActiveBackend(unittest.TestCase):
    def test_context_backend_differs_from_runner_backend(self):
        draft_extend_backend = MagicMock(name="draft_extend_backend")
        step_backend = MagicMock(name="step_backend")
        runner, seen = _make_runner(draft_extend_backend)
        batch = _idle_batch(batch_size=1)

        with forward_context(ForwardContext(attn_backend=step_backend)):
            out = runner._execute_idle(batch)

        self.assertEqual(out, "out")
        self.assertIs(seen["backend"], step_backend)
        draft_extend_backend.init_forward_metadata.assert_called_once_with(batch)
        step_backend.init_forward_metadata.assert_called_once_with(batch)

    def test_context_backend_is_runner_backend(self):
        backend = MagicMock(name="backend")
        runner, seen = _make_runner(backend)
        batch = _idle_batch(batch_size=1)

        with forward_context(ForwardContext(attn_backend=backend)):
            runner._execute_idle(batch)

        self.assertIs(seen["backend"], backend)
        backend.init_forward_metadata.assert_called_once_with(batch)

    def test_empty_idle_batch_does_not_init_metadata(self):
        draft_extend_backend = MagicMock(name="draft_extend_backend")
        step_backend = MagicMock(name="step_backend")
        runner, _ = _make_runner(draft_extend_backend)

        with forward_context(ForwardContext(attn_backend=step_backend)):
            runner._execute_idle(_idle_batch(batch_size=0))

        draft_extend_backend.init_forward_metadata.assert_not_called()
        step_backend.init_forward_metadata.assert_not_called()
        self.assertIsNone(draft_extend_backend.forward_metadata)


if __name__ == "__main__":
    unittest.main()
