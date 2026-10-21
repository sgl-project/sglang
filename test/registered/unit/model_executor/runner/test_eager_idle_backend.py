"""Idle metadata must belong to the backend consumed by the model forward."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
    get_forward_context,
    has_forward_context,
    set_forward_context,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Backend:
    def __init__(self):
        self.forward_metadata = object()

    def init_forward_metadata(self, batch):
        self.forward_metadata = batch


class TestEagerIdleBackend(CustomTestCase):
    def setUp(self):
        super().setUp()
        previous = set_forward_context(None)
        self.addCleanup(set_forward_context, previous)

    def _runner(self, default, *, pdmux=False, loaded_batch=None):
        runner = object.__new__(EagerRunner)
        runner.enable_pdmux = pdmux
        runner.model_context = nullcontext
        runner.load_batch = Mock(side_effect=lambda batch, _: loaded_batch or batch)

        def forward(input_ids, positions, batch, **kwargs):
            backend = get_attn_backend() if has_forward_context() else default
            return SimpleNamespace(
                metadata=backend.forward_metadata,
                batch=batch,
                input_ids=input_ids,
                positions=positions,
                kwargs=kwargs,
                moe_bound_getter=get_forward_context().moe_num_token_non_padded,
            )

        runner.model_runner = SimpleNamespace(
            attn_backend=default,
            device_timer=None,
            model=SimpleNamespace(forward=forward),
            _pp_kwargs=lambda tensors: {"pp_proxy_tensors": tensors},
        )
        return runner

    @staticmethod
    def _batch(size):
        return SimpleNamespace(
            batch_size=size,
            forward_mode=ForwardMode.IDLE,
            input_ids=object(),
            positions=object(),
            moe_num_token_non_padded=lambda: None,
        )

    def test_padded_idle_initializes_the_active_draft_child(self):
        default = _Backend()
        default_metadata = default.forward_metadata
        runner = self._runner(default)
        children = [_Backend() for _ in range(4)]
        for child in children:
            for size in (1, 8, 2):
                with self.subTest(child=children.index(child), batch_size=size):
                    batch = self._batch(size)
                    with forward_context(ForwardContext(attn_backend=child)):
                        result = runner.execute(batch)
                    self.assertIs(result.metadata, batch)
                    self.assertIs(default.forward_metadata, default_metadata)

    def test_unpadded_idle_clears_only_the_active_child(self):
        default, active, sibling = _Backend(), _Backend(), _Backend()
        default_metadata, sibling_metadata = (
            default.forward_metadata,
            sibling.forward_metadata,
        )
        runner = self._runner(default)
        with forward_context(ForwardContext(attn_backend=active)):
            result = runner.execute(self._batch(0))
        self.assertIsNone(result.metadata)
        self.assertIs(default.forward_metadata, default_metadata)
        self.assertIs(sibling.forward_metadata, sibling_metadata)
        runner.load_batch.assert_not_called()

    def test_metadata_uses_the_loaded_batch_and_preserves_pp_inputs(self):
        default, active = _Backend(), _Backend()
        batch, loaded = self._batch(1), self._batch(4)
        runner = self._runner(default, loaded_batch=loaded)
        pp_inputs = object()
        with forward_context(ForwardContext(attn_backend=active)):
            result = runner.execute(batch, pp_proxy_tensors=pp_inputs)
        self.assertIs(result.metadata, loaded)
        self.assertIs(result.batch, loaded)
        self.assertIs(result.input_ids, loaded.input_ids)
        self.assertIs(result.positions, loaded.positions)
        self.assertIs(result.kwargs["pp_proxy_tensors"], pp_inputs)
        self.assertIs(result.moe_bound_getter, loaded.moe_num_token_non_padded)

    def test_pdmux_idle_keeps_the_active_backend_without_loading_a_batch(self):
        default, active = _Backend(), _Backend()
        runner = self._runner(default, pdmux=True)
        batch = self._batch(3)
        with forward_context(ForwardContext(attn_backend=active)):
            result = runner.execute(batch)
        self.assertIs(result.metadata, batch)
        runner.load_batch.assert_not_called()

    def test_default_context_and_direct_call_retain_default_backend(self):
        for direct in (False, True):
            for size in (0, 3):
                with self.subTest(direct=direct, batch_size=size):
                    default = _Backend()
                    runner = self._runner(default)
                    batch = self._batch(size)
                    with forward_context(ForwardContext(attn_backend=default)):
                        execute = runner._execute_idle if direct else runner.execute
                        result = execute(batch)
                    self.assertIs(result.metadata, batch if size else None)
                    self.assertFalse(has_forward_context())

    def test_switching_children_does_not_clear_another_steps_metadata(self):
        default, first, second = _Backend(), _Backend(), _Backend()
        runner = self._runner(default)
        batch = self._batch(2)
        with forward_context(ForwardContext(attn_backend=first)):
            runner.execute(batch)
            with forward_context(ForwardContext(attn_backend=second)):
                runner.execute(self._batch(0))
            self.assertIs(get_attn_backend(), first)
            self.assertIs(first.forward_metadata, batch)
        self.assertIsNone(second.forward_metadata)
        self.assertFalse(has_forward_context())


if __name__ == "__main__":
    unittest.main()
