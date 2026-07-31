# SPDX-License-Identifier: Apache-2.0
"""Contracts for declared-parallel pipeline stage groups."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
    ParallelExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)


class _FakeStage:
    """Stage double carrying only what the executor consumes."""

    _execution_group = None
    role_affinity = None
    parallelism_type = StageParallelismType.REPLICATED
    concurrency_safe = True

    def __init__(self, name, body=None):
        self._name = name
        self._body = body
        self.calls = []

    def set_execution_group(self, token):
        self._execution_group = token

    def set_registered_stage_name(self, stage_name):
        self._registered_stage_name = stage_name

    def set_profile_stage_name(self, stage_name):
        self._profile_stage_name = stage_name

    def set_component_residency_manager(self, manager):
        pass

    def _component_stage_name(self, stage_name=None):
        return self._name

    def _active_component_stage_name(self):
        return self._name

    def component_uses(self, server_args, stage_name):
        return ()

    def __call__(self, payload, server_args):
        self.calls.append(time.perf_counter())
        if self._body is not None:
            return self._body(payload)
        return payload


def _server_args(**overrides):
    defaults = dict(
        enable_layerwise_nvtx_marker=False,
        enable_cfg_parallel=False,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        layerwise_offload_components=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _executor():
    executor = ParallelExecutor.__new__(ParallelExecutor)
    executor.component_residency_manager = None
    return executor


_EXECUTOR_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor"
)


def _run(executor, stages, payload, server_args, allow_concurrency=True):
    with (
        patch.object(ParallelExecutor, "before_stage", lambda *a, **k: None),
        patch.object(
            ParallelExecutor,
            "_component_residency_request",
            lambda self, *a, **k: _null_context(),
        ),
        patch.object(ParallelExecutor, "_is_warmup_payload", lambda self, p: True),
        patch(f"{_EXECUTOR_MODULE}.get_world_rank", return_value=0),
        patch(f"{_EXECUTOR_MODULE}.get_cfg_group", return_value=SimpleNamespace()),
        patch(f"{_EXECUTOR_MODULE}.get_world_group", return_value=SimpleNamespace()),
        patch(f"{_EXECUTOR_MODULE}.Req", SimpleNamespace),
    ):
        return executor._execute_stages(
            stages,
            payload,
            server_args,
            lambda stage, current: stage(current, server_args),
            allow_concurrency=allow_concurrency,
        )


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestExecutionLevelDerivation(unittest.TestCase):
    def test_consecutive_group_tokens_form_one_level(self):
        a, b, c, d = (_FakeStage(n) for n in "abcd")
        b.set_execution_group("g1")
        c.set_execution_group("g1")
        levels = PipelineExecutor.group_stages_into_execution_levels([a, b, c, d])
        self.assertEqual(
            [[s._name for s in level] for level in levels],
            [["a"], ["b", "c"], ["d"]],
        )

    def test_distinct_tokens_do_not_merge(self):
        a, b = _FakeStage("a"), _FakeStage("b")
        a.set_execution_group("g1")
        b.set_execution_group("g2")
        levels = PipelineExecutor.group_stages_into_execution_levels([a, b])
        self.assertEqual(len(levels), 2)


class TestParallelLevelExecution(unittest.TestCase):
    def test_group_members_truly_overlap(self):
        """Both members must be in flight at once: each waits on a barrier
        the other releases, which deadlocks under serial execution."""
        barrier = threading.Barrier(2, timeout=10)

        def body(payload):
            barrier.wait()
            return payload

        first, second = _FakeStage("first", body), _FakeStage("second", body)
        first.set_execution_group("g")
        second.set_execution_group("g")
        payload = SimpleNamespace()

        result = _run(_executor(), [first, second], payload, _server_args())

        self.assertIs(result, payload)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)

    def test_level_boundaries_stay_ordered(self):
        order = []

        def recorder(name):
            def body(payload):
                order.append(name)
                return payload

            return body

        head = _FakeStage("head", recorder("head"))
        left = _FakeStage("left", recorder("left"))
        right = _FakeStage("right", recorder("right"))
        tail = _FakeStage("tail", recorder("tail"))
        left.set_execution_group("g")
        right.set_execution_group("g")
        payload = SimpleNamespace()

        _run(_executor(), [head, left, right, tail], payload, _server_args())

        self.assertEqual(order[0], "head")
        self.assertEqual(order[-1], "tail")
        self.assertEqual(set(order[1:3]), {"left", "right"})

    def test_member_must_return_its_payload(self):
        replacing = _FakeStage("replacing", lambda payload: SimpleNamespace())
        passthrough = _FakeStage("passthrough")
        replacing.set_execution_group("g")
        passthrough.set_execution_group("g")

        with self.assertRaisesRegex(RuntimeError, "must[\\s\\S]*return the payload"):
            _run(
                _executor(),
                [replacing, passthrough],
                SimpleNamespace(),
                _server_args(),
            )

    def test_communicating_member_downgrades_to_serial(self):
        """A member that has not declared itself communication-free keeps
        the whole level in declaration order."""
        order = []
        stages = [
            _FakeStage("a", body=lambda p: (order.append("a"), p)[1]),
            _FakeStage("b", body=lambda p: (order.append("b"), p)[1]),
        ]
        stages[1].concurrency_safe = False
        for token_stage in stages:
            token_stage.set_execution_group("g1")
        payload = SimpleNamespace()
        _run(_executor(), stages, payload, _server_args())
        self.assertEqual(order, ["a", "b"])

    def test_offload_flags_downgrade_to_serial(self):
        """With offload enabled the same barrier pair must deadlock-timeout
        under serial execution — proving the downgrade — so the bodies here
        run without a barrier and we assert strict ordering instead."""
        order = []

        def recorder(name):
            def body(payload):
                order.append((name, "start"))
                time.sleep(0.05)
                order.append((name, "end"))
                return payload

            return body

        first = _FakeStage("first", recorder("first"))
        second = _FakeStage("second", recorder("second"))
        first.set_execution_group("g")
        second.set_execution_group("g")

        _run(
            _executor(),
            [first, second],
            SimpleNamespace(),
            _server_args(text_encoder_cpu_offload=True),
        )

        self.assertEqual(
            order,
            [
                ("first", "start"),
                ("first", "end"),
                ("second", "start"),
                ("second", "end"),
            ],
        )

    def test_runtime_layerwise_offload_downgrades_to_serial(self):
        """No offload flag is set, but a pipeline module is layerwise-offloaded
        at runtime (memory-aware loader / compile-scoped offload); the level
        must fall back to strict serial ordering."""
        order = []

        def recorder(name):
            def body(payload):
                order.append((name, "start"))
                time.sleep(0.05)
                order.append((name, "end"))
                return payload

            return body

        first = _FakeStage("first", recorder("first"))
        second = _FakeStage("second", recorder("second"))
        first.set_execution_group("g")
        second.set_execution_group("g")

        executor = _executor()
        executor.component_residency_manager = SimpleNamespace(
            pipeline=SimpleNamespace(modules={"transformer": torch.nn.Linear(1, 1)})
        )

        with patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers"
            ".layerwise_offload.is_layerwise_offloaded_module",
            return_value=True,
        ):
            _run(executor, [first, second], SimpleNamespace(), _server_args())

        self.assertEqual(
            order,
            [
                ("first", "start"),
                ("first", "end"),
                ("second", "start"),
                ("second", "end"),
            ],
        )

    def test_pool_members_inherit_caller_grad_mode(self):
        """grad mode is thread-local; pool-thread members must mirror the
        caller's inference_mode/no_grad or their forwards retain autograd
        graphs (observed as a >100GB live-memory leak on a real pipeline)."""
        states = {}
        barrier = threading.Barrier(2, timeout=5)

        def probe(name):
            def body(payload):
                barrier.wait()
                states[name] = (
                    torch.is_inference_mode_enabled(),
                    torch.is_grad_enabled(),
                )
                return payload

            return body

        first = _FakeStage("first", probe("first"))
        second = _FakeStage("second", probe("second"))
        first.set_execution_group("g")
        second.set_execution_group("g")

        with torch.inference_mode():
            _run(_executor(), [first, second], SimpleNamespace(), _server_args())

        self.assertEqual(states["first"], (True, False))
        self.assertEqual(states["second"], (True, False))

    def test_member_error_propagates_after_join(self):
        def boom(payload):
            raise ValueError("member failed")

        failing = _FakeStage("failing", boom)
        healthy = _FakeStage("healthy")
        failing.set_execution_group("g")
        healthy.set_execution_group("g")

        with self.assertRaisesRegex(ValueError, "member failed"):
            _run(_executor(), [failing, healthy], SimpleNamespace(), _server_args())


class TestAddParallelStagesDeclaration(unittest.TestCase):
    def test_members_share_one_token_and_stay_flat(self):
        from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
            ComposedPipelineBase,
        )

        class _DeclOnlyPipeline(ComposedPipelineBase):
            def create_pipeline_stages(self, server_args):
                pass

        pipeline = _DeclOnlyPipeline.__new__(_DeclOnlyPipeline)
        pipeline.modules = {}
        pipeline._stages = []
        pipeline._stage_name_mapping = {}
        with patch.object(
            ComposedPipelineBase, "_should_add_stage_for_role", return_value=True
        ):
            pipeline.add_stage(_FakeStage("head"), "head")
            pipeline.add_parallel_stages(
                [(_FakeStage("left"), "left"), (_FakeStage("right"), "right")]
            )
            pipeline.add_stage(_FakeStage("tail"), "tail")

        tokens = [stage._execution_group for stage in pipeline._stages]
        self.assertIsNone(tokens[0])
        self.assertIsNone(tokens[3])
        self.assertIsNotNone(tokens[1])
        self.assertEqual(tokens[1], tokens[2])


if __name__ == "__main__":
    unittest.main()
