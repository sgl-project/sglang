# SPDX-License-Identifier: Apache-2.0
"""Contracts for declared-parallel pipeline stage groups."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
    ComponentResidencyStrategy,
    ResidentStrategy,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
    ParallelExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)


class _FakeStage:
    """Stage double carrying only what the executor consumes."""

    _execution_group = None
    role_affinity = None
    parallelism_type = StageParallelismType.REPLICATED
    concurrency_safe = True
    may_use_collectives = False

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

    def component_uses(self, server_args, stage_name=None):
        return ()

    def __call__(self, payload, server_args):
        self.calls.append(time.perf_counter())
        if self._body is not None:
            return self._body(payload)
        return payload


class _ComponentStage:
    def __init__(self, name, component_name):
        self.name = name
        self.component_name = component_name

    def component_uses(self, _server_args, stage_name=None):
        return [ComponentUse(stage_name or self.name, self.component_name)]


class _CountingResidentStrategy(ResidentStrategy):
    def __init__(self):
        self.prepared = []
        self.finished = []

    def prepare_for_use(self, module, use, state):
        self.prepared.append((module, use, state))

    def finish_use(self, module, use, state):
        self.finished.append((module, use, state))


class _NoopComponentResidencyManager:
    def __init__(self):
        self.pipeline = SimpleNamespace(modules={})

    def supports_parallel_stage_group(self, _stages, _server_args):
        return True

    def parallel_stage_group(self):
        return _null_context()


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
        parallel_stage_execution="serial",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _executor():
    executor = ParallelExecutor.__new__(ParallelExecutor)
    executor.component_residency_manager = _NoopComponentResidencyManager()
    return executor


_EXECUTOR_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor"
)


def _run(
    executor,
    stages,
    payload,
    server_args,
    allow_concurrency=True,
    parallel_supported=True,
):
    with (
        patch.object(ParallelExecutor, "before_stage", lambda *a, **k: None),
        patch.object(
            ParallelExecutor,
            "_parallel_execution_supported",
            return_value=parallel_supported,
        ),
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


class TestStageCollectiveCapabilities(unittest.TestCase):
    def test_wan_vae_collective_capability_tracks_parallel_encode(self):
        vae = AutoencoderKLWan.__new__(AutoencoderKLWan)
        object.__setattr__(vae, "use_parallel_encode", True)
        stage = ImageVAEEncodingStage(vae)

        with patch(
            "sglang.multimodal_gen.runtime.models.vaes.wanvae.get_sp_world_size",
            return_value=2,
        ):
            self.assertTrue(stage.may_use_collectives)

        with patch(
            "sglang.multimodal_gen.runtime.models.vaes.wanvae.get_sp_world_size",
            return_value=1,
        ):
            self.assertFalse(stage.may_use_collectives)


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
        first.may_use_collectives = True
        payload = SimpleNamespace()

        result = _run(
            _executor(),
            [first, second],
            payload,
            _server_args(parallel_stage_execution="auto"),
        )

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

        _run(
            _executor(),
            [head, left, right, tail],
            payload,
            _server_args(parallel_stage_execution="auto"),
        )

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
                _server_args(parallel_stage_execution="auto"),
            )

    def test_unsafe_member_downgrades_to_serial(self):
        """A stage that has not opted in keeps the whole level serial."""
        order = []
        stages = [
            _FakeStage("a", body=lambda p: (order.append("a"), p)[1]),
            _FakeStage("b", body=lambda p: (order.append("b"), p)[1]),
        ]
        stages[1].concurrency_safe = False
        for token_stage in stages:
            token_stage.set_execution_group("g1")
        payload = SimpleNamespace()
        _run(
            _executor(),
            stages,
            payload,
            _server_args(parallel_stage_execution="auto"),
        )
        self.assertEqual(order, ["a", "b"])

    def test_multiple_collectives_use_ordered_epochs(self):
        """A TP collective cannot race another TP collective, but can overlap
        the non-collective VAE sibling in its first epoch."""
        barrier = threading.Barrier(2, timeout=10)
        completed = set()

        def first_collective(payload):
            barrier.wait()
            completed.add("first")
            return payload

        def non_collective(payload):
            barrier.wait()
            completed.add("vae")
            return payload

        def second_collective(payload):
            self.assertEqual(completed, {"first", "vae"})
            completed.add("second")
            return payload

        first = _FakeStage("first", first_collective)
        second = _FakeStage("second", second_collective)
        vae = _FakeStage("vae", non_collective)
        for stage in (first, second, vae):
            stage.set_execution_group("g1")
        first.may_use_collectives = True
        second.may_use_collectives = True

        _run(
            _executor(),
            [first, second, vae],
            SimpleNamespace(),
            _server_args(parallel_stage_execution="auto"),
        )

        self.assertEqual(completed, {"first", "vae", "second"})

    def test_collective_epochs_keep_declared_collective_order(self):
        first = _FakeStage("first")
        second = _FakeStage("second")
        vae = _FakeStage("vae")
        first.may_use_collectives = True
        second.may_use_collectives = True

        epochs = ParallelExecutor._parallel_execution_epochs([first, vae, second])

        self.assertEqual(
            [[stage._name for stage in epoch] for epoch in epochs],
            [["first", "vae"], ["second"]],
        )

    def test_serial_policy_downgrades_to_serial(self):
        order = []
        stages = [
            _FakeStage("a", body=lambda p: (order.append("a"), p)[1]),
            _FakeStage("b", body=lambda p: (order.append("b"), p)[1]),
        ]
        for stage in stages:
            stage.set_execution_group("g1")

        _run(
            _executor(),
            stages,
            SimpleNamespace(),
            _server_args(parallel_stage_execution="serial"),
        )

        self.assertEqual(order, ["a", "b"])

    def test_non_cuda_downgrades_to_serial(self):
        order = []
        stages = [
            _FakeStage("a", body=lambda p: (order.append("a"), p)[1]),
            _FakeStage("b", body=lambda p: (order.append("b"), p)[1]),
        ]
        for stage in stages:
            stage.set_execution_group("g1")

        _run(
            _executor(),
            stages,
            SimpleNamespace(),
            _server_args(parallel_stage_execution="auto"),
            parallel_supported=False,
        )

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
            _server_args(
                parallel_stage_execution="auto", text_encoder_cpu_offload=True
            ),
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
            _run(
                executor,
                [first, second],
                SimpleNamespace(),
                _server_args(parallel_stage_execution="auto"),
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
            _run(
                _executor(),
                [first, second],
                SimpleNamespace(),
                _server_args(parallel_stage_execution="auto"),
            )

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
            _run(
                _executor(),
                [failing, healthy],
                SimpleNamespace(),
                _server_args(parallel_stage_execution="auto"),
            )


class TestParallelComponentResidency(unittest.TestCase):
    def _manager(self):
        modules = {
            "first": torch.nn.Linear(1, 1),
            "second": torch.nn.Linear(1, 1),
        }
        pipeline = SimpleNamespace(
            modules=modules,
            _stage_name_mapping={},
            component_residency_strategies={},
        )
        manager = ComponentResidencyManager(
            pipeline, SimpleNamespace(enable_layerwise_nvtx_marker=False)
        )
        return manager, modules

    def test_parallel_group_prepares_resident_components_without_active_use(self):
        manager, modules = self._manager()
        strategy = _CountingResidentStrategy()
        manager.strategy_for = lambda _name, _module: strategy
        stages = [
            _ComponentStage("first", "first"),
            _ComponentStage("second", "second"),
        ]

        self.assertTrue(manager.supports_parallel_stage_group(stages, _server_args()))
        with manager.parallel_stage_group():
            for stage in stages:
                use = stage.component_uses(_server_args())[0]
                manager.begin_use(use, modules[use.component_name])
                manager.end_use(use, modules[use.component_name])
            self.assertIsNone(manager._active_use)

        self.assertEqual(
            [use.component_name for _, use, _ in strategy.prepared], ["first", "second"]
        )
        self.assertEqual(strategy.finished, [])
        self.assertEqual(manager._parallel_stage_group_depth, 0)

    def test_nonresident_component_rejects_parallel_group(self):
        manager, _ = self._manager()
        manager.strategy_for = lambda _name, _module: ComponentResidencyStrategy()

        self.assertFalse(
            manager.supports_parallel_stage_group(
                [_ComponentStage("first", "first")], _server_args()
            )
        )


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
