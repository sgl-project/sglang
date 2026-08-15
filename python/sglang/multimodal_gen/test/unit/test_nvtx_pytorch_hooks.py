"""Unit tests for ``sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks``.

These tests cover the CPU-only surface: the ``maybe_nvtx_range`` helper,
``DiffusionNvtxHooks.register_hooks`` / ``remove_hooks`` lifecycle, and the
shape-collection helper. The actual ``nvtx.range_push`` / ``range_pop`` calls
require CUDA and are exercised end-to-end by Nsight-Systems profiling runs.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
    ComponentUse,
)
from sglang.multimodal_gen.runtime.utils import nvtx_pytorch_hooks
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import (
    DiffusionNvtxHooks,
    _collect_input_shapes,
    maybe_nvtx_range,
)


class TestMaybeNvtxRange(unittest.TestCase):
    def test_disabled_returns_noop_context_manager(self) -> None:
        ran = False
        with maybe_nvtx_range("never", enabled=False):
            ran = True
        self.assertTrue(ran)

    def test_disabled_propagates_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with maybe_nvtx_range("never", enabled=False):
                raise RuntimeError("boom")

    def test_disabled_does_not_call_nvtx(self) -> None:
        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as pop,
        ):
            with maybe_nvtx_range("never", enabled=False):
                pass
        push.assert_not_called()
        pop.assert_not_called()

    def test_enabled_calls_matched_push_pop(self) -> None:
        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as pop,
        ):
            with maybe_nvtx_range("stage_X", enabled=True):
                pass
        push.assert_called_once_with("stage_X")
        pop.assert_called_once_with()

    def test_enabled_pops_on_exception(self) -> None:
        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as pop,
        ):
            with self.assertRaises(RuntimeError):
                with maybe_nvtx_range("stage_X", enabled=True):
                    raise RuntimeError("boom")
        push.assert_called_once_with("stage_X")
        pop.assert_called_once_with()

    def test_marker_with_braces_does_not_raise(self) -> None:
        """Regression: torch.cuda.nvtx.range() str-formats its argument,
        which would raise on a marker containing a literal ``{``. The helper
        calls range_push directly to sidestep that."""
        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push"),
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop"),
        ):
            with maybe_nvtx_range("layer in={1, 2, 3}", enabled=True):
                pass


class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        # Dropout is in _DEFAULT_SKIP_TYPES and must not be instrumented.
        self.drop = torch.nn.Dropout(p=0.0)


class TestDiffusionNvtxHooks(unittest.TestCase):
    def test_register_hooks_counts_non_skipped_submodules(self) -> None:
        block = _TinyBlock()
        hooks = DiffusionNvtxHooks()
        # 4 modules total (block, linear, norm, drop); drop is skipped.
        self.assertEqual(hooks.register_hooks(block, prefix="block"), 3)
        # 2 hooks (pre + post) registered per instrumented module.
        self.assertEqual(len(hooks._hook_handles), 6)

    def test_register_hooks_skips_duplicate_instances(self) -> None:
        shared = torch.nn.Linear(4, 4)

        class TiedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = shared
                self.b = shared

        model = TiedModel()
        hooks = DiffusionNvtxHooks()
        # Root + 1 unique linear (the second occurrence is skipped).
        self.assertEqual(hooks.register_hooks(model), 2)

    def test_remove_hooks_is_idempotent(self) -> None:
        block = _TinyBlock()
        hooks = DiffusionNvtxHooks()
        hooks.register_hooks(block)
        hooks.remove_hooks()
        self.assertEqual(hooks._hook_handles, [])
        self.assertEqual(hooks._module_to_name_map, {})
        # Second call is a no-op, not an error.
        hooks.remove_hooks()

    def test_set_enabled_false_suppresses_nvtx_calls(self) -> None:
        """When disabled, neither pre- nor post-hook should call nvtx —
        guarantees no half-open push without a matching pop."""
        hooks = DiffusionNvtxHooks()
        dummy = torch.nn.Linear(2, 2)
        hooks._module_to_name_map[dummy] = "dummy"
        hooks.set_enabled(False)
        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as pop,
        ):
            hooks._forward_pre_hook(dummy, (torch.zeros(2),), {})
            hooks._forward_hook(dummy, (), None)
        push.assert_not_called()
        pop.assert_not_called()

    def test_set_enabled_true_emits_matched_push_pop(self) -> None:
        """When enabled, a forward pre/post pair emits exactly one push
        and one pop with the qualified module name as the marker."""
        hooks = DiffusionNvtxHooks()
        dummy = torch.nn.Linear(2, 2)
        hooks._module_to_name_map[dummy] = "dummy"
        hooks.set_enabled(True)
        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as pop,
        ):
            hooks._forward_pre_hook(dummy, (torch.zeros(2, 3),), {})
            hooks._forward_hook(dummy, (), None)
        push.assert_called_once()
        marker = push.call_args.args[0]
        self.assertIn("dummy", marker)
        self.assertIn("[2, 3]", marker)
        pop.assert_called_once_with()

    def test_default_enabled_is_false(self) -> None:
        """Default off so an unguarded forward (e.g. early warmup) cannot
        emit ranges; the caller must explicitly enable via set_enabled."""
        self.assertFalse(DiffusionNvtxHooks()._enabled)

    def test_post_hook_fires_on_forward_exception(self) -> None:
        """Regression: ``always_call=True`` on the registered post-hook
        guarantees ``range_pop`` runs even when the wrapped ``forward``
        raises. Without it an OOM (or any other forward-time exception)
        would leak a half-open NVTX range."""

        class _RaisingModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("simulated forward exception")

        model = _RaisingModule()
        hooks = DiffusionNvtxHooks()
        hooks.register_hooks(model, prefix="raising")
        hooks.set_enabled(True)
        with (
            patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push,
            patch.object(nvtx_pytorch_hooks.nvtx, "range_pop") as pop,
        ):
            with self.assertRaises(RuntimeError):
                model(torch.zeros(2))
        # One push from the pre-hook, one pop from the post-hook fired via
        # always_call=True; they must match to keep the stack balanced.
        self.assertEqual(push.call_count, pop.call_count)
        self.assertEqual(push.call_count, 1)


class _NoOpResidencyStrategy:
    name = "noop"

    def prepare_for_use(self, module, use, state) -> None:
        pass

    def wait_for_use(self, module, use, state) -> None:
        pass

    def finish_use(self, module, use, state) -> None:
        pass

    def finish_request(self, module, use, state, *, preferred: bool) -> None:
        pass

    def prefetch_for_use(self, module, use, state) -> bool:
        return False

    def prepare_after_request(self, module, use, state) -> None:
        pass


def _test_manager(
    modules: dict[str, torch.nn.Module],
    *,
    enable_flag: bool = True,
    is_warmup: bool = False,
) -> ComponentResidencyManager:
    pipeline = SimpleNamespace(
        modules=modules,
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    server_args = SimpleNamespace(enable_layerwise_nvtx_marker=enable_flag)
    manager = ComponentResidencyManager(pipeline, server_args)
    manager.state.batch_is_warmup = is_warmup
    manager.strategy_for = lambda _component_name, _module: _NoOpResidencyStrategy()
    return manager


class TestComponentResidencyNvtxHooks(unittest.TestCase):
    def test_disabled_flag_is_noop(self) -> None:
        module = torch.nn.Linear(2, 2)
        manager = _test_manager({"linear": module}, enable_flag=False)
        manager.begin_use(ComponentUse("Stage", "linear"), module=module)
        self.assertEqual(manager._nvtx_hooks_by_use_key, {})

    def test_warmup_is_noop(self) -> None:
        module = torch.nn.Linear(2, 2)
        manager = _test_manager({"linear": module}, is_warmup=True)
        manager.begin_use(ComponentUse("Stage", "linear"), module=module)
        self.assertEqual(manager._nvtx_hooks_by_use_key, {})

    def test_begin_use_registers_and_enables_component_hooks(self) -> None:
        module = torch.nn.Linear(2, 2)
        manager = _test_manager({"linear": module})
        use = ComponentUse("Stage", "linear")

        manager.begin_use(use, module=module)

        _, hooks = manager._nvtx_hooks_by_use_key[("Stage", "linear", None)]
        self.assertTrue(hooks._enabled)
        self.assertIn(module, hooks._module_to_name_map)
        self.assertTrue(hooks._module_to_name_map[module].startswith("Stage.linear"))

    def test_end_use_disables_component_hooks(self) -> None:
        module = torch.nn.Linear(2, 2)
        manager = _test_manager({"linear": module})
        use = ComponentUse("Stage", "linear")

        manager.begin_use(use, module=module)
        _, hooks = manager._nvtx_hooks_by_use_key[("Stage", "linear", None)]
        manager.end_use(use, module=module)

        self.assertFalse(hooks._enabled)
        self.assertIsNone(manager._active_nvtx_key)

    def test_remove_nvtx_hooks_for_module_drops_stale_reference(self) -> None:
        module = torch.nn.Linear(2, 2)
        manager = _test_manager({"linear": module})
        use = ComponentUse("Stage", "linear")

        manager.begin_use(use, module=module)
        _, hooks = manager._nvtx_hooks_by_use_key[("Stage", "linear", None)]
        manager.remove_nvtx_hooks_for_module(module)

        self.assertEqual(manager._nvtx_hooks_by_use_key, {})
        self.assertEqual(hooks._module_to_name_map, {})
        self.assertIsNone(manager._active_nvtx_key)

    def test_re_registers_when_module_identity_changes(self) -> None:
        use = ComponentUse("Stage", "linear")
        first_module = torch.nn.Linear(2, 2)
        manager = _test_manager({"linear": first_module})

        manager.begin_use(use, module=first_module)
        _, first_hooks = manager._nvtx_hooks_by_use_key[("Stage", "linear", None)]
        manager.end_use(use, module=first_module)

        second_module = torch.nn.Linear(2, 2)
        manager.pipeline.modules["linear"] = second_module
        manager.begin_use(use, module=second_module)

        _, second_hooks = manager._nvtx_hooks_by_use_key[("Stage", "linear", None)]
        self.assertIsNot(second_hooks, first_hooks)
        self.assertEqual(first_hooks._module_to_name_map, {})
        self.assertIn(second_module, second_hooks._module_to_name_map)

    def test_same_component_in_different_stages_switches_active_prefix(self) -> None:
        shared = torch.nn.Linear(2, 2)
        manager = _test_manager({"vae": shared})
        first_use = ComponentUse("ImageVAEEncodingStage", "vae")
        second_use = ComponentUse("DecodingStage", "vae")

        manager.begin_use(first_use, module=shared)
        _, first_hooks = manager._nvtx_hooks_by_use_key[
            ("ImageVAEEncodingStage", "vae", None)
        ]
        manager.begin_use(second_use, module=shared)
        _, second_hooks = manager._nvtx_hooks_by_use_key[("DecodingStage", "vae", None)]

        self.assertFalse(first_hooks._enabled)
        self.assertTrue(second_hooks._enabled)
        self.assertTrue(
            second_hooks._module_to_name_map[shared].startswith("DecodingStage.vae")
        )

    def test_pipeline_stage_call_sets_explicit_range_gate_before_forward(self) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
            PipelineStage,
        )

        class _Spy(PipelineStage):
            def __init__(self) -> None:
                self.server_args = type(
                    "Args",
                    (),
                    {
                        "enable_layerwise_nvtx_marker": True,
                        "comfyui_mode": False,
                    },
                )()
                self._component_residency_manager = None
                self._registered_stage_name = None
                self._profile_stage_name = None
                self._current_use_nvtx = False
                self.use_nvtx_during_forward: bool | None = None

            def forward(self, batch, server_args):
                self.use_nvtx_during_forward = self.current_use_nvtx
                return batch

        class _Batch:
            is_warmup = False
            metrics = None
            perf_dump_path = None

        spy = _Spy()
        spy(_Batch(), spy.server_args)
        self.assertTrue(spy.use_nvtx_during_forward)
        self.assertFalse(spy.current_use_nvtx)


class TestCollectInputShapes(unittest.TestCase):
    def test_flat_positional_tensors(self) -> None:
        a = torch.zeros(2, 3)
        b = torch.zeros(4)
        self.assertEqual(_collect_input_shapes((a, b)), [[2, 3], [4]])

    def test_kwarg_tensors_are_captured(self) -> None:
        kw = {"hidden_states": torch.zeros(1, 4)}
        self.assertEqual(_collect_input_shapes((), kw), [[1, 4]])

    def test_nested_tuple_kwarg_recurses(self) -> None:
        rope = (torch.zeros(8, 16), torch.zeros(8, 16))
        kw = {"image_rotary_emb": rope}
        self.assertEqual(_collect_input_shapes((), kw), [[8, 16], [8, 16]])

    def test_non_tensor_values_are_skipped(self) -> None:
        kw = {"scale": 1.0, "use_cache": True, "extras": None}
        self.assertEqual(_collect_input_shapes((42, "s"), kw), [])


if __name__ == "__main__":
    unittest.main()
