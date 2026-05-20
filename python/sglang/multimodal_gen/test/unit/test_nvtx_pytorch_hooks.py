"""Unit tests for ``sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks``.

These tests cover the CPU-only surface: the ``maybe_nvtx_range`` helper,
``DiffusionNvtxHooks.register_hooks`` / ``remove_hooks`` lifecycle, and the
shape-collection helper. The actual ``nvtx.range_push`` / ``range_pop`` calls
require CUDA and are exercised end-to-end by Nsight-Systems profiling runs.
"""

import unittest
from unittest.mock import patch

import torch

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
        with patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push, patch.object(
            nvtx_pytorch_hooks.nvtx, "range_pop"
        ) as pop:
            with maybe_nvtx_range("never", enabled=False):
                pass
        push.assert_not_called()
        pop.assert_not_called()

    def test_enabled_calls_matched_push_pop(self) -> None:
        with patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push, patch.object(
            nvtx_pytorch_hooks.nvtx, "range_pop"
        ) as pop:
            with maybe_nvtx_range("stage_X", enabled=True):
                pass
        push.assert_called_once_with("stage_X")
        pop.assert_called_once_with()

    def test_enabled_pops_on_exception(self) -> None:
        with patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push, patch.object(
            nvtx_pytorch_hooks.nvtx, "range_pop"
        ) as pop:
            with self.assertRaises(RuntimeError):
                with maybe_nvtx_range("stage_X", enabled=True):
                    raise RuntimeError("boom")
        push.assert_called_once_with("stage_X")
        pop.assert_called_once_with()

    def test_marker_with_braces_does_not_raise(self) -> None:
        """Regression: torch.cuda.nvtx.range() str-formats its argument,
        which would raise on a marker containing a literal ``{``. The helper
        calls range_push directly to sidestep that."""
        with patch.object(nvtx_pytorch_hooks.nvtx, "range_push"), patch.object(
            nvtx_pytorch_hooks.nvtx, "range_pop"
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
        with patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push, patch.object(
            nvtx_pytorch_hooks.nvtx, "range_pop"
        ) as pop:
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
        with patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push, patch.object(
            nvtx_pytorch_hooks.nvtx, "range_pop"
        ) as pop:
            hooks._forward_pre_hook(dummy, (torch.zeros(2, 3),), {})
            hooks._forward_hook(dummy, (), None)
        push.assert_called_once()
        marker = push.call_args.args[0]
        self.assertIn("dummy", marker)
        self.assertIn("[2, 3]", marker)
        pop.assert_called_once_with()

    def test_default_enabled_is_false(self) -> None:
        """Default off so an unguarded forward (e.g. early warmup) cannot
        emit ranges; the caller must explicitly enable via set_enabled
        (typically through ``PipelineStage._apply_nvtx_gate``)."""
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
        with patch.object(nvtx_pytorch_hooks.nvtx, "range_push") as push, patch.object(
            nvtx_pytorch_hooks.nvtx, "range_pop"
        ) as pop:
            with self.assertRaises(RuntimeError):
                model(torch.zeros(2))
        # One push from the pre-hook, one pop from the post-hook fired via
        # always_call=True; they must match to keep the stack balanced.
        self.assertEqual(push.call_count, pop.call_count)
        self.assertEqual(push.call_count, 1)


class _StubStage:
    """Pipeline-stage stand-in that exercises the shared NVTX helpers.

    Importing the real ``PipelineStage`` would drag in unrelated runtime
    dependencies (server_args, residency manager); the copies below are
    deliberately byte-identical to the methods in ``stages/base.py``.
    """

    def __init__(self, modules, enable_flag: bool) -> None:
        self._nvtx_hooks = None
        self._nvtx_registered_ids: frozenset[int] = frozenset()
        self._nvtx_zero_warned = False
        self._current_use_nvtx = False
        self._modules = modules
        self.server_args = type(
            "Args", (), {"enable_layerwise_nvtx_marker": enable_flag}
        )()
        self.zero_warn_count = 0  # tracked by the stub for spam-test

    def nvtx_hookable_modules(self):
        return self._modules

    def _maybe_register_nvtx_hooks(self):
        if not self.server_args.enable_layerwise_nvtx_marker:
            return
        current = self.nvtx_hookable_modules()
        current_ids = frozenset(id(m) for m, _ in current if m is not None)
        if self._nvtx_hooks is not None:
            if current_ids == self._nvtx_registered_ids:
                return
            self._nvtx_hooks.remove_hooks()
            self._nvtx_hooks = None
            self._nvtx_registered_ids = frozenset()
            self._nvtx_zero_warned = False
        hooks = DiffusionNvtxHooks()
        total = 0
        for module, prefix in current:
            if module is None:
                continue
            total += hooks.register_hooks(module, prefix=prefix)
        if total == 0:
            if not self._nvtx_zero_warned:
                self.zero_warn_count += 1
                self._nvtx_zero_warned = True
            return
        self._nvtx_hooks = hooks
        self._nvtx_registered_ids = current_ids

    def _apply_nvtx_gate(self, is_warmup: bool) -> bool:
        self._maybe_register_nvtx_hooks()
        use_nvtx = self.server_args.enable_layerwise_nvtx_marker and not is_warmup
        if self._nvtx_hooks is not None:
            self._nvtx_hooks.set_enabled(use_nvtx)
        self._current_use_nvtx = use_nvtx
        return use_nvtx

    @property
    def current_use_nvtx(self) -> bool:
        return self._current_use_nvtx

    def _detach_nvtx_hooks(self):
        if self._nvtx_hooks is not None:
            self._nvtx_hooks.remove_hooks()
            self._nvtx_hooks = None
        self._nvtx_registered_ids = frozenset()
        self._nvtx_zero_warned = False


class TestStageMixin(unittest.TestCase):
    def test_disabled_flag_is_noop(self) -> None:
        stage = _StubStage([(torch.nn.Linear(2, 2), "m")], enable_flag=False)
        stage._maybe_register_nvtx_hooks()
        self.assertIsNone(stage._nvtx_hooks)

    def test_apply_nvtx_gate_registers_and_toggles(self) -> None:
        stage = _StubStage([(torch.nn.Linear(2, 2), "linear")], enable_flag=True)
        self.assertTrue(stage._apply_nvtx_gate(is_warmup=False))
        self.assertTrue(stage._nvtx_hooks._enabled)
        # Warmup flips the toggle but keeps hooks registered.
        self.assertFalse(stage._apply_nvtx_gate(is_warmup=True))
        self.assertFalse(stage._nvtx_hooks._enabled)

    def test_detach_clears_state_and_allows_reregister(self) -> None:
        stage = _StubStage([(torch.nn.Linear(2, 2), "linear")], enable_flag=True)
        stage._maybe_register_nvtx_hooks()
        stage._detach_nvtx_hooks()
        self.assertIsNone(stage._nvtx_hooks)
        # Idempotent + re-registration still works.
        stage._detach_nvtx_hooks()
        stage._maybe_register_nvtx_hooks()
        self.assertIsNotNone(stage._nvtx_hooks)

    def test_zero_modules_warning_fires_once(self) -> None:
        # A lazy-loaded stage may have zero modules for several requests;
        # the warning must not spam every call.
        stage = _StubStage(modules=[], enable_flag=True)
        for _ in range(5):
            stage._maybe_register_nvtx_hooks()
        self.assertEqual(stage.zero_warn_count, 1)

    def test_current_use_nvtx_reflects_last_gate(self) -> None:
        stage = _StubStage([(torch.nn.Linear(2, 2), "m")], enable_flag=True)
        self.assertFalse(stage.current_use_nvtx)  # before any gate
        stage._apply_nvtx_gate(is_warmup=False)
        self.assertTrue(stage.current_use_nvtx)
        stage._apply_nvtx_gate(is_warmup=True)
        self.assertFalse(stage.current_use_nvtx)

    def test_call_invokes_gate_before_forward(self) -> None:
        """Integration: ``PipelineStage.__call__`` must run
        ``_apply_nvtx_gate`` before ``forward``. Catches drift between
        the stub above and the real base class (which may grow new
        responsibilities the stub doesn't mirror)."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
            PipelineStage,
        )

        class _Spy(PipelineStage):
            def __init__(self, mod) -> None:
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
                self._nvtx_hooks = None
                self._nvtx_registered_ids = frozenset()
                self._nvtx_zero_warned = False
                self._current_use_nvtx = False
                self._mod = mod
                self.use_nvtx_during_forward: bool | None = None

            def nvtx_hookable_modules(self):
                return [(self._mod, "spy")]

            def forward(self, batch, server_args):
                self.use_nvtx_during_forward = self.current_use_nvtx
                return batch

        class _Batch:
            is_warmup = False
            metrics = None
            perf_dump_path = None

        spy = _Spy(torch.nn.Linear(2, 2))
        spy(_Batch(), spy.server_args)
        # __call__ must have set current_use_nvtx before forward executed.
        self.assertTrue(spy.use_nvtx_during_forward)
        self.assertIsNotNone(spy._nvtx_hooks)

    def test_re_registers_when_module_identity_changes(self) -> None:
        """Regression: when the underlying module reference is replaced
        between calls (lazy load, cache-dit wrap, hot swap), hooks must
        rebind to the new instance. The previous design's idempotency
        guard would have silently kept hooks on the orphan module."""
        a = torch.nn.Linear(2, 2)
        stage = _StubStage([(a, "t")], enable_flag=True)
        stage._maybe_register_nvtx_hooks()
        first_hooks = stage._nvtx_hooks
        self.assertIsNotNone(first_hooks)
        self.assertIn(a, first_hooks._module_to_name_map)

        # Replace the declared module with a different instance — same
        # role / same prefix, different identity (mimics cache-dit wrap).
        b = torch.nn.Linear(2, 2)
        stage._modules = [(b, "t")]
        stage._maybe_register_nvtx_hooks()

        # A fresh DiffusionNvtxHooks instance must have replaced the old
        # one and registered hooks on B; A must no longer be tracked.
        self.assertIsNot(stage._nvtx_hooks, first_hooks)
        self.assertIn(b, stage._nvtx_hooks._module_to_name_map)
        self.assertNotIn(a, stage._nvtx_hooks._module_to_name_map)


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
