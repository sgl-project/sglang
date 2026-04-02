import unittest
from types import ModuleType
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.utils import CompilableRegionMixin, MultiPlatformOp
from sglang.srt.utils.torch_compile_utils import (
    CompileConfig,
    parse_compile_op_config,
    resolve_compile_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _ForwardTrackerOp(MultiPlatformOp):
    def dispatch_forward(self):
        return self.forward_platform

    def forward_platform(self, *args, **kwargs):
        return "platform"

    def forward_native(self, *args, **kwargs):
        return "native"


class DummyLayer(_ForwardTrackerOp):
    pass


class TopK(_ForwardTrackerOp):
    pass


class DummyFusedMoE(_ForwardTrackerOp):
    pass


class DummyLayerSubclass(DummyLayer):
    """Subclass that inherits everything — simulates YaRNScalingRotaryEmbedding."""

    pass


class TestMultiPlatformTorchCompileOverride(CustomTestCase):
    def test_default_heuristic_switches_regular_layers(self):
        layer = DummyLayer()

        self.assertEqual(layer.forward(), "platform")

        layer.enter_torch_compile(num_tokens=4)

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_default_topk_heuristic_keeps_multi_token_path(self):
        layer = TopK()

        self.assertEqual(layer.forward(), "platform")

        layer.enter_torch_compile(num_tokens=4)

        self.assertEqual(layer.forward(), "platform")
        self.assertTrue(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_default_topk_heuristic_switches_single_token_path(self):
        layer = TopK()

        layer.enter_torch_compile(num_tokens=1)

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

    def test_default_fused_moe_heuristic_switches_single_token_path(self):
        layer = DummyFusedMoE()
        fake_fused_moe_native = ModuleType("sglang.srt.layers.moe.fused_moe_native")

        def fake_forward(*args, **kwargs):
            return "fused-moe-native"

        fake_fused_moe_native.fused_moe_forward_native = fake_forward

        with patch.dict(
            "sys.modules",
            {"sglang.srt.layers.moe.fused_moe_native": fake_fused_moe_native},
        ):
            layer.enter_torch_compile(num_tokens=1)

        self.assertIs(layer._forward_method, fake_forward)
        self.assertTrue(layer.is_torch_compile)

    def test_override_layers_switch_fused_moe_for_multi_token(self):
        layer = DummyFusedMoE()

        layer.enter_torch_compile(num_tokens=4, override_layers=["DummyFusedMoE"])

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

    def test_override_layers_switch_topk_for_multi_token(self):
        layer = TopK()

        layer.enter_torch_compile(num_tokens=4, override_layers=["TopK"])

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_override_layers_require_exact_class_name(self):
        layer = TopK()

        layer.enter_torch_compile(num_tokens=4, override_layers=["Top"])

        self.assertEqual(layer.forward(), "platform")
        self.assertFalse(layer.is_torch_compile)

    def test_override_layers_skip_non_listed_layers(self):
        layer = DummyLayer()

        layer.enter_torch_compile(num_tokens=4, override_layers=["TopK"])

        self.assertEqual(layer.forward(), "platform")
        self.assertFalse(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_override_layers_match_subclass_via_parent_name(self):
        layer = DummyLayerSubclass()

        layer.enter_torch_compile(num_tokens=4, override_layers=["DummyLayer"])

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_override_layers_match_subclass_via_own_name(self):
        layer = DummyLayerSubclass()

        layer.enter_torch_compile(num_tokens=4, override_layers=["DummyLayerSubclass"])

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

    def test_override_layers_unrelated_name_skips_subclass(self):
        layer = DummyLayerSubclass()

        layer.enter_torch_compile(num_tokens=4, override_layers=["TopK"])

        self.assertFalse(layer.is_torch_compile)


class _DummyRegionModule(nn.Module, CompilableRegionMixin):
    """Mock module with a single compilable region for testing."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))

    def get_compilable_regions(self):
        return {"TestRegion": "_region_fn"}

    def _region_fn(self, x):
        return x * self.weight


class TestCompilableRegionMixin(CustomTestCase):
    def test_not_compiled_initially(self):
        mod = _DummyRegionModule()
        self.assertFalse(mod.is_region_compiled("TestRegion"))

    def test_enter_leave_round_trip(self):
        mod = _DummyRegionModule()
        x = torch.ones(4)
        result_before = mod._region_fn(x)

        mod.enter_region_compile("TestRegion")
        self.assertTrue(mod.is_region_compiled("TestRegion"))
        # The compiled function is stored as an instance attribute
        self.assertIn("_region_fn", mod.__dict__)

        mod.leave_region_compile("TestRegion")
        self.assertFalse(mod.is_region_compiled("TestRegion"))
        # Instance attr removed — class method is visible again
        self.assertNotIn("_region_fn", mod.__dict__)
        result_after = mod._region_fn(x)
        torch.testing.assert_close(result_before, result_after)

    def test_enter_is_idempotent(self):
        mod = _DummyRegionModule()
        mod.enter_region_compile("TestRegion")
        first_compiled = mod.__dict__["_region_fn"]

        mod.enter_region_compile("TestRegion")
        self.assertIs(mod.__dict__["_region_fn"], first_compiled)

    def test_leave_without_enter_is_noop(self):
        mod = _DummyRegionModule()
        self.assertNotIn("_region_fn", mod.__dict__)
        mod.leave_region_compile("TestRegion")
        self.assertNotIn("_region_fn", mod.__dict__)
        self.assertFalse(mod.is_region_compiled("TestRegion"))

    def test_leave_unknown_region_is_noop(self):
        mod = _DummyRegionModule()
        mod.leave_region_compile("NoSuchRegion")
        self.assertFalse(mod.is_region_compiled("NoSuchRegion"))

    def test_get_compilable_regions_returns_mapping(self):
        mod = _DummyRegionModule()
        regions = mod.get_compilable_regions()
        self.assertEqual(regions, {"TestRegion": "_region_fn"})

    def test_default_mixin_returns_empty_regions(self):
        class _Bare(nn.Module, CompilableRegionMixin):
            pass

        mod = _Bare()
        self.assertEqual(mod.get_compilable_regions(), {})


class _OpWithCustomConfig(_ForwardTrackerOp):
    compile_config = CompileConfig(
        mode="max-autotune", options={"max_autotune_gemm": True}
    )


class _OpWithModeOnly(_ForwardTrackerOp):
    compile_config = CompileConfig(mode="max-autotune-no-cudagraphs")


class _SubOpInheritsConfig(_OpWithCustomConfig):
    """Subclass that inherits parent's compile_config."""

    pass


class TestCompileConfig(CustomTestCase):
    """Tests for per-op CompileConfig resolution chain."""

    def test_global_defaults_when_no_class_or_override(self):
        op = DummyLayer()
        cfg = resolve_compile_config(op)
        self.assertIsInstance(cfg.mode, str)
        self.assertIsInstance(cfg.options, dict)

    @patch.dict("os.environ", {"SGLANG_TORCH_COMPILE_MODE": "max-autotune-no-cudagraphs"})
    def test_global_mode_from_env(self):
        op = DummyLayer()
        cfg = resolve_compile_config(op)
        self.assertEqual(cfg.mode, "max-autotune-no-cudagraphs")

    @patch.dict("os.environ", {}, clear=True)
    def test_global_mode_fallback_to_default(self):
        op = DummyLayer()
        cfg = resolve_compile_config(op)
        self.assertEqual(cfg.mode, "default")

    def test_class_level_config_overrides_global(self):
        op = _OpWithCustomConfig()
        cfg = resolve_compile_config(op)
        self.assertEqual(cfg.mode, "max-autotune")
        self.assertEqual(cfg.options, {"max_autotune_gemm": True})

    def test_class_level_partial_override_inherits_global_options(self):
        op = _OpWithModeOnly()
        cfg = resolve_compile_config(op)
        self.assertEqual(cfg.mode, "max-autotune-no-cudagraphs")
        self.assertIsInstance(cfg.options, dict)
        self.assertNotEqual(cfg.options, {"max_autotune_gemm": True})

    def test_subclass_inherits_parent_compile_config(self):
        op = _SubOpInheritsConfig()
        cfg = resolve_compile_config(op)
        self.assertEqual(cfg.mode, "max-autotune")
        self.assertEqual(cfg.options, {"max_autotune_gemm": True})

    def test_server_override_beats_class_default(self):
        op = _OpWithCustomConfig()
        overrides = {
            "_OpWithCustomConfig": CompileConfig(
                mode="eager", options={"combo_kernels": True}
            )
        }
        cfg = resolve_compile_config(op, overrides)
        self.assertEqual(cfg.mode, "eager")
        self.assertEqual(cfg.options, {"combo_kernels": True})

    def test_server_override_partial_mode_keeps_class_options(self):
        op = _OpWithCustomConfig()
        overrides = {"_OpWithCustomConfig": CompileConfig(mode="eager")}
        cfg = resolve_compile_config(op, overrides)
        self.assertEqual(cfg.mode, "eager")
        self.assertEqual(cfg.options, {"max_autotune_gemm": True})

    def test_server_override_matches_via_mro(self):
        op = _SubOpInheritsConfig()
        overrides = {
            "_OpWithCustomConfig": CompileConfig(mode="eager"),
        }
        cfg = resolve_compile_config(op, overrides)
        self.assertEqual(cfg.mode, "eager")

    def test_server_override_own_name_preferred_over_parent(self):
        op = _SubOpInheritsConfig()
        overrides = {
            "_SubOpInheritsConfig": CompileConfig(mode="sub-mode"),
            "_OpWithCustomConfig": CompileConfig(mode="parent-mode"),
        }
        cfg = resolve_compile_config(op, overrides)
        self.assertEqual(cfg.mode, "sub-mode")

    def test_unrelated_override_does_not_apply(self):
        op = DummyLayer()
        overrides = {"_OpWithCustomConfig": CompileConfig(mode="nope")}
        cfg = resolve_compile_config(op)
        self.assertNotEqual(cfg.mode, "nope")

    def test_parse_compile_op_config_none(self):
        self.assertIsNone(parse_compile_op_config(None))
        self.assertIsNone(parse_compile_op_config(""))

    def test_parse_compile_op_config_full(self):
        raw = '{"RMSNorm": {"mode": "max-autotune", "options": {"combo_kernels": true}}}'
        result = parse_compile_op_config(raw)
        self.assertIn("RMSNorm", result)
        self.assertEqual(result["RMSNorm"].mode, "max-autotune")
        self.assertEqual(result["RMSNorm"].options, {"combo_kernels": True})

    def test_parse_compile_op_config_mode_only(self):
        raw = '{"TopK": {"mode": "max-autotune-no-cudagraphs"}}'
        result = parse_compile_op_config(raw)
        self.assertEqual(result["TopK"].mode, "max-autotune-no-cudagraphs")
        self.assertIsNone(result["TopK"].options)


if __name__ == "__main__":
    unittest.main(verbosity=3)
