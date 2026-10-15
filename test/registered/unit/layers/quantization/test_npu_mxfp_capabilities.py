"""CPU preflight contracts; deliberately NOT Ascend kernel/e2e validation."""

import ast
import re
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.kernels.spec import KernelBackend, PlatformInfo
from sglang.srt.hardware_backend.npu.quantization import linear_method_npu
from sglang.srt.hardware_backend.npu.quantization.mxfp_capabilities import (
    MXFP_CAPABILITIES,
    W4A4,
    W4A4_DUAL,
    W4A8,
    W8A8,
    get_mxfp_capability,
    render_mxfp_support_table,
    require_mxfp_operators,
    validate_mxfp_configuration,
    validate_mxfp_loaded_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNPUMXFPCapabilities(unittest.TestCase):
    def test_runtime_kind_scheme_source_matrix(self):
        known = {
            (c.runtime, c.layer_kind, c.scheme, c.source): c for c in MXFP_CAPABILITIES
        }
        self.assertEqual(len(known), len(MXFP_CAPABILITIES))
        for key in product(
            ("srt", "diffusion"),
            ("dense", "moe"),
            (W8A8, W4A8, W4A4),
            ("online", "modelslim"),
        ):
            with self.subTest(key=key):
                if key not in known:
                    with self.assertRaisesRegex(
                        ValueError, "no implementation.*no fallback"
                    ):
                        get_mxfp_capability(*key)
                    continue
                cap = get_mxfp_capability(*key)
                reason = validate_mxfp_configuration(
                    cap, platform=PlatformInfo(device_type="npu"), input_size=64
                )
                self.assertEqual(reason, cap.fallback)
                self.assertEqual(
                    cap.hardware_validation, "Not validated on this revision"
                )
                require_mxfp_operators(cap, cap.required_ops)
                if cap.fallback:
                    self.assertIsNone(cap.kernel)
                    self.assertEqual(cap.required_ops, ())
                else:
                    self.assertEqual(cap.kernel.backend, KernelBackend.TORCH_NPU)
                    self.assertFalse(
                        cap.kernel.is_available(PlatformInfo(device_type="cpu"))
                    )
                    self.assertTrue(
                        cap.kernel.is_available(PlatformInfo(device_type="npu"))
                    )

    def test_unknown_hardware_validation_does_not_mean_unsupported(self):
        cap = get_mxfp_capability("srt", "dense", W8A8, "online")
        self.assertIsNone(
            validate_mxfp_configuration(cap, platform=PlatformInfo(device_type="npu"))
        )
        for device in ("cpu", "cuda", "hip"):
            with (
                self.subTest(device=device),
                self.assertRaisesRegex(ValueError, "requires NPU"),
            ):
                validate_mxfp_configuration(
                    cap, platform=PlatformInfo(device_type=device)
                )

    def test_dense_dual_is_not_moe_single(self):
        dense = get_mxfp_capability("srt", "dense", W4A4, "online")
        moe = get_mxfp_capability("srt", "moe", W4A4, "modelslim")
        self.assertEqual(dense.scale_mode, "dual")
        self.assertEqual(moe.scale_mode, "single")
        self.assertIn("npu_dual_level_quant_matmul", dense.required_ops)
        self.assertNotIn("npu_dual_level_quant_matmul", moe.required_ops)
        self.assertIn("uint8", moe.checkpoint_dtype)
        from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp4_scheme import (
            ModelSlimMXFP4Scheme,
        )

        # Evidence for the blocker: this is a pre-existing constructor error,
        # not an inference from absent hardware validation.
        with self.assertRaisesRegex(AttributeError, "is_dual_scale_key"):
            ModelSlimMXFP4Scheme(
                {"linear.weight_dual_scale": "FLOAT"}, "linear", W4A4_DUAL
            )
        blocked = get_mxfp_capability("diffusion", "dense", W4A4_DUAL, "modelslim")
        with self.assertRaisesRegex(ValueError, "undefined is_dual_scale_key"):
            validate_mxfp_configuration(blocked)

    def test_raw_checkpoint_dtype_shape_and_scales(self):
        for cap in MXFP_CAPABILITIES:
            if cap.source != "modelslim" or cap.blocker:
                continue
            with self.subTest(cap=cap.key):
                shape = (4, 8, 64) if cap.layer_kind == "moe" else (8, 64)
                w8 = cap.scheme == W8A8
                weight = torch.empty(
                    (*shape[:-1], 64 if w8 else 32),
                    dtype=torch.float8_e4m3fn if w8 else torch.uint8,
                )
                scale = torch.full((*shape[:-1], 2), 127, dtype=torch.uint8)
                validate_mxfp_loaded_weights(cap, weight, scale)
                self.assertTrue(torch.all(scale == 127))
                with self.assertRaisesRegex(ValueError, "checkpoint requires"):
                    validate_mxfp_loaded_weights(cap, weight.float(), scale)
                with self.assertRaisesRegex(ValueError, "checkpoint requires"):
                    validate_mxfp_loaded_weights(cap, weight, scale.float())
                with self.assertRaisesRegex(
                    ValueError, "incompatible checkpoint.*shapes"
                ):
                    validate_mxfp_loaded_weights(cap, weight, scale[..., :1, :])
                with self.assertRaisesRegex(ValueError, "expected ceil\\(K/32\\)"):
                    validate_mxfp_loaded_weights(cap, weight, scale[..., :1])
                with self.assertRaisesRegex(
                    ValueError, "requires weight and weight_scale"
                ):
                    validate_mxfp_loaded_weights(cap, weight, None)

    def test_current_offline_loader_rejects_unpairable_scale_count(self):
        for cap in MXFP_CAPABILITIES:
            if cap.source != "modelslim" or cap.input_alignment != 64:
                continue
            for k in (0, 32, 96, 4304):
                with (
                    self.subTest(cap=cap.key, k=k),
                    self.assertRaisesRegex(
                        ValueError, "current modelslim loader/operator contract"
                    ),
                ):
                    validate_mxfp_configuration(cap, input_size=k)

    def test_diffusion_mxfp8_partial_scales_reach_shared_post_load(self):
        from sglang.multimodal_gen.runtime.layers.quantization import (
            modelslim as diffusion,
        )

        config = diffusion.ModelSlimConfig({"linear.weight": W8A8})
        method = diffusion.ModelSlimLinearMethod(config)
        cap = get_mxfp_capability("diffusion", "dense", W8A8, "modelslim")
        for k in (32, 64, 96, 4304):
            with self.subTest(k=k):
                layer = torch.nn.Module()
                layer.scheme = config.get_scheme(layer, "linear")
                method.create_weights(
                    layer,
                    k,
                    [8],
                    k,
                    8,
                    torch.bfloat16,
                    weight_loader=lambda *args: None,
                )
                blocks = (k + 31) // 32
                self.assertEqual(layer.weight_scale.shape, (8, blocks))
                layer.weight.data.copy_(
                    torch.zeros(8, k, dtype=torch.uint8).view(torch.float8_e4m3fn)
                )
                layer.weight_scale.data.fill_(127)
                validate_mxfp_loaded_weights(cap, layer.weight, layer.weight_scale)
                with self.assertRaisesRegex(ValueError, "expected ceil\\(K/32\\)"):
                    validate_mxfp_loaded_weights(
                        cap, layer.weight, layer.weight_scale[..., :-1]
                    )
                with patch.object(diffusion, "require_mxfp_operators") as require:
                    method.process_weights_after_loading(layer)
                require.assert_called_once_with(cap)
                self.assertEqual(layer.weight.shape, (k, 8))
                self.assertEqual(layer.weight_scale_inv.shape, ((k + 63) // 64, 8, 2))
                self.assertNotIn("weight_scale", dict(layer.named_parameters()))

    def test_backend_clamp_and_existing_unfused_path(self):
        mx8 = get_mxfp_capability("srt", "moe", W8A8, "online")
        for kwargs, text in (
            ({"runner": "flashinfer_trtllm"}, "moe-runner-backend"),
            ({"a2a_backend": "ascend_fuseep"}, "different INT8"),
            ({"swiglu_limit": 7.0}, "no clamp or unfused"),
            ({"activation": "gelu"}, "requires SiLU"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, text):
                validate_mxfp_configuration(mx8, **kwargs)
        w4a8 = get_mxfp_capability("srt", "moe", W4A8, "modelslim")
        self.assertIsNone(
            validate_mxfp_configuration(
                w4a8,
                runner="ascend",
                a2a_backend="deepep",
                swiglu_limit=7.0,
                activation="silu",
            )
        )
        require_mxfp_operators(w4a8, w4a8.required_ops, fused=False)
        with self.assertRaisesRegex(ValueError, "npu_grouped_matmul_swiglu_quant_v2"):
            require_mxfp_operators(w4a8, w4a8.required_ops, fused=True)

    def test_operator_presence_failure_before_online_weight_processing(self):
        for method_class in (
            linear_method_npu.NPUMXFP8LinearMethod,
            linear_method_npu.NPUMXFP4W4A8LinearMethod,
            linear_method_npu.NPUDualLevelMXFP4LinearMethod,
        ):
            with self.subTest(method=method_class.__name__):
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(
                    torch.ones(8, 64, dtype=torch.bfloat16), requires_grad=False
                )
                weight = layer.weight
                with patch.object(
                    linear_method_npu,
                    "require_mxfp_operators",
                    side_effect=lambda cap: require_mxfp_operators(cap, ()),
                ):
                    with self.assertRaisesRegex(ValueError, "operators not registered"):
                        method_class().process_weights_after_loading(layer)
                self.assertIs(layer.weight, weight)
                self.assertEqual(list(layer.named_parameters()), [("weight", weight)])

    def test_fp4_enum_and_ue8m0_dtype_preflight(self):
        cap = get_mxfp_capability("srt", "dense", W4A8, "modelslim")
        ops = SimpleNamespace(**{name: object() for name in cap.required_ops})
        with (
            patch("sglang.srt.utils.is_npu", return_value=True),
            patch.object(
                PlatformInfo, "detect", return_value=PlatformInfo(device_type="npu")
            ),
            patch.object(torch.ops, "npu", ops),
        ):
            with patch.dict(
                "sys.modules",
                {"torch_npu": SimpleNamespace(float4_e2m1fn_x2=torch.float8_e4m3fn)},
            ):
                with self.assertRaisesRegex(ValueError, "packed FP4 enum"):
                    require_mxfp_operators(cap)
            with (
                patch.dict(
                    "sys.modules", {"torch_npu": SimpleNamespace(float4_e2m1fn_x2=296)}
                ),
                patch.object(torch, "float8_e8m0fnu", None, create=True),
            ):
                with self.assertRaisesRegex(ValueError, "float8_e8m0fnu is required"):
                    require_mxfp_operators(cap)

    def test_modelslim_selectors_and_method_preflight(self):
        from sglang.multimodal_gen.runtime.layers.quantization import (
            modelslim as diffusion,
        )
        from sglang.srt.layers.quantization.modelslim import modelslim as srt

        for runtime, module, schemes in (
            ("srt", srt, (W8A8, W4A8, W4A4)),
            ("diffusion", diffusion, (W8A8, W4A4)),
        ):
            for name in schemes:
                with self.subTest(runtime=runtime, scheme=name):
                    config = module.ModelSlimConfig({"linear.weight": name})
                    layer = torch.nn.Module()
                    layer.scheme = (
                        config.get_linear_scheme(layer, "linear")
                        if runtime == "srt"
                        else config.get_scheme(layer, "linear")
                    )
                    self.assertIs(
                        layer.scheme.mxfp_capability,
                        get_mxfp_capability(runtime, "dense", name, "modelslim"),
                    )
                    method = module.ModelSlimLinearMethod(config)
                    with patch.object(layer.scheme, "create_weights") as create:
                        with self.assertRaisesRegex(ValueError, "local input K=0"):
                            method.create_weights(layer, 0, [8], 0, 8, torch.bfloat16)
                        create.assert_not_called()
                    layer.weight = torch.nn.Parameter(
                        torch.empty(8, 64, dtype=torch.bfloat16), requires_grad=False
                    )
                    layer.weight_scale = torch.nn.Parameter(
                        torch.empty(8, 2, dtype=torch.uint8), requires_grad=False
                    )
                    with patch.object(
                        layer.scheme, "process_weights_after_loading"
                    ) as process:
                        with self.assertRaisesRegex(ValueError, "checkpoint requires"):
                            method.process_weights_after_loading(layer)
                        process.assert_not_called()
        config = diffusion.ModelSlimConfig({"linear.weight": W4A4_DUAL})
        with self.assertRaisesRegex(ValueError, "undefined is_dual_scale_key"):
            config.get_scheme(torch.nn.Module(), "linear")

    def test_modelslim_moe_selection_and_local_partition_preflight(self):
        from sglang.srt.layers.quantization.modelslim import modelslim as srt

        for name, scheme_cls in (
            (W8A8, srt.ModelSlimMXFP8MoEScheme),
            (W4A8, srt.ModelSlimW4A8MXFP4MoE),
            (W4A4, srt.ModelSlimW4A4MXFP4MoE),
        ):
            with self.subTest(scheme=name):
                cap = get_mxfp_capability("srt", "moe", name, "modelslim")
                kernel_name = cap.kernel.target.split(":")[1].split(".")[0]
                config = srt.ModelSlimConfig(
                    {
                        f"experts.0.{proj}.weight": name
                        for proj in ("gate_proj", "up_proj", "down_proj")
                    }
                )
                layer = torch.nn.Module()
                with patch(
                    f"{scheme_cls.__module__}.{kernel_name}",
                    return_value=SimpleNamespace(use_fused_gmm1=True),
                ):
                    layer.w13_scheme, layer.w2_scheme = config.get_moe_scheme(
                        layer, "experts"
                    )
                self.assertIs(layer.w13_scheme.mxfp_capability, cap)
                self.assertIs(layer.w2_scheme.mxfp_capability, cap)
                method = srt.ModelSlimFusedMoEMethod(config)
                with (
                    patch.object(layer.w13_scheme, "create_weights") as w13,
                    patch.object(layer.w2_scheme, "create_weights") as w2,
                ):
                    with self.assertRaisesRegex(ValueError, "local input K=32"):
                        method.create_weights(layer, 4, 64, 32, torch.bfloat16)
                    w13.assert_not_called()
                    w2.assert_not_called()
                layer.w13_weight = torch.nn.Parameter(
                    torch.empty(4, 8, 64, dtype=torch.bfloat16), requires_grad=False
                )
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.empty(4, 8, 2, dtype=torch.uint8), requires_grad=False
                )
                with patch.object(
                    layer.w13_scheme, "process_weights_after_loading"
                ) as process:
                    with self.assertRaisesRegex(ValueError, "checkpoint requires"):
                        method.process_weights_after_loading(layer)
                    process.assert_not_called()
                backend = SimpleNamespace(
                    value="flashinfer_trtllm", is_auto=lambda: False
                )
                runner_config = SimpleNamespace(activation="silu", swiglu_limit=None)
                with (
                    patch.object(srt, "get_moe_runner_backend", return_value=backend),
                    patch.object(
                        srt,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(value="none"),
                    ),
                    patch.object(srt, "MoeRunner") as runner,
                ):
                    with self.assertRaisesRegex(ValueError, "moe-runner-backend"):
                        method.create_moe_runner(layer, runner_config)
                    runner.assert_not_called()

    def test_online_moe_runner_guard_precedes_kernel_creation(self):
        from sglang.srt.hardware_backend.npu.quantization import (
            online_moe_methods as online,
        )

        method = online.NPUMXFP8OnlineMoEMethod.__new__(online.NPUMXFP8OnlineMoEMethod)
        backend = SimpleNamespace(value="flashinfer_trtllm")
        config = SimpleNamespace(activation="silu", swiglu_limit=None)
        with (
            patch.object(online, "get_moe_runner_backend", return_value=backend),
            patch.object(
                online,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(value="none"),
            ),
            patch.object(online, "NPUMXFP8MoEMethod") as kernel,
        ):
            with self.assertRaisesRegex(ValueError, "moe-runner-backend"):
                method.create_moe_runner(torch.nn.Module(), config)
            kernel.assert_not_called()

    def test_existing_online_w4_moe_fallback_selection(self):
        from sglang.srt.layers.quantization.npu_mxfp4 import Mxfp4W4A8Config
        from sglang.srt.layers.quantization.npu_mxfp4_w4a4 import Mxfp4W4A4Config
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        class FakeMoE(torch.nn.Module):
            use_triton_kernels = False
            use_flashinfer_trtllm_moe = False

        for config_class, name in ((Mxfp4W4A8Config, W4A8), (Mxfp4W4A4Config, W4A4)):
            with (
                self.subTest(scheme=name),
                patch("sglang.srt.layers.moe.fused_moe_triton.FusedMoE", FakeMoE),
            ):
                with self.assertLogs(config_class.__module__, level="WARNING") as logs:
                    method = config_class().get_quant_method(FakeMoE(), "experts")
                self.assertIsInstance(method, UnquantizedFusedMoEMethod)
                self.assertIn(
                    get_mxfp_capability("srt", "moe", name, "online").fallback,
                    logs.output[0],
                )

    def test_kernel_targets_exist_without_importing_npu_backends(self):
        root = Path(__file__).resolve().parents[5]
        for cap in MXFP_CAPABILITIES:
            if cap.kernel is None:
                continue
            with self.subTest(cap=cap.key):
                module, target = cap.kernel.target.split(":")
                cls_name, method = target.split(".")
                tree = ast.parse(
                    (root / "python" / (module.replace(".", "/") + ".py")).read_text()
                )
                classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

                def implements(name):
                    cls = classes[name]
                    return any(
                        isinstance(n, ast.FunctionDef) and n.name == method
                        for n in cls.body
                    ) or any(
                        isinstance(base, ast.Name)
                        and base.id in classes
                        and implements(base.id)
                        for base in cls.bases
                    )

                self.assertTrue(implements(cls_name))

    def test_support_document_matches_inventory(self):
        root = Path(__file__).resolve().parents[5]
        doc = (
            root
            / "docs/docs/hardware-platforms/ascend-npus/optimization/mxfp-capabilities.mdx"
        ).read_text()
        table = doc.split("{/* BEGIN ASCEND MXFP CAPABILITIES */}", 1)[1].split(
            "{/* END ASCEND MXFP CAPABILITIES */}", 1
        )[0]
        normalize = lambda s: re.sub(
            r"\s+", " ", re.sub(r"\s*<", "<", re.sub(r">\s*", ">", s))
        ).strip()
        self.assertEqual(normalize(table), normalize(render_mxfp_support_table()))


if __name__ == "__main__":
    unittest.main()
