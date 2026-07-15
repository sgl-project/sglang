"""GPU-free import/registry tests for the ``sglang.kernels`` namespace.

Part of RFC #29630, Phase 2. These tests exercise the public namespace, the
kernel registry, and the heuristic selector without touching a GPU or importing
any kernel backend (``sgl_kernel`` / ``sglang.jit_kernel``). They run in the CPU
CI lane.
"""

import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A must-contain subset of registered operators and their backends. The
# registry holds many more entries (every migrated Triton kernel), so this is
# checked as a subset, not an exact match.
EXPECTED_OPS = {
    # BaseFusedOp-backed ops: native + torch_compile always available,
    # plus the overridden CUDA backends.
    "activation.silu_and_mul": {"aot", "jit", "torch", "torch_compile"},
    "activation.gelu_and_mul": {"aot", "jit", "torch", "torch_compile"},
    "activation.gelu_tanh_and_mul": {
        "aot",
        "jit",
        "torch",
        "torch_compile",
    },
    "layernorm.rmsnorm": {"aot", "jit", "torch", "torch_compile"},
    "layernorm.fused_add_rmsnorm": {
        "aot",
        "jit",
        "torch",
        "torch_compile",
    },
    "layernorm.gemma_rmsnorm": {"aot", "torch", "torch_compile"},
    "layernorm.gemma_fused_add_rmsnorm": {"aot", "torch", "torch_compile"},
    # curated dual/single-backend wrapper ops
    "gemm.fp8_scaled_mm": {"aot"},
    "gemm.dsv3_fused_a_gemm": {"aot", "jit"},
    "gemm.dsv3_router_gemm": {"jit"},
    "kvcache.reshape_and_cache_flash": {"triton"},
    "moe.moe_align_block_size": {"aot", "jit"},
    "moe.topk_softmax": {"aot"},
    "quantization.sgl_per_token_quant_fp8": {"aot"},
    # migrated from srt/layers/quantization (Phase 2.5)
    "quantization.w8a8_block_fp8_matmul": {"triton"},
    "quantization.per_token_quant_int8": {"triton"},
    "quantization.awq_dequantize_triton": {"triton"},
    "quantization.nvfp4_gemm_swiglu_nvfp4_quant": {"cute_dsl"},
    "moe.pack_topk_ids": {"triton"},
    "quantization.sgl_per_token_group_quant_8bit": {"aot", "jit"},
    "quantization.sgl_per_token_group_quant_fp8": {"aot"},
    "quantization.sgl_per_token_group_quant_int8": {"aot"},
    # deferred-group wrappers, now populated
    "sampling.top_k_renorm_probs": {"aot"},
    "sampling.top_p_renorm_probs": {"aot"},
    "spatial.get_sm_available": {"aot"},
    "spatial.create_greenctx_stream_by_value": {"aot"},
    "mamba.causal_conv1d_fwd": {"aot"},
    "mamba.causal_conv1d_update": {"aot"},
    "diffusion.apply_group_norm_silu": {"jit"},
    "diffusion.residual_gate_add": {"jit"},
    "diffusion.fused_inplace_qknorm_rope": {"jit"},
    # representative migrated Triton kernels (inventory)
    "grammar.apply_token_bitmask_inplace_triton": {"triton"},
    "memory.alloc_extend_kernel": {"triton"},
    "attention.decode_attention_fwd": {"triton"},
    "kvcache.create_flashinfer_kv_indices_triton": {"triton"},
    "speculative.gather_spec_extras": {"triton"},
}

# Public wrapper callables that each populated group must expose.
EXPECTED_WRAPPERS = {
    "sglang.kernels.ops.layernorm": [
        "rmsnorm",
        "fused_add_rmsnorm",
        "gemma_rmsnorm",
        "gemma_fused_add_rmsnorm",
    ],
    "sglang.kernels.ops.activation": [
        "silu_and_mul",
        "gelu_and_mul",
        "gelu_tanh_and_mul",
    ],
    "sglang.kernels.ops.gemm": [
        "fp8_scaled_mm",
        "dsv3_fused_a_gemm",
        "dsv3_router_gemm",
    ],
    "sglang.kernels.ops.quantization": [
        "sgl_per_token_quant_fp8",
        "sgl_per_token_group_quant_8bit",
        "sgl_per_token_group_quant_fp8",
        "sgl_per_token_group_quant_int8",
    ],
    "sglang.kernels.ops.moe": ["moe_align_block_size", "topk_softmax"],
    "sglang.kernels.ops.kvcache": ["reshape_and_cache_flash"],
    "sglang.kernels.ops.sampling": ["top_k_renorm_probs", "top_p_renorm_probs"],
    "sglang.kernels.ops.spatial": [
        "get_sm_available",
        "create_greenctx_stream_by_value",
    ],
    "sglang.kernels.ops.mamba": ["causal_conv1d_fwd", "causal_conv1d_update"],
    "sglang.kernels.ops.diffusion": [
        "apply_group_norm_silu",
        "residual_gate_add",
        "fused_inplace_qknorm_rope",
    ],
}

# All operator groups from the RFC's proposed shape must import as packages.
ALL_GROUPS = [
    "activation",
    "attention",
    "communication",
    "diffusion",
    "gemm",
    "grammar",
    "kvcache",
    "layernorm",
    "mamba",
    "memory",
    "moe",
    "quantization",
    "sampling",
    "spatial",
    "speculative",
]


class TestKernelsNamespace(unittest.TestCase):
    def setUp(self):
        import importlib

        import sglang.kernels
        import sglang.kernels.ops  # populate the registry

        self.K = sglang.kernels
        self.importlib = importlib

    def test_top_level_exports(self):
        for name in (
            "KernelSpec",
            "KernelBackend",
            "FormatSignature",
            "CapabilityRequirement",
            "PlatformInfo",
            "registry",
            "get_kernel",
            "select_kernel",
        ):
            self.assertTrue(hasattr(self.K, name), f"missing export: {name}")

    def test_all_groups_importable(self):
        for group in ALL_GROUPS:
            mod = self.importlib.import_module(f"sglang.kernels.ops.{group}")
            self.assertTrue(hasattr(mod, "__all__"))

    def test_registry_contents(self):
        registry = self.K.registry
        ops = set(registry.ops())
        # EXPECTED_OPS is a must-contain subset (many more migrated kernels
        # are also registered).
        missing = set(EXPECTED_OPS) - ops
        self.assertFalse(missing, f"missing registered ops: {sorted(missing)}")
        for op, backends in EXPECTED_OPS.items():
            got = {s.backend.value for s in registry.get(op)}
            self.assertEqual(got, backends, f"backend mismatch for {op}")
        self.assertGreaterEqual(len(ops), 80, "registry unexpectedly small")

    def test_specs_are_well_formed(self):
        for spec in self.K.registry.all_specs():
            self.assertIn(".", spec.op)
            self.assertEqual(spec.op, f"{spec.group}.{spec.name}")
            # target must be an importable "module:attr" path
            module_path, sep, attr = spec.target.partition(":")
            self.assertEqual(sep, ":", f"bad target for {spec.op}: {spec.target}")
            self.assertTrue(module_path and attr, spec.target)

    def test_wrappers_exposed_and_callable(self):
        for module_name, names in EXPECTED_WRAPPERS.items():
            mod = self.importlib.import_module(module_name)
            for name in names:
                self.assertTrue(callable(getattr(mod, name)), f"{module_name}.{name}")

    def test_single_backend_op_resolves_without_backend(self):
        # An op with exactly one registered backend has a fixed call path.
        for op, backends in EXPECTED_OPS.items():
            if len(backends) == 1:
                spec = self.K.select_kernel(op)
                self.assertEqual(spec.backend.value, next(iter(backends)), op)

    def test_multi_backend_op_requires_explicit_backend(self):
        # Device is a HARD eligibility filter, not a preference ranking: when
        # more than one backend is usable on the current device, selection must
        # be explicit (no hidden auto-ranking). Force a CUDA platform so the
        # result is deterministic regardless of the test host.
        import sglang.kernels.selector as sel

        saved = sel._platform
        try:
            sel._platform = lambda: self.K.PlatformInfo(
                device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0
            )
            # rmsnorm exposes torch/torch_compile/jit/aot, all eligible on CUDA.
            with self.assertRaises(ValueError):
                self.K.select_kernel("layernorm.rmsnorm")
            # An explicit backend is always the fixed call path.
            spec = self.K.select_kernel(
                "layernorm.rmsnorm", backend=self.K.KernelBackend.JIT
            )
            self.assertEqual(spec.backend, self.K.KernelBackend.JIT)
        finally:
            sel._platform = saved

    def test_selector_explicit_backend(self):
        spec = self.K.select_kernel(
            "layernorm.rmsnorm", backend=self.K.KernelBackend.JIT
        )
        self.assertEqual(
            spec.target, "sglang.kernels.ops.layernorm:_RMSNORM.forward_jit"
        )

    def test_selector_unknown_op_raises(self):
        with self.assertRaises(KeyError):
            self.K.select_kernel("does_not.exist")
        with self.assertRaises(KeyError):
            self.K.select_kernel(
                "gemm.fp8_scaled_mm", backend=self.K.KernelBackend.TRITON
            )

    def test_capability_requirement_logic(self):
        cap = self.K.CapabilityRequirement
        dev = self.K.DeviceType
        plat = self.K.PlatformInfo
        cpu = plat(device_type="cpu")
        sm90 = plat(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
        sm100 = plat(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0)
        hip = plat(device_type="hip")

        self.assertFalse(cap(device=dev.CUDA).is_satisfied_by(cpu))
        self.assertTrue(cap(device=dev.CUDA).is_satisfied_by(sm90))
        self.assertFalse(cap(device=dev.CUDA).is_satisfied_by(hip))
        self.assertTrue(cap(device=dev.HIP).is_satisfied_by(hip))
        self.assertFalse(
            cap(device=dev.CUDA, min_cuda_arch=(10, 0)).is_satisfied_by(sm90)
        )
        self.assertTrue(
            cap(device=dev.CUDA, min_cuda_arch=(10, 0)).is_satisfied_by(sm100)
        )
        self.assertFalse(
            cap(device=dev.CUDA, max_cuda_arch=(9, 0)).is_satisfied_by(sm100)
        )

        # OR semantics: a (cuda, hip) tuple is satisfied by either device.
        cuda_or_hip = (cap(device=dev.CUDA), cap(device=dev.HIP))
        self.assertTrue(self.K.capabilities_satisfied(cuda_or_hip, sm90))
        self.assertTrue(self.K.capabilities_satisfied(cuda_or_hip, hip))
        self.assertFalse(self.K.capabilities_satisfied(cuda_or_hip, cpu))
        self.assertTrue(self.K.capabilities_satisfied((), cpu))  # empty = unrestricted

    def test_platform_detect_does_not_raise(self):
        plat = self.K.PlatformInfo.detect()
        self.assertIn(plat.device_type, ("cpu", "cuda", "hip"))

    def test_import_does_not_load_kernel_backends(self):
        # Importing the namespace must stay metadata-only: no sgl_kernel or
        # sglang.jit_kernel import, and no JIT compilation, on a CPU box.
        code = (
            "import sys; import sglang.kernels.ops; "
            "backend = ('sgl_kernel' in sys.modules) or "
            "any(m.startswith('sglang.jit_kernel') for m in sys.modules); "
            "print('BACKEND_IMPORTED' if backend else 'CLEAN')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CLEAN", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
