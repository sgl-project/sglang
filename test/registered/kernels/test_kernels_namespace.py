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
    "activation.silu_and_mul": {"cuda_aot", "cuda_jit", "torch", "torch_compile"},
    "activation.gelu_and_mul": {"cuda_aot", "cuda_jit", "torch", "torch_compile"},
    "activation.gelu_tanh_and_mul": {
        "cuda_aot",
        "cuda_jit",
        "torch",
        "torch_compile",
    },
    "layernorm.rmsnorm": {"cuda_aot", "cuda_jit", "torch", "torch_compile"},
    "layernorm.fused_add_rmsnorm": {
        "cuda_aot",
        "cuda_jit",
        "torch",
        "torch_compile",
    },
    "layernorm.gemma_rmsnorm": {"cuda_aot", "torch", "torch_compile"},
    "layernorm.gemma_fused_add_rmsnorm": {"cuda_aot", "torch", "torch_compile"},
    # curated dual/single-backend wrapper ops
    "gemm.fp8_scaled_mm": {"cuda_aot"},
    "gemm.dsv3_fused_a_gemm": {"cuda_aot", "cuda_jit"},
    "gemm.dsv3_router_gemm": {"cuda_jit"},
    "kvcache.reshape_and_cache_flash": {"triton"},
    "moe.moe_align_block_size": {"cuda_aot", "cuda_jit"},
    "moe.topk_softmax": {"cuda_aot"},
    "quantization.sgl_per_token_quant_fp8": {"cuda_aot"},
    # migrated from srt/layers/quantization (Phase 2.5)
    "quantization.w8a8_block_fp8_matmul": {"triton"},
    "quantization.per_token_quant_int8": {"triton"},
    "quantization.awq_dequantize_triton": {"triton"},
    "quantization.nvfp4_gemm_swiglu_nvfp4_quant": {"cute_dsl"},
    "moe.pack_topk_ids": {"triton"},
    "quantization.sgl_per_token_group_quant_8bit": {"cuda_aot", "cuda_jit"},
    "quantization.sgl_per_token_group_quant_fp8": {"cuda_aot"},
    "quantization.sgl_per_token_group_quant_int8": {"cuda_aot"},
    # deferred-group wrappers, now populated
    "sampling.top_k_renorm_probs": {"cuda_aot"},
    "sampling.top_p_renorm_probs": {"cuda_aot"},
    "spatial.get_sm_available": {"cuda_aot"},
    "spatial.create_greenctx_stream_by_value": {"cuda_aot"},
    "mamba.causal_conv1d_fwd": {"cuda_aot"},
    "mamba.causal_conv1d_update": {"cuda_aot"},
    "diffusion.apply_group_norm_silu": {"cuda_jit"},
    "diffusion.residual_gate_add": {"cuda_jit"},
    "diffusion.fused_inplace_qknorm_rope": {"cuda_jit"},
    # representative migrated Triton kernels (inventory)
    "grammar.apply_token_bitmask_inplace_triton": {"triton"},
    "memory.alloc_extend_kernel": {"triton"},
    "memory.write_req_to_token_pool": {"triton"},
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

    def test_every_in_tree_spec_target_names_a_symbol_that_exists(self):
        checked = 0
        for spec in self.K.registry.all_specs():
            module_path, _, attr = spec.target.partition(":")
            if not module_path.startswith("sglang.kernels.ops."):
                continue
            head, _, sub_attr = attr.partition(".")

            source = self._module_source(module_path)
            self.assertIsNotNone(source, f"{spec.op}: no source for {module_path}")
            checked += 1
            self.assertIn(
                head,
                self._top_level_names(source),
                f"{spec.op}: stale target {spec.target} "
                f"({head} is not defined in {module_path})",
            )

            if sub_attr:
                methods = self._class_methods(source, head)
                if methods is not None:
                    self.assertIn(
                        sub_attr,
                        methods,
                        f"{spec.op}: stale target {spec.target} "
                        f"({head} has no {sub_attr})",
                    )

        self.assertGreaterEqual(checked, 30, "in-tree targets unexpectedly few")

    def _module_source(self, module_path: str):
        import ast
        import importlib.util
        import pathlib

        cache = self.__dict__.setdefault("_source_cache", {})
        if module_path in cache:
            return cache[module_path]

        try:
            spec = importlib.util.find_spec(module_path)
        except (ImportError, AttributeError, ValueError):
            spec = None
        origin = getattr(spec, "origin", None) if spec is not None else None
        tree = None
        if origin and origin.endswith(".py"):
            tree = ast.parse(pathlib.Path(origin).read_text())
        cache[module_path] = tree
        return tree

    @staticmethod
    def _top_level_names(tree) -> set:
        import ast

        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.If, ast.Try)):
                for inner in ast.walk(node):
                    if isinstance(
                        inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        names.add(inner.name)
                    elif isinstance(inner, (ast.Import, ast.ImportFrom)):
                        for alias in inner.names:
                            names.add((alias.asname or alias.name).split(".")[0])
                    elif isinstance(inner, ast.Assign):
                        for target in inner.targets:
                            if isinstance(target, ast.Name):
                                names.add(target.id)
        return names

    @staticmethod
    def _class_methods(tree, class_name: str):
        import ast

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return {
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
        return None

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
        # No hidden ranking: a multi-backend op must be resolved explicitly.
        multi = [op for op, b in EXPECTED_OPS.items() if len(b) > 1]
        self.assertTrue(multi)  # sanity: we do have multi-backend ops
        for op in multi:
            with self.assertRaises(ValueError):
                self.K.select_kernel(op)

    def test_selector_explicit_backend(self):
        spec = self.K.select_kernel(
            "layernorm.rmsnorm", backend=self.K.KernelBackend.CUDA_JIT
        )
        self.assertEqual(
            spec.target, "sglang.kernels.ops.layernorm:_RMSNORM.forward_cuda_jit"
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
        plat = self.K.PlatformInfo
        cpu = plat(device_type="cpu")
        sm90 = plat(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
        sm100 = plat(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0)

        self.assertFalse(cap(requires_cuda=True).is_satisfied_by(cpu))
        self.assertTrue(cap(requires_cuda=True).is_satisfied_by(sm90))
        self.assertFalse(
            cap(requires_cuda=True, min_cuda_arch=(10, 0)).is_satisfied_by(sm90)
        )
        self.assertTrue(
            cap(requires_cuda=True, min_cuda_arch=(10, 0)).is_satisfied_by(sm100)
        )
        self.assertFalse(
            cap(requires_cuda=True, max_cuda_arch=(9, 0)).is_satisfied_by(sm100)
        )

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
