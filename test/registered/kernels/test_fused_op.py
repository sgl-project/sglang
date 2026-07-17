"""GPU-free unit tests for ``sglang.kernels``: BaseFusedOp + registry/selector/spec.

Part of RFC #29630, Phase 2. Covers the multi-backend operator contract
(structural backend detection, priority dispatch, forced backend, runtime
eligibility, tracing), the registry/selector units in isolation, and the
pure-torch reference implementations of the reworked layernorm / activation
ops. Runs in the CPU CI lane; every-backend-vs-native parity lives in
``test_fused_op_gpu_parity.py``.
"""

import unittest

import torch

import sglang.kernels as K
from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.registry import KernelRegistry
from sglang.kernels.spec import (
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _ToyAddOp(BaseFusedOp):
    """Toy op: element-wise a + b, with a fake 'triton' backend."""

    op = "test.toy_add"
    priority = (KernelBackend.TRITON, KernelBackend.TORCH)

    def forward_native(self, a, b):
        return a + b

    def forward_triton(self, a, b):
        # Marker so tests can tell which backend ran.
        return a + b + 1000


class _CudaOnlyToyOp(BaseFusedOp):
    """Toy op whose optimized backend requires CUDA (never eligible on CPU)."""

    op = "test.toy_cuda_only"
    priority = (KernelBackend.AOT, KernelBackend.TORCH)
    capabilities = {KernelBackend.AOT: {CapabilityRequirement.CUDA}}

    def forward_native(self, a):
        return a * 2

    def forward_aot(self, a):
        raise AssertionError("must not be selected on a CPU-only box")


class TestBaseFusedOp(unittest.TestCase):
    def tearDown(self):
        K.set_fused_op_backend(None)
        K.disable_fused_op_trace()
        K.clear_fused_op_trace()

    def test_structural_backend_detection(self):
        backends = set(_ToyAddOp().available_backends())
        self.assertEqual(
            backends,
            {KernelBackend.TORCH, KernelBackend.TORCH_COMPILE, KernelBackend.TRITON},
        )

    def test_native_always_available(self):
        backends = _CudaOnlyToyOp().available_backends()
        self.assertIn(KernelBackend.TORCH, backends)
        self.assertIn(KernelBackend.TORCH_COMPILE, backends)

    def test_priority_dispatch(self):
        op = _ToyAddOp()
        a, b = torch.tensor([1.0]), torch.tensor([2.0])
        # TRITON is first in priority and always eligible (no capability).
        self.assertEqual(op(a, b).item(), 1003.0)

    def test_explicit_backend_overrides_priority(self):
        op = _ToyAddOp()
        a, b = torch.tensor([1.0]), torch.tensor([2.0])
        self.assertEqual(op.forward(a, b, backend=KernelBackend.TORCH).item(), 3.0)

    def test_capability_gates_runtime_eligibility(self):
        # On a CPU-only box the CUDA backend is filtered out and auto-selection
        # falls back to native instead of raising.
        op = _CudaOnlyToyOp()
        if K.PlatformInfo.detect().is_cuda:
            self.skipTest("test requires a CPU-only environment")
        self.assertEqual(op(torch.tensor([3.0])).item(), 6.0)

    def test_forced_backend_global_switch(self):
        op = _ToyAddOp()
        a, b = torch.tensor([1.0]), torch.tensor([2.0])
        K.set_fused_op_backend(KernelBackend.TORCH)
        self.assertEqual(op(a, b).item(), 3.0)
        K.set_fused_op_backend(None)
        self.assertEqual(op(a, b).item(), 1003.0)

    def test_forced_backend_env_var(self):
        import sglang.kernels.fused_op as fused_op_module
        from sglang.srt.environ import envs

        op = _ToyAddOp()
        a, b = torch.tensor([1.0]), torch.tensor([2.0])
        with envs.SGLANG_FORCE_FUSED_OP_BACKEND.override("torch"):
            # Reset the module cache so the env var is re-read.
            fused_op_module._forced_backend = fused_op_module._UNRESOLVED
            self.assertEqual(K.get_fused_op_backend(), KernelBackend.TORCH)
            self.assertEqual(op(a, b).item(), 3.0)
        fused_op_module._forced_backend = fused_op_module._UNRESOLVED

    def test_unimplemented_backend_raises(self):
        op = _ToyAddOp()
        with self.assertRaises(NotImplementedError):
            op.forward(
                torch.tensor([1.0]),
                torch.tensor([2.0]),
                backend=KernelBackend.AOT,
            )

    def test_torch_compile_backend(self):
        op = _ToyAddOp()
        a, b = torch.tensor([1.0]), torch.tensor([2.0])
        try:
            result = op.forward(a, b, backend=KernelBackend.TORCH_COMPILE)
        except Exception as e:  # inductor toolchain missing in some CI images
            self.skipTest(f"torch.compile unavailable: {e}")
        self.assertEqual(result.item(), 3.0)

    def test_trace_records_op_backend_and_shapes(self):
        op = _ToyAddOp()
        K.enable_fused_op_trace()
        op(torch.zeros(2, 3), torch.zeros(2, 3))
        records = K.get_fused_op_trace()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].op, "test.toy_add")
        self.assertEqual(records[0].backend, "triton")
        self.assertEqual(
            records[0].tensor_args,
            ("torch.float32[2, 3]", "torch.float32[2, 3]"),
        )

    def test_register_fused_op_specs(self):
        op = K.registry.get("layernorm.rmsnorm")
        backends = {s.backend for s in op}
        self.assertEqual(
            backends,
            {
                KernelBackend.TORCH,
                KernelBackend.TORCH_COMPILE,
                KernelBackend.JIT,
                KernelBackend.AOT,
                KernelBackend.AITER,
                KernelBackend.TORCH_NPU,
            },
        )
        # Dotted targets resolve to the bound backend methods.
        native = K.registry.get_backend("layernorm.rmsnorm", KernelBackend.TORCH)
        fn = native.load()
        x = torch.randn(4, 64)
        w = torch.randn(64)
        self.assertTrue(torch.allclose(fn(x, w), _ref_rmsnorm(x, w, 1e-6)))


class TestKernelRegistryUnit(unittest.TestCase):
    """Isolated KernelRegistry behavior (fresh instance, no global state)."""

    def _spec(self, op="g.n", backend=KernelBackend.TORCH, target="math:sqrt"):
        return KernelSpec(op=op, backend=backend, target=target)

    def test_register_and_get(self):
        reg = KernelRegistry()
        spec = self._spec()
        reg.register(spec)
        self.assertEqual(reg.get("g.n"), [spec])
        self.assertTrue(reg.has("g.n"))
        self.assertEqual(reg.ops(), ["g.n"])

    def test_get_unknown_op_returns_empty(self):
        reg = KernelRegistry()
        self.assertEqual(reg.get("no.such"), [])
        self.assertFalse(reg.has("no.such"))

    def test_reregister_same_backend_replaces(self):
        reg = KernelRegistry()
        reg.register(self._spec(target="math:sqrt"))
        reg.register(self._spec(target="math:floor"))
        specs = reg.get("g.n")
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].target, "math:floor")

    def test_get_backend_missing_raises(self):
        reg = KernelRegistry()
        reg.register(self._spec(backend=KernelBackend.TORCH))
        with self.assertRaises(KeyError):
            reg.get_backend("g.n", KernelBackend.TRITON)
        with self.assertRaises(KeyError):
            reg.get_backend("no.such", KernelBackend.TORCH)


class TestKernelSpecUnit(unittest.TestCase):
    def test_load_simple_target(self):
        import math

        spec = KernelSpec(op="g.n", backend=KernelBackend.TORCH, target="math:sqrt")
        self.assertIs(spec.load(), math.sqrt)

    def test_load_dotted_target(self):
        spec = KernelSpec(
            op="g.n",
            backend=KernelBackend.TORCH,
            target="sglang.kernels.ops.layernorm:_RMSNORM.forward_native",
        )
        self.assertTrue(callable(spec.load()))

    def test_load_bad_target_raises(self):
        spec = KernelSpec(op="g.n", backend=KernelBackend.TORCH, target="no-colon")
        with self.assertRaises(ValueError):
            spec.load()


def _ref_rmsnorm(x, w, eps):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w).to(x.dtype)


class TestNativeReferenceImplementations(unittest.TestCase):
    """The forward_native math of the reworked ops, on CPU tensors."""

    def setUp(self):
        torch.manual_seed(0)

    def test_rmsnorm_native(self):
        from sglang.kernels.ops.layernorm import _RMSNORM

        x = torch.randn(8, 128)
        w = torch.randn(128)
        out = _RMSNORM.forward_native(x, w, 1e-6)
        self.assertTrue(torch.allclose(out, _ref_rmsnorm(x, w, 1e-6)))
        # out= writes in place and returns out
        buf = torch.empty_like(x)
        self.assertIs(_RMSNORM.forward_native(x, w, 1e-6, out=buf), buf)
        self.assertTrue(torch.allclose(buf, out))

    def test_fused_add_rmsnorm_native(self):
        from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

        x = torch.randn(8, 128)
        residual = torch.randn(8, 128)
        w = torch.randn(128)
        x2, r2 = x.clone(), residual.clone()
        self.assertIsNone(_FUSED_ADD_RMSNORM.forward_native(x, residual, w, 1e-6))
        acc = x2.to(torch.float32) + r2.to(torch.float32)
        self.assertTrue(torch.allclose(residual, acc))
        ref = acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        self.assertTrue(torch.allclose(x, ref))

    def test_gemma_rmsnorm_native(self):
        from sglang.kernels.ops.layernorm import _GEMMA_RMSNORM

        x = torch.randn(8, 128)
        w = torch.randn(128)
        out = _GEMMA_RMSNORM.forward_native(x, w, 1e-6)
        xf = x.to(torch.float32)
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w)
        self.assertTrue(torch.allclose(out, ref))

    def test_gated_activations_native(self):
        import torch.nn.functional as F

        from sglang.kernels.ops.activation import (
            _GELU_AND_MUL,
            _GELU_TANH_AND_MUL,
            _SILU_AND_MUL,
        )

        x = torch.randn(8, 256)
        gate, up = x[..., :128], x[..., 128:]
        cases = [
            (_SILU_AND_MUL, F.silu(gate) * up),
            (_GELU_AND_MUL, F.gelu(gate, approximate="none") * up),
            (_GELU_TANH_AND_MUL, F.gelu(gate, approximate="tanh") * up),
        ]
        for op, ref in cases:
            self.assertTrue(torch.allclose(op.forward_native(x), ref), op.op)


if __name__ == "__main__":
    unittest.main()
