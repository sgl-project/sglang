"""GPU-free BaseFusedOp + registry / spec unit tests (RFC #29630).

Every-backend-vs-native parity on real hardware lives in
``test_fused_op_gpu_parity.py``.
"""

import math

import pytest
import torch

import sglang.kernels as K
from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.registry import KernelRegistry
from sglang.kernels.spec import CapabilityRequirement as Cap
from sglang.kernels.spec import KernelBackend, KernelSpec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=23, suite="base-a-test-cpu")


class _ToyAdd(BaseFusedOp):
    op = "test.toy_add"
    priority = (KernelBackend.TRITON, KernelBackend.TORCH)

    def forward_native(self, a, b):
        return a + b

    def forward_triton(self, a, b):
        return a + b + 1000  # marker so tests can tell which backend ran


class _CudaOnlyToy(BaseFusedOp):
    op = "test.toy_cuda_only"
    priority = (KernelBackend.AOT, KernelBackend.TORCH)
    capabilities = {KernelBackend.AOT: {Cap.CUDA}}

    def forward_native(self, a):
        return a * 2

    def forward_aot(self, a):
        raise AssertionError("CUDA backend must not be selected on a CPU-only box")


@pytest.fixture(autouse=True)
def _reset_global_state():
    yield
    K.set_fused_op_backend(None)
    K.disable_fused_op_trace()
    K.clear_fused_op_trace()


def _t(*vals):
    return torch.tensor(list(vals))


def test_available_backends():
    assert set(_ToyAdd().available_backends()) == {
        KernelBackend.TORCH,
        KernelBackend.TORCH_COMPILE,
        KernelBackend.TRITON,
    }


def test_priority_dispatch():
    # TRITON is first in priority and always eligible (no capability).
    assert _ToyAdd()(_t(1.0), _t(2.0)).item() == 1003.0


def test_explicit_backend_overrides_priority():
    assert (
        _ToyAdd().forward(_t(1.0), _t(2.0), backend=KernelBackend.TORCH).item() == 3.0
    )


def test_capability_gates_eligibility():
    if K.PlatformInfo.detect().is_cuda:
        pytest.skip("requires a CPU-only environment")
    # CUDA backend is filtered out; auto-selection falls back to native.
    assert _CudaOnlyToy()(_t(3.0)).item() == 6.0


def test_forced_backend_global_switch():
    op = _ToyAdd()
    K.set_fused_op_backend(KernelBackend.TORCH)
    assert op(_t(1.0), _t(2.0)).item() == 3.0
    K.set_fused_op_backend(None)
    assert op(_t(1.0), _t(2.0)).item() == 1003.0


def test_forced_backend_env_var():
    import sglang.kernels.fused_op as m
    from sglang.srt.environ import envs

    op = _ToyAdd()
    with envs.SGLANG_FORCE_FUSED_OP_BACKEND.override("torch"):
        m._forced_backend = m._UNRESOLVED  # drop cache so the env var is re-read
        assert K.get_fused_op_backend() is KernelBackend.TORCH
        assert op(_t(1.0), _t(2.0)).item() == 3.0
    m._forced_backend = m._UNRESOLVED


def test_unimplemented_backend_raises():
    with pytest.raises(NotImplementedError):
        _ToyAdd().forward(_t(1.0), _t(2.0), backend=KernelBackend.AOT)


def test_trace_records_op_backend_and_shapes():
    K.enable_fused_op_trace()
    _ToyAdd()(torch.zeros(2, 3), torch.zeros(2, 3))
    (rec,) = K.get_fused_op_trace()
    assert rec.op == "test.toy_add"
    assert rec.backend == "triton"
    assert rec.tensor_args == ("torch.float32[2, 3]", "torch.float32[2, 3]")


def test_fused_op_registers_all_backends():
    backends = {s.backend for s in K.registry.get("layernorm.rmsnorm")}
    assert backends == {
        KernelBackend.TORCH,
        KernelBackend.TORCH_COMPILE,
        KernelBackend.JIT,
        KernelBackend.AOT,
        KernelBackend.AITER,
        KernelBackend.TORCH_NPU,
    }


# --- KernelRegistry unit ---


def _spec(op="g.n", backend=KernelBackend.TORCH, target="math:sqrt"):
    return KernelSpec(op=op, backend=backend, target=target)


def test_registry_register_and_get():
    reg = KernelRegistry()
    spec = _spec()
    reg.register(spec)
    assert reg.get("g.n") == [spec]
    assert reg.has("g.n")
    assert reg.ops() == ["g.n"]
    assert reg.get("no.such") == []


def test_registry_reregister_replaces():
    reg = KernelRegistry()
    reg.register(_spec(target="math:sqrt"))
    reg.register(_spec(target="math:floor"))
    assert [s.target for s in reg.get("g.n")] == ["math:floor"]


def test_registry_get_backend_missing_raises():
    reg = KernelRegistry()
    reg.register(_spec())
    with pytest.raises(KeyError):
        reg.get_backend("g.n", KernelBackend.TRITON)


# --- KernelSpec.load ---


def test_spec_load_simple_and_dotted():
    assert _spec(target="math:sqrt").load() is math.sqrt
    dotted = _spec(target="sglang.kernels.ops.layernorm:_RMSNORM.forward_native")
    assert callable(dotted.load())


def test_spec_load_bad_target_raises():
    with pytest.raises(ValueError):
        _spec(target="no-colon").load()


# --- native reference math of the reworked ops (CPU) ---


def _ref_rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


def test_rmsnorm_native():
    from sglang.kernels.ops.layernorm import _RMSNORM

    x, w = torch.randn(8, 128), torch.randn(128)
    out = _RMSNORM.forward_native(x, w, 1e-6)
    assert torch.allclose(out, _ref_rmsnorm(x, w, 1e-6))
    buf = torch.empty_like(x)
    assert _RMSNORM.forward_native(x, w, 1e-6, out=buf) is buf  # out= is in place
    assert torch.allclose(buf, out)


def test_fused_add_rmsnorm_native():
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

    x, residual, w = torch.randn(8, 128), torch.randn(8, 128), torch.randn(128)
    x0, r0 = x.clone(), residual.clone()
    assert _FUSED_ADD_RMSNORM.forward_native(x, residual, w, 1e-6) is None
    acc = x0.float() + r0.float()
    assert torch.allclose(residual, acc)
    assert torch.allclose(
        x, acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    )


@pytest.mark.parametrize(
    "op_attr, approximate",
    [
        ("_SILU_AND_MUL", None),
        ("_GELU_AND_MUL", "none"),
        ("_GELU_TANH_AND_MUL", "tanh"),
    ],
)
def test_gated_activation_native(op_attr, approximate):
    import torch.nn.functional as F

    import sglang.kernels.ops.activation as act

    x = torch.randn(8, 256)
    gate, up = x[..., :128], x[..., 128:]
    ref = (
        F.silu(gate) if approximate is None else F.gelu(gate, approximate=approximate)
    ) * up
    assert torch.allclose(getattr(act, op_attr).forward_native(x), ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
