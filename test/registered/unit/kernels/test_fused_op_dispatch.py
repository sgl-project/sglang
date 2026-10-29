"""Dispatch-contract tests for the unified ``BaseFusedOp`` (RFC #29630, #26426).

``BaseFusedOp`` replaced ``MultiPlatformOp`` as the single operator
abstraction; these tests pin down the parts of that contract that a refactor
could silently break:

- the priority ladder: explicit ``backend=`` > global forced backend > OOT
  platform override > declared optimized kernel backends > platform-specific
  forward > native fallback;
- the standard ``nn.Module`` behavior (hooks, traversal);
- static-dispatch caching and per-call ``backend_eligible`` gating;
- the torch.compile enter/leave protocol (idempotency, TopK / FusedMoE
  special paths);
- the deprecated ``MultiPlatformOp`` alias and its OOT plugin surface.

Platform detection is mocked, so everything here runs on a CPU-only box.
"""

import warnings

import pytest
import torch
from torch import nn

import sglang.kernels.fused_op as fo
from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.spec import CapabilityRequirement as Cap
from sglang.kernels.spec import KernelBackend, PlatformInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

_CUDA = PlatformInfo(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
_HIP = PlatformInfo(device_type="hip")
_CPU = PlatformInfo()


@pytest.fixture(autouse=True)
def _reset_global_state():
    saved_oot = {k: dict(v) for k, v in BaseFusedOp._oot_forward_registry.items()}
    saved_compile = {
        k: dict(v) for k, v in BaseFusedOp._oot_compile_forward_registry.items()
    }
    yield
    fo.set_fused_op_backend(None)
    fo.disable_fused_op_trace()
    fo.clear_fused_op_trace()
    BaseFusedOp._oot_forward_registry.clear()
    BaseFusedOp._oot_forward_registry.update(saved_oot)
    BaseFusedOp._oot_compile_forward_registry.clear()
    BaseFusedOp._oot_compile_forward_registry.update(saved_compile)


def _mock_platform(monkeypatch, *, key="", info=_CPU, oot_key=None):
    monkeypatch.setattr(fo, "_platform_key", lambda: key)
    monkeypatch.setattr(fo, "_platform", lambda: info)
    monkeypatch.setattr(fo, "_oot_dispatch_key", lambda: oot_key)


@pytest.fixture
def _compiler_state(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class _OotTensorOp(BaseFusedOp):
    """Return different tensor values to identify the selected implementation.

    Plugin functions are defined outside this class, so the tests also check
    that dispatch and compilation work without platform-specific methods.
    """

    op = "test.oot_tensor"

    def __init__(self):
        super().__init__()
        self.offset = 1

    def forward_native(self, x, *, scale=1):
        return x * scale - self.offset

    def forward_cuda(self, x, *, scale=1):
        return x * scale - 2 * self.offset


class _KeepOotDispatch(_OotTensorOp):
    def _torch_compile_forward(self, num_tokens):
        return None


def forward_myplat(self, x, *, scale=1):
    return x * scale + self.offset


def forward_myplat_compile(self, x, *, scale=1):
    return x * scale + 2 * self.offset


def forward_myplat_graph_break(self, x, *, scale=1):
    y = x * scale
    torch._dynamo.graph_break()
    return y + 2 * self.offset


class _OotModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x, scale=2) * 3


class _OotModelWithBreak(_OotModel):
    def forward(self, x):
        y = self.op(x, scale=2)
        torch._dynamo.graph_break()
        return y * 3


class _AllPlatformsOp(BaseFusedOp):
    """Marks which path ran by returning its name."""

    op = "test.all_platforms"

    def forward_native(self, x):
        return "native"

    def forward_cuda(self, x):
        return "cuda"

    def forward_hip(self, x):
        return "hip"

    def forward_npu(self, x):
        return "npu"

    def forward_xpu(self, x):
        return "xpu"

    def forward_musa(self, x):
        return "musa"

    def forward_cpu(self, x):
        return "cpu"


class _CudaOnlyPlatformOp(BaseFusedOp):
    op = "test.cuda_only_platform"

    def forward_native(self, x):
        return "native"

    def forward_cuda(self, x):
        return "cuda"


class _NativeOnlyOp(BaseFusedOp):
    op = "test.native_only"

    def forward_native(self, x):
        return "native"


class _BackendAndPlatformOp(BaseFusedOp):
    """Declared JIT backend + a CUDA platform forward."""

    op = "test.backend_and_platform"
    priority = (KernelBackend.JIT, KernelBackend.TORCH)
    capabilities = {KernelBackend.JIT: frozenset({Cap.CUDA})}

    def forward_native(self, x):
        return "native"

    def forward_jit(self, x):
        return "jit"

    def forward_cuda(self, x):
        return "cuda"


class _UndeclaredBackendOp(BaseFusedOp):
    """Overrides forward_aiter but does not declare it in ``capabilities``."""

    op = "test.undeclared_backend"

    def forward_native(self, x):
        return "native"

    def forward_aiter(self, x):
        return "aiter"


# --- nn.Module contract -------------------------------------------------------


def test_is_standard_nn_module(monkeypatch):
    _mock_platform(monkeypatch)
    op = _NativeOnlyOp()
    assert isinstance(op, nn.Module)

    parent = nn.Module()
    parent.act = op
    assert dict(parent.named_modules())["act"] is op

    seen = []
    op.register_forward_hook(lambda module, args, output: seen.append(output))
    assert op(torch.zeros(1)) == "native"
    assert seen == ["native"]  # __call__ goes through nn.Module, hooks fire


# --- platform dispatch + native fallback ---------------------------------------


@pytest.mark.parametrize(
    "key, expect",
    [
        ("cuda", "cuda"),
        ("hip", "hip"),
        ("npu", "npu"),
        ("xpu", "xpu"),
        ("musa", "musa"),
        ("cpu", "cpu"),
        ("", "native"),
    ],
)
def test_platform_forward_dispatch(monkeypatch, key, expect):
    _mock_platform(monkeypatch, key=key)
    assert _AllPlatformsOp()(torch.zeros(1)) == expect


@pytest.mark.parametrize(
    "key, expect",
    [
        ("hip", "cuda"),  # HIP falls back to the CUDA path (hipified kernels)
        # MUSA has no implicit CUDA fallback: srt kernel imports are gated on
        # is_cuda(), so silently entering forward_cuda on a MUSA box can
        # NameError; ops opt in with an explicit forward_musa instead.
        ("musa", "native"),
        ("npu", "native"),  # no NPU path -> native
        ("cpu", "native"),
        ("cuda", "cuda"),
    ],
)
def test_platform_default_chains(monkeypatch, key, expect):
    _mock_platform(monkeypatch, key=key)
    assert _CudaOnlyPlatformOp()(torch.zeros(1)) == expect


def test_native_fallback_without_any_override(monkeypatch):
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    assert _NativeOnlyOp()(torch.zeros(1)) == "native"


# --- optimized-backend selection ------------------------------------------------


def test_declared_backend_beats_platform_forward(monkeypatch):
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    assert _BackendAndPlatformOp()(torch.zeros(1)) == "jit"


def test_capability_filters_backend_to_platform_forward(monkeypatch):
    # JIT is declared CUDA-only; on HIP the platform chain (-> forward_cuda) runs.
    _mock_platform(monkeypatch, key="hip", info=_HIP)
    assert _BackendAndPlatformOp()(torch.zeros(1)) == "cuda"


def test_undeclared_backend_not_auto_selected(monkeypatch):
    _mock_platform(monkeypatch, key="", info=_CPU)
    op = _UndeclaredBackendOp()
    assert op(torch.zeros(1)) == "native"
    # ... but stays reachable by explicit request.
    assert op(torch.zeros(1), backend=KernelBackend.AITER) == "aiter"


def test_priority_order_decides_between_backends(monkeypatch):
    class _TwoBackends(BaseFusedOp):
        op = "test.two_backends"
        priority = (KernelBackend.TRITON, KernelBackend.JIT, KernelBackend.TORCH)
        capabilities = {
            KernelBackend.TRITON: frozenset(),
            KernelBackend.JIT: frozenset(),
        }

        def forward_native(self, x):
            return "native"

        def forward_triton(self, x):
            return "triton"

        def forward_jit(self, x):
            return "jit"

    class _Flipped(_TwoBackends):
        priority = (KernelBackend.JIT, KernelBackend.TRITON, KernelBackend.TORCH)

    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    assert _TwoBackends()(torch.zeros(1)) == "triton"
    assert _Flipped()(torch.zeros(1)) == "jit"


def test_explicit_backend_beats_forced_global(monkeypatch):
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    op = _BackendAndPlatformOp()
    fo.set_fused_op_backend(KernelBackend.TORCH)
    assert op(torch.zeros(1)) == "native"  # forced global
    assert op(torch.zeros(1), backend=KernelBackend.JIT) == "jit"  # explicit wins


def test_forced_global_falls_back_when_unimplemented(monkeypatch):
    # The global debug switch must not take down ops that lack the forced
    # backend (e.g. forcing "torch" on a device-only op like the DSA indexer,
    # whose forward_native raises NotImplementedError).
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)

    class _DeviceOnly(BaseFusedOp):
        op = "test.device_only"

        def forward_native(self, x):
            raise NotImplementedError

        def forward_cuda(self, x):
            return "cuda"

    op = _DeviceOnly()
    fo.set_fused_op_backend(KernelBackend.TORCH)
    assert op(torch.zeros(1)) == "cuda"  # fell back to normal dispatch
    fo.set_fused_op_backend(KernelBackend.JIT)
    assert op(torch.zeros(1)) == "cuda"  # no jit backend -> fall back too
    # Explicit per-call selection stays strict.
    fo.set_fused_op_backend(None)
    with pytest.raises(NotImplementedError):
        op(torch.zeros(1), backend=KernelBackend.JIT)


def test_forced_global_beats_platform_and_oot(monkeypatch):
    _mock_platform(monkeypatch, key="", oot_key="myplat")
    BaseFusedOp.register_oot_forward(
        _CudaOnlyPlatformOp, lambda self, x: "oot", "myplat"
    )
    op = _CudaOnlyPlatformOp()
    fo.set_fused_op_backend(KernelBackend.TORCH)
    assert op(torch.zeros(1)) == "native"
    fo.set_fused_op_backend(None)
    assert op(torch.zeros(1)) == "oot"


# --- OOT platform overrides ----------------------------------------------------


def test_oot_registered_forward_wins_over_method(monkeypatch):
    class _OotOp(BaseFusedOp):
        op = "test.oot"

        def forward_native(self, x):
            return "native"

        def forward_myplat(self, x):
            return "method"

    _mock_platform(monkeypatch, oot_key="myplat")
    assert _OotOp()(torch.zeros(1)) == "method"  # forward_<key> lookup

    BaseFusedOp.register_oot_forward(_OotOp, lambda self, x: "registered", "myplat")
    assert _OotOp()(torch.zeros(1)) == "registered"  # registry beats method


def test_oot_registration_is_exact_type(monkeypatch):
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(
        _CudaOnlyPlatformOp, lambda self, x: "oot", "myplat"
    )

    class _Sub(_CudaOnlyPlatformOp):
        pass

    assert _CudaOnlyPlatformOp()(torch.zeros(1)) == "oot"
    # Subclasses do not inherit the registered forward (pre-existing
    # MultiPlatformOp semantics: lookup is by exact type).
    assert _Sub()(torch.zeros(1)) == "native"


def test_oot_falls_back_to_native(monkeypatch):
    _mock_platform(monkeypatch, oot_key="myplat")
    assert _CudaOnlyPlatformOp()(torch.zeros(1)) == "native"


def test_oot_registered_fn_is_bound(monkeypatch):
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(
        _CudaOnlyPlatformOp, lambda self, x: type(self).__name__, "myplat"
    )
    assert _CudaOnlyPlatformOp()(torch.zeros(1)) == "_CudaOnlyPlatformOp"


def test_oot_registered_bound_method_receives_op_instance(monkeypatch):
    _mock_platform(monkeypatch, oot_key="myplat")
    donor = _OotTensorOp()
    donor.offset = 9
    BaseFusedOp.register_oot_forward(_OotTensorOp, donor.forward_native, "myplat")
    op = _OotTensorOp()
    op.offset = 3
    x = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(op(x), x - 3)


# --- dispatch caching + per-call gates ------------------------------------------


def test_static_dispatch_resolved_once(monkeypatch):
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    op = _CudaOnlyPlatformOp()
    calls = []
    original = op._resolve_forward_method
    monkeypatch.setattr(
        op,
        "_resolve_forward_method",
        lambda: calls.append(1) or original(),
    )
    op(torch.zeros(1))
    op(torch.zeros(1))
    assert len(calls) == 1  # hot path must not re-resolve per call


def test_init_preseeded_forward_method_is_kept(monkeypatch):
    # srt layers pin instance paths in __init__ (e.g. env-gated aiter modes);
    # lazy resolution must not clobber that.
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    op = _AllPlatformsOp()
    op._forward_method = op.forward_xpu
    assert op(torch.zeros(1)) == "xpu"


def test_backend_eligible_override_gates_per_call(monkeypatch):
    class _Gated(BaseFusedOp):
        op = "test.gated"
        priority = (KernelBackend.JIT, KernelBackend.TORCH)
        capabilities = {KernelBackend.JIT: frozenset()}

        def forward_native(self, x):
            return "native"

        def forward_jit(self, x):
            return "jit"

        def backend_eligible(self, backend, *args, **kwargs):
            if not super().backend_eligible(backend, *args, **kwargs):
                return False
            if backend is KernelBackend.JIT:
                return args[0].shape[-1] % 2 == 0
            return True

    _mock_platform(monkeypatch, key="", info=_CPU)
    op = _Gated()
    assert op(torch.zeros(4)) == "jit"
    assert op(torch.zeros(3)) == "native"  # same instance, per-call bounce
    assert op(torch.zeros(8)) == "jit"


# --- torch.compile protocol -----------------------------------------------------


@pytest.mark.parametrize("op_cls", [_OotTensorOp, _KeepOotDispatch])
@pytest.mark.parametrize("warmed", [False, True])
def test_oot_compile_registration_roundtrip(monkeypatch, op_cls, warmed):
    """A registered compile function wins over the subclass selection method.

    Leaving compile mode restores the previous eager function, including an
    instance whose eager function had not yet been selected.
    """
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(op_cls, forward_myplat, "myplat")
    BaseFusedOp.register_oot_compile_forward(op_cls, forward_myplat_compile, "myplat")
    op = op_cls()
    x = torch.tensor([1.0, 2.0, 3.0])
    if warmed:
        torch.testing.assert_close(op(x), x + 1)
    for _ in range(2):
        original = op._forward_method
        op.enter_torch_compile(num_tokens=1)
        assert op.is_torch_compile
        torch.testing.assert_close(op(x, scale=2), x * 2 + 2)
        op.enter_torch_compile(num_tokens=16)
        torch.testing.assert_close(op(x, scale=2), x * 2 + 2)
        op.leave_torch_compile()
        assert not op.is_torch_compile
        assert op._forward_method is original
        op.leave_torch_compile()
        torch.testing.assert_close(op(x), x + 1)


def test_oot_compile_registration_is_exact_type(monkeypatch):
    class _Sub(_OotTensorOp):
        pass

    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_OotTensorOp, forward_myplat, "myplat")
    BaseFusedOp.register_oot_compile_forward(
        _OotTensorOp, forward_myplat_compile, "myplat"
    )
    x = torch.tensor([1.0, 2.0])
    parent, child = _OotTensorOp(), _Sub()
    parent.enter_torch_compile(num_tokens=1)
    child.enter_torch_compile(num_tokens=1)
    torch.testing.assert_close(parent(x), x + 2)
    torch.testing.assert_close(child(x), x - 1)
    child.leave_torch_compile()
    # Compile-only registration is independent of eager registration.
    _Sub.register_oot_compile_forward(_Sub, forward_myplat_compile, "myplat")
    child.enter_torch_compile(num_tokens=1)
    torch.testing.assert_close(child(x), x + 2)
    child.leave_torch_compile()
    torch.testing.assert_close(child(x), x - 1)


def test_in_tree_compile_ignores_oot_registry(monkeypatch):
    _mock_platform(monkeypatch, key="cuda")
    BaseFusedOp.register_oot_compile_forward(
        _AllPlatformsOp, lambda self, x: "oot-compile", "cuda"
    )
    op = _AllPlatformsOp()
    x = torch.zeros(1)
    assert op(x) == "cuda"
    op.enter_torch_compile(num_tokens=1)
    assert op(x) == "native"
    op.leave_torch_compile()
    assert op(x) == "cuda"


def test_oot_compile_registration_requires_active_key(monkeypatch):
    _mock_platform(monkeypatch, oot_key="other")
    BaseFusedOp.register_oot_compile_forward(
        _OotTensorOp, forward_myplat_compile, "myplat"
    )
    op = _OotTensorOp()
    x = torch.tensor([1.0, 2.0])
    op.enter_torch_compile(num_tokens=1)
    torch.testing.assert_close(op(x), x - 1)
    op.leave_torch_compile()
    torch.testing.assert_close(op(x), x - 1)


@pytest.mark.parametrize("warmed", [False, True])
def test_oot_eager_registration_preserves_compile_fallback(monkeypatch, warmed):
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_OotTensorOp, forward_myplat, "myplat")
    op = _OotTensorOp()
    x = torch.tensor([1.0, 2.0])
    if warmed:
        torch.testing.assert_close(op(x), x + 1)
    op.enter_torch_compile(num_tokens=1)
    torch.testing.assert_close(op(x), x - 1)
    op.leave_torch_compile()
    torch.testing.assert_close(op(x), x + 1)


def test_oot_compile_respects_explicit_and_forced_backends(monkeypatch):
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_OotTensorOp, forward_myplat, "myplat")
    BaseFusedOp.register_oot_compile_forward(
        _OotTensorOp, forward_myplat_compile, "myplat"
    )
    op = _OotTensorOp()
    x = torch.tensor([1.0, 2.0])
    op.enter_torch_compile(num_tokens=1)
    torch.testing.assert_close(op(x), x + 2)
    torch.testing.assert_close(op(x, backend=KernelBackend.TORCH), x - 1)
    fo.set_fused_op_backend(KernelBackend.TORCH)
    torch.testing.assert_close(op(x), x - 1)
    fo.set_fused_op_backend(None)
    op.leave_torch_compile()
    torch.testing.assert_close(op(x), x + 1)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("caching_precompile", [False, True])
@pytest.mark.parametrize("warmed", [False, True])
def test_oot_outer_torch_compile(
    monkeypatch, _compiler_state, backend, fullgraph, caching_precompile, warmed
):
    """Compile registered functions that are not methods on the op class.

    Also cover saving compiler guards after eager execution has selected a
    function that must be restored when compilation finishes.
    """
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_KeepOotDispatch, forward_myplat, "myplat")
    BaseFusedOp.register_oot_compile_forward(
        _KeepOotDispatch, forward_myplat_compile, "myplat"
    )
    op = _KeepOotDispatch()
    assert not hasattr(op, "forward_myplat")
    assert not hasattr(op, "forward_myplat_compile")
    x = torch.tensor([1.0, 2.0, 3.0])
    if warmed:
        torch.testing.assert_close(op(x), x + 1)
    original = op._forward_method
    op.enter_torch_compile(num_tokens=1)
    try:
        with torch._dynamo.config.patch(caching_precompile=caching_precompile):
            compiled = torch.compile(
                _OotModel(op).forward, backend=backend, fullgraph=fullgraph
            )
            for values in (x, x + 10):
                torch.testing.assert_close(compiled(values), (values * 2 + 2) * 3)
    finally:
        op.leave_torch_compile()
    assert op._forward_method is original
    torch.testing.assert_close(op(x), x + 1)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("caching_precompile", [False, True])
def test_oot_compile_with_outer_graph_break(
    monkeypatch, _compiler_state, backend, caching_precompile
):
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_OotTensorOp, forward_myplat, "myplat")
    BaseFusedOp.register_oot_compile_forward(
        _OotTensorOp, forward_myplat_compile, "myplat"
    )
    op = _OotTensorOp()
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(op(x), x + 1)
    op.enter_torch_compile(num_tokens=1)
    try:
        with torch._dynamo.config.patch(caching_precompile=caching_precompile):
            compiled = torch.compile(
                _OotModelWithBreak(op).forward, backend=backend, fullgraph=False
            )
            for values in (x, x + 10):
                torch.testing.assert_close(compiled(values), (values * 2 + 2) * 3)
    finally:
        op.leave_torch_compile()
    torch.testing.assert_close(op(x), x + 1)


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("caching_precompile", [False, True])
def test_oot_compile_registration_preserves_tracing_errors(
    monkeypatch, _compiler_state, fullgraph, caching_precompile
):
    """A graph break inside the registered function still fails with fullgraph=True."""
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_OotTensorOp, forward_myplat, "myplat")
    BaseFusedOp.register_oot_compile_forward(
        _OotTensorOp, forward_myplat_graph_break, "myplat"
    )
    op = _OotTensorOp()
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(op(x), x + 1)
    original = op._forward_method
    op.enter_torch_compile(num_tokens=1)
    try:
        with torch._dynamo.config.patch(caching_precompile=caching_precompile):
            compiled = torch.compile(
                _OotModel(op).forward, backend="eager", fullgraph=fullgraph
            )
            if fullgraph:
                with pytest.raises(torch._dynamo.exc.Unsupported, match="graph_break"):
                    compiled(x)
            else:
                torch.testing.assert_close(compiled(x), (x * 2 + 2) * 3)
    finally:
        op.leave_torch_compile()
    assert op._forward_method is original
    torch.testing.assert_close(op(x), x + 1)


@pytest.mark.parametrize("warmed", [False, True])
def test_oot_eager_forward_survives_guard_serialization(
    monkeypatch, _compiler_state, warmed
):
    """Saving compiler guards must work when compilation uses the eager function.

    A function registered outside the op class may have no matching attribute
    on that class. Binding it directly to the op makes guard serialization
    fail when Dynamo looks it up by name. Enable precompile caching to check
    this lookup, which ordinary backend="eager" tracing does not exercise.
    """
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_KeepOotDispatch, forward_myplat, "myplat")
    op = _KeepOotDispatch()
    x = torch.tensor([1.0, 2.0, 3.0])
    if warmed:
        torch.testing.assert_close(op(x), x + 1)
    op.enter_torch_compile(num_tokens=8)
    try:
        with torch._dynamo.config.patch(caching_precompile=True):
            compiled = torch.compile(
                _OotModel(op).forward, backend="eager", fullgraph=True
            )
            torch.testing.assert_close(compiled(x), (x * 2 + 1) * 3)
    finally:
        op.leave_torch_compile()
    torch.testing.assert_close(op(x), x + 1)


def test_enter_leave_torch_compile_roundtrip(monkeypatch):
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    op = _CudaOnlyPlatformOp()
    assert op(torch.zeros(1)) == "cuda"

    op.enter_torch_compile(num_tokens=16)
    assert op.is_torch_compile
    assert op(torch.zeros(1)) == "native"

    # Reused-module idempotency: a second enter must not overwrite the saved
    # original forward, otherwise leave() cannot restore it.
    op.enter_torch_compile(num_tokens=16)
    op.leave_torch_compile()
    assert not op.is_torch_compile
    assert op(torch.zeros(1)) == "cuda"
    op.leave_torch_compile()  # double leave is a no-op
    assert op(torch.zeros(1)) == "cuda"


def test_torch_compile_hook_none_keeps_dispatch(monkeypatch):
    class _KeepOptimized(_CudaOnlyPlatformOp):
        def _torch_compile_forward(self, num_tokens):
            return None if num_tokens > 1 else self.forward_native

    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    op = _KeepOptimized()
    op.enter_torch_compile(num_tokens=8)
    assert op.is_torch_compile
    assert op(torch.zeros(1)) == "cuda"  # dispatch unchanged for bs > 1
    op.leave_torch_compile()

    op.enter_torch_compile(num_tokens=1)
    assert op(torch.zeros(1)) == "native"
    op.leave_torch_compile()


def test_topk_compile_hook_is_bs1_only():
    from sglang.srt.layers.moe.topk import TopK

    class _Probe:
        forward_native = "native-sentinel"

    assert TopK._torch_compile_forward(_Probe(), num_tokens=1) == "native-sentinel"
    assert TopK._torch_compile_forward(_Probe(), num_tokens=2) is None


def test_fused_moe_compile_hook_is_bs1_only():
    from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    probe = object.__new__(UnquantizedFusedMoEMethod)
    assert (
        UnquantizedFusedMoEMethod._torch_compile_forward(probe, num_tokens=1)
        is fused_moe_forward_native
    )
    assert UnquantizedFusedMoEMethod._torch_compile_forward(probe, num_tokens=2) is None


# --- tracing --------------------------------------------------------------------


def test_oot_registered_functions_preserve_trace_labels(monkeypatch):
    _mock_platform(monkeypatch, oot_key="myplat")
    BaseFusedOp.register_oot_forward(_OotTensorOp, forward_myplat, "myplat")
    BaseFusedOp.register_oot_compile_forward(
        _OotTensorOp, forward_myplat_compile, "myplat"
    )
    op = _OotTensorOp()
    x = torch.zeros(2, 3)
    fo.enable_fused_op_trace()
    op(x)
    op.enter_torch_compile(num_tokens=1)
    op(x)
    op.leave_torch_compile()
    op(x)
    records = fo.get_fused_op_trace()
    assert [rec.backend for rec in records] == ["myplat", "myplat_compile", "myplat"]
    assert all(rec.op == op.op for rec in records)
    assert all(rec.tensor_args == ("torch.float32[2, 3]",) for rec in records)


def test_trace_labels_explicit_backend(monkeypatch):
    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    op = _CudaOnlyPlatformOp()
    fo.enable_fused_op_trace()
    op(torch.zeros(2, 3))
    op(torch.zeros(2, 3), backend=KernelBackend.TORCH)
    _, explicit_rec = fo.get_fused_op_trace()
    assert explicit_rec.backend == "torch"


# --- deprecated MultiPlatformOp alias --------------------------------------------


def test_deprecated_alias_contract(monkeypatch):
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    assert issubclass(MultiPlatformOp, BaseFusedOp)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        class _LegacyOp(MultiPlatformOp):
            # Old-style subclass: platform forwards only, no forward_native.
            def forward_cuda(self, x):
                return "cuda"

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    op = _LegacyOp()  # instantiable without forward_native (lenient alias)
    assert op(torch.zeros(1)) == "cuda"
    with pytest.raises(NotImplementedError):
        op.forward_native(torch.zeros(1))

    # register_oot_forward via the alias lands in the shared registry.
    MultiPlatformOp.register_oot_forward(_LegacyOp, lambda self, x: "oot", "aliasplat")
    _mock_platform(monkeypatch, key="", oot_key="aliasplat")
    assert _LegacyOp()(torch.zeros(1)) == "oot"


def test_deprecated_alias_keeps_legacy_platform_defaults(monkeypatch):
    """Old MultiPlatformOp defined per-platform default methods (hip/musa ->
    cuda, npu/xpu/cpu -> native); plugin code may call them directly, and a
    subclass without forward_cuda must still raise on CUDA like before."""
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        class _NativeOnlyLegacy(MultiPlatformOp):
            def forward_native(self, x):
                return "native"

    op = _NativeOnlyLegacy()
    assert op.forward_cpu(torch.zeros(1)) == "native"
    assert op.forward_npu(torch.zeros(1)) == "native"
    with pytest.raises(NotImplementedError):
        op.forward_hip(torch.zeros(1))  # chains to the raising forward_cuda

    _mock_platform(monkeypatch, key="cuda", info=_CUDA)
    with pytest.raises(NotImplementedError):
        _NativeOnlyLegacy()(torch.zeros(1))  # old CUDA behavior preserved


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
