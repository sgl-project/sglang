"""``register_custom_op`` ops skip the dispatcher in plain eager execution.

So do ops from ``register_custom_op_from_extern`` and the multimodal_gen
``register_custom_op``. Only plain device tensors under no_grad take the direct call.
Tracing, dispatch and function modes, the profiler, autograd, tensor subclasses, host
tensors and a name registered by another function first all keep the registered op.
"""

import sys

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.multimodal_gen.runtime.layers.utils import (
    register_custom_op as register_diffusion_custom_op,
)
from sglang.srt.utils.custom_op import (
    register_custom_op,
    register_custom_op_from_extern,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

calls = []  # (function, its caller's frame name)


def _fake_add(x, y):
    return torch.empty_like(x)


@register_custom_op(
    op_name="test_eager_direct_add", mutates_args=[], fake_impl=_fake_add
)
def eager_direct_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    calls.append(("eager_direct_add", sys._getframe(1).f_code.co_name))
    return x + y


def _extern_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    calls.append(("eager_direct_add", sys._getframe(1).f_code.co_name))
    return x + y


OPS = {
    "register_custom_op": eager_direct_add,
    "register_custom_op_from_extern": register_custom_op_from_extern(
        _extern_add, op_name="test_eager_direct_extern_add", out_shape=0
    ),
    "multimodal_gen_register_custom_op": register_diffusion_custom_op(
        op_name="test_eager_direct_diffusion_add", fake_impl=_fake_add
    )(_extern_add),
}


def _operands(device="cuda"):
    return torch.randn(8, device=device), torch.randn(8, device=device)


def _took_direct_call():
    return calls[-1] == ("eager_direct_add", "call")


@pytest.mark.parametrize("registration", sorted(OPS))
@torch.no_grad()
def test_eager_call_skips_the_dispatcher(registration):
    x, y = _operands()
    assert torch.equal(OPS[registration](x, y), x + y)
    assert _took_direct_call()


class _SeenOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("registration", sorted(OPS))
@torch.no_grad()
def test_dispatch_mode_sees_the_registered_op(registration):
    x, y = _operands()
    with _SeenOps() as mode:
        assert torch.equal(OPS[registration](x, y), x + y)
    assert not _took_direct_call()
    assert any(op.startswith("sglang.test_eager_direct_") for op in mode.ops)


class _Passthrough(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class _Sub(torch.Tensor):
    pass


@pytest.mark.parametrize("case", ["grad", "function_mode", "subclass"])
def test_other_contexts_use_the_dispatcher(case):
    x, y = _operands()
    if case == "grad":
        out = eager_direct_add(x, y)
    else:
        with torch.no_grad():
            if case == "subclass":
                out = eager_direct_add(x.as_subclass(_Sub), y)
            else:
                with _Passthrough():
                    out = eager_direct_add(x, y)
    assert torch.equal(out, x + y)
    assert not _took_direct_call()


@torch.no_grad()
def test_profiler_records_the_op():
    x, y = _operands()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        eager_direct_add(x, y)
    assert any("test_eager_direct_add" in event.name for event in prof.events())


@torch.no_grad()
def test_host_tensors_are_still_rejected():
    with pytest.raises(NotImplementedError):
        eager_direct_add(*_operands("cpu"))


@torch.no_grad()
def test_a_name_registered_first_elsewhere_keeps_that_function():
    def other(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        calls.append(("other", sys._getframe(1).f_code.co_name))
        return x - y

    second = register_custom_op(
        op_name="test_eager_direct_add", mutates_args=[], fake_impl=_fake_add
    )(other)
    x, y = _operands()
    assert torch.equal(second(x, y), x + y)
    assert calls[-1][0] == "eager_direct_add"


@pytest.mark.parametrize(
    "op_name, native_name",
    [
        ("fused_norm_scale_shift", "try_fused_norm_scale_shift"),
        (
            "fused_scale_residual_norm_scale_shift",
            "try_fused_scale_residual_norm_scale_shift",
        ),
    ],
)
@torch.no_grad()
def test_cutedsl_norm_ops_skip_the_dispatcher(monkeypatch, op_name, native_name):
    cutedsl = pytest.importorskip(
        "sglang.kernels.ops.diffusion.norm.scale_residual_norm_cutedsl"
    )
    from sglang.kernels.kda_kernels import norm_scale_shift_jit

    callers = []

    def native(*args):
        # frame 1 is the op body; frame 2 is whatever called it
        callers.append(sys._getframe(2).f_code.co_name)
        out = torch.empty_like(args[0])
        return out if op_name == "fused_norm_scale_shift" else (out, out.clone())

    monkeypatch.setattr(norm_scale_shift_jit, native_name, native)
    x, scale = torch.randn(1, 4, 256, device="cuda"), torch.randn(1, 256, device="cuda")
    if op_name == "fused_norm_scale_shift":
        args = (x, None, None, scale, scale, "rms", 1e-6)
    else:
        args = (x, x.clone(), None, None, None, scale, scale, "rms", 1e-6)
    getattr(cutedsl, op_name)(*args)
    assert callers == ["call"]


def test_compile_traces_the_registered_op():
    x, y = _operands()
    compiled = torch.compile(
        lambda a, b: eager_direct_add(a, b) * 2, fullgraph=True, backend="eager"
    )
    assert torch.equal(compiled(x, y), (x + y) * 2)
