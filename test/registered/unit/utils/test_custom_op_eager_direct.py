"""``register_custom_op`` ops skip the dispatcher in plain eager execution.

Only plain device tensors under no_grad take the direct call. Tracing, dispatch and
function modes, the profiler, autograd, tensor subclasses, host tensors and a name
registered by another function first all keep the registered op.
"""

import sys

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.utils.custom_op import register_custom_op
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


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


def _operands(device="cuda"):
    return torch.randn(8, device=device), torch.randn(8, device=device)


def _took_direct_call():
    return calls[-1] == ("eager_direct_add", "call")


@torch.no_grad()
def test_eager_call_skips_the_dispatcher():
    x, y = _operands()
    assert torch.equal(eager_direct_add(x, y), x + y)
    assert _took_direct_call()


class _SeenOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


@torch.no_grad()
def test_dispatch_mode_sees_the_registered_op():
    x, y = _operands()
    with _SeenOps() as mode:
        assert torch.equal(eager_direct_add(x, y), x + y)
    assert "sglang.test_eager_direct_add.default" in mode.ops


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


def test_compile_traces_the_registered_op():
    x, y = _operands()
    compiled = torch.compile(
        lambda a, b: eager_direct_add(a, b) * 2, fullgraph=True, backend="eager"
    )
    assert torch.equal(compiled(x, y), (x + y) * 2)
