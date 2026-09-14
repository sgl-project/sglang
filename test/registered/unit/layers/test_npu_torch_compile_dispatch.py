import torch

from sglang.srt.layers import activation as activation_module
from sglang.srt.layers import layernorm as layernorm_module
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm


def _bare_fused_op(cls):
    """Create only the dispatch surface; constructors need runtime context."""
    op = object.__new__(cls)
    torch.nn.Module.__init__(op)
    return op


def test_torch_compile_keeps_npu_rmsnorm_dispatch(monkeypatch):
    op = _bare_fused_op(RMSNorm)
    monkeypatch.setattr(layernorm_module, "_is_npu", True)

    assert op._torch_compile_forward(num_tokens=1) is None


def test_torch_compile_keeps_npu_swiglu_dispatch(monkeypatch):
    op = _bare_fused_op(SiluAndMul)
    monkeypatch.setattr(activation_module, "_is_npu", True)

    assert op._torch_compile_forward(num_tokens=1) is None


def test_torch_compile_keeps_non_npu_reference_policy(monkeypatch):
    rmsnorm = _bare_fused_op(RMSNorm)
    swiglu = _bare_fused_op(SiluAndMul)
    monkeypatch.setattr(layernorm_module, "_is_npu", False)
    monkeypatch.setattr(activation_module, "_is_npu", False)

    assert rmsnorm._torch_compile_forward(num_tokens=1) == rmsnorm.forward_native
    assert swiglu._torch_compile_forward(num_tokens=1) == swiglu.forward_native
