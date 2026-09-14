import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopK


def _bare_topk() -> TopK:
    """Build only the dispatch surface; TopK.__init__ needs runtime context."""
    topk = object.__new__(TopK)
    torch.nn.Module.__init__(topk)
    return topk


def test_torch_compile_keeps_npu_topk_dispatch(monkeypatch):
    topk = _bare_topk()
    monkeypatch.setattr(topk_module, "_is_npu", True)

    assert topk._torch_compile_forward(num_tokens=1) is None
    assert topk._torch_compile_forward(num_tokens=2) is None


def test_torch_compile_keeps_existing_non_npu_policy(monkeypatch):
    topk = _bare_topk()
    monkeypatch.setattr(topk_module, "_is_npu", False)

    assert topk._torch_compile_forward(num_tokens=1) == topk.forward_native
    assert topk._torch_compile_forward(num_tokens=2) is None
