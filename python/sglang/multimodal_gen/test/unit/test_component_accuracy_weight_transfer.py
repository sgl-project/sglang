import torch
from torch import nn

from sglang.multimodal_gen.test.single_test_file.component_accuracy.engine import (
    AccuracyEngine,
)


class _SourceProjectionSet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(2, 6, bias=False)
        self.gate_proj = nn.Linear(2, 3, bias=False)
        self.up_proj = nn.Linear(2, 3, bias=False)
        self.down_proj = nn.Linear(3, 2, bias=False)


class _TargetProjectionSet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(2, 6, bias=False)
        self.gate_up_proj = nn.Linear(2, 8, bias=False)
        self.down_proj = nn.Linear(4, 2, bias=False)

        self.qkv_proj.weight.weight_loader = self._load_qkv
        self.gate_up_proj.weight.weight_loader = self._load_gate_up
        self.down_proj.weight.weight_loader = self._load_down

    @staticmethod
    def _load_qkv(param: nn.Parameter, source: torch.Tensor) -> None:
        param.data.copy_(source)

    @staticmethod
    def _load_gate_up(param: nn.Parameter, source: torch.Tensor, shard_id: int) -> None:
        offset = shard_id * 4
        param.data[offset : offset + source.shape[0]].copy_(source)

    @staticmethod
    def _load_down(param: nn.Parameter, source: torch.Tensor) -> None:
        param.data[:, : source.shape[1]].copy_(source)


def test_transfer_weights_uses_loaders_for_fused_aliases_and_padding() -> None:
    source = _SourceProjectionSet().to(dtype=torch.bfloat16)
    target = _TargetProjectionSet()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(index)
        for parameter in target.parameters():
            parameter.zero_()

    AccuracyEngine.transfer_weights(source, target, target_device=torch.device("cpu"))

    torch.testing.assert_close(target.qkv_proj.weight, source.qkv.weight)
    torch.testing.assert_close(target.gate_up_proj.weight[:3], source.gate_proj.weight)
    torch.testing.assert_close(target.gate_up_proj.weight[4:7], source.up_proj.weight)
    assert torch.count_nonzero(target.gate_up_proj.weight[[3, 7]]) == 0
    torch.testing.assert_close(target.down_proj.weight[:, :3], source.down_proj.weight)
    assert torch.count_nonzero(target.down_proj.weight[:, 3]) == 0
