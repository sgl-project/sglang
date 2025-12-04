import unittest

import torch
from torch import nn

from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import add_prefix

TEST_HIDDEN_SIZE = 32


class SimpleModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = TEST_HIDDEN_SIZE
        self.rms_norm_eps = 1e-5
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.hidden_size,
            hidden_act="silu",
            quant_config=None,
            prefix=add_prefix("mlp", ""),
        )
        self.layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class MockCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleModel()

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


def init_weights(module):
    if isinstance(module, LinearBase):
        torch.nn.init.uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, RMSNorm):
        torch.nn.init.ones_(module.weight)


def test_model_forward_dump(tmp_path):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:2646",
    )
    initialize_model_parallel()
    model = MockCausalLM()
    model.apply(init_weights)
    model = model.cuda().bfloat16()
    dumper = register_forward_hook_for_model(
        model, tmp_path / "sglang_dump", [0], 0, 0, 0
    )

    dir_path = dumper.get_dump_dir()
    inp = torch.randn(4, TEST_HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01
    result = model(inp.cuda())
    data = torch.load(f"{dir_path}/Pass00000.pt")
    assert "model.layernorm" in data
    assert "model.mlp.down_proj" in data
    assert torch.allclose(
        data["model.mlp.down_proj"], result.cpu(), rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
    unittest.main()
