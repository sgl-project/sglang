import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Block:
    attn = SimpleNamespace(
        qkv_backend_name="ascend_attn",
        num_attention_heads_per_partition=2,
        head_size=4,
    )

    def forward(self, x, output_ws=None):
        return x


def _load_npu_graph_runner_module():
    # Load shared modules before stubbing torch_npu so platform detection stays on CPU.
    importlib.import_module("sglang.srt.multimodal.vit_cuda_graph_runner")
    torch_npu = SimpleNamespace()
    with patch.dict(
        sys.modules,
        {
            "torch_npu": torch_npu,
        },
    ):
        return importlib.import_module(
            "sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner"
        )


def test_npu_vit_graph_keys_and_buffer_ownership():
    """Distinct layouts get distinct graphs; capture/replay preserve caller data."""
    module = _load_npu_graph_runner_module()
    runner = module.ViTNpuGraphRunner(
        SimpleNamespace(blocks=[_Block()], device=torch.device("cpu"), dtype=torch.float32)
    )

    def capture(key):
        # Simulate in-place capture and a static output reused by replay.
        runner.block_input[key].add_(10)
        runner.block_output[key] = runner.block_input[key].clone()
        runner.block_graphs[key] = SimpleNamespace(
            replay=lambda: runner.block_output[key].copy_(runner.block_input[key] + 10)
        )

    runner._create_graph = capture
    x = torch.ones(8, 8)
    rotary = torch.zeros(8, 4)
    first_layout = torch.tensor([0, 4, 8], dtype=torch.int32)
    second_layout = torch.tensor([0, 2, 8], dtype=torch.int32)
    with (
        patch.object(module, "set_graph_pool_id"),
        patch.object(module.ViTNpuGraphRunner, "_graph_memory_pool", object()),
    ):
        out = runner.run(x, first_layout, rotary, rotary)
        retained = out[:4]  # Embedding-cache splits retain views across calls.
        torch.testing.assert_close(x, torch.ones_like(x))
        torch.testing.assert_close(out, torch.full_like(out, 11))
        later = runner.run(torch.full_like(x, 2), first_layout, rotary, rotary)
        runner.run(torch.full_like(x, 3), second_layout, rotary, rotary)
    torch.testing.assert_close(retained, torch.full_like(retained, 11))
    torch.testing.assert_close(later, torch.full_like(later, 12))
    assert len(runner.block_graphs) == 2
    assert {key[1][0] for key in runner.block_graphs} == {(0, 4, 8), (0, 2, 8)}
    assert all(ws.shape[0] == x.shape[0] for ws in runner.block_ws.values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
