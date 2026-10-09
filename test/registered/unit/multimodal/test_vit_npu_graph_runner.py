import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Block:
    attn = SimpleNamespace(
        qkv_backend_name="ascend_attn",
        num_attention_heads_per_partition=2,
        head_size=4,
    )

    def forward(self, x, output_ws=None):
        return x


class _FakeGraph:
    def replay(self):
        pass


def _load_npu_graph_runner():
    # Load shared modules before stubbing torch_npu so platform detection stays on CPU.
    importlib.import_module("sglang.srt.multimodal.vit_cuda_graph_runner")
    torch_npu = SimpleNamespace()
    with patch.dict(
        sys.modules,
        {
            "torch_npu": torch_npu,
        },
    ):
        module = importlib.import_module(
            "sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner"
        )
    return module.ViTNpuGraphRunner


def test_npu_vit_graph_keys_include_attention_boundaries():
    runner_cls = _load_npu_graph_runner()
    vit = SimpleNamespace(
        blocks=[_Block()],
        merger=lambda x: x,
        device=torch.device("cpu"),
        dtype=torch.float32,
        deepstack_visual_indexes=[],
        deepstack_merger_list=None,
    )

    with patch(
        "torch.get_device_module",
        return_value=SimpleNamespace(graph_pool_handle=lambda: object()),
    ):
        runner = runner_cls(vit)

    runner_cls._graph_memory_pool = None
    runner._create_graph = lambda graph_key: runner.block_graphs.__setitem__(
        graph_key, _FakeGraph()
    )

    x = torch.zeros(8, 8)
    rotary = torch.zeros(8, 4)
    first_layout = torch.tensor([0, 4, 8], dtype=torch.int32)
    second_layout = torch.tensor([0, 2, 8], dtype=torch.int32)

    with patch(
        "sglang.srt.hardware_backend.npu.graph_runner."
        "vit_npu_graph_runner.set_graph_pool_id"
    ):
        runner.run(
            x,
            first_layout,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )
        runner.run(
            x,
            second_layout,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )

    assert len(runner.block_graphs) == 2
    assert {key[1][0] for key in runner.block_graphs} == {
        (0, 4, 8),
        (0, 2, 8),
    }
    assert all(
        workspace.shape[0] == x.shape[0] for workspace in runner.block_ws.values()
    )


def test_npu_vit_graph_replay_output_survives_the_next_replay():
    """Callers cache a replay's result and may replay again before consuming it,
    so replay must not hand out the graph's static output buffer."""
    runner_cls = _load_npu_graph_runner()
    vit = SimpleNamespace(
        blocks=[_Block()],
        merger=lambda x: x,
        device=torch.device("cpu"),
        dtype=torch.float32,
        deepstack_visual_indexes=[],
        deepstack_merger_list=None,
    )
    with patch(
        "torch.get_device_module",
        return_value=SimpleNamespace(graph_pool_handle=lambda: object()),
    ):
        runner = runner_cls(vit)
    runner.block_input["key"] = torch.empty(4, 1, 2)
    runner.block_output["key"] = torch.empty(4, 1, 2)
    runner.block_graphs["key"] = SimpleNamespace(
        replay=lambda: runner.block_output["key"].copy_(runner.block_input["key"] * 2)
    )

    first = runner.replay(graph_key="key", x_3d=torch.ones(4, 1, 2))
    runner.replay(graph_key="key", x_3d=torch.full((4, 1, 2), 3.0))

    assert torch.equal(first, torch.full((4, 1, 2), 2.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
