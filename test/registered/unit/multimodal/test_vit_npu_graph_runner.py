import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

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
    torch_npu = SimpleNamespace()
    allocator = SimpleNamespace(set_graph_pool_id=lambda pool: None)
    with patch.dict(
        sys.modules,
        {
            "torch_npu": torch_npu,
            "sglang.srt.distributed.device_communicators.pynccl_allocator": allocator,
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
