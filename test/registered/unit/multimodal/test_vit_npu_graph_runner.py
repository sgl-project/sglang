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


def test_npu_vit_graph_keys_include_attention_boundaries():
    module = _load_npu_graph_runner_module()
    runner_cls = module.ViTNpuGraphRunner
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

    # patch.dict dropped the module from sys.modules, so patch it by object.
    with patch.object(target=module, attribute="set_graph_pool_id"):
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


def test_capture_and_replay_do_not_overwrite_inputs_or_retained_embeddings():
    """In-place vision blocks and later replays must not change cached outputs."""
    module = _load_npu_graph_runner_module()
    vit = SimpleNamespace(
        blocks=[_Block()],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    runner = module.ViTNpuGraphRunner(vit)

    class Graph:
        def __init__(self, key):
            self.key = key

        def replay(self):
            runner.block_output[self.key].copy_(runner.block_input[self.key] + 10)

    # Model capture's in-place update and the reusable graph output buffer.
    def capture(graph_key):
        runner.block_input[graph_key].add_(10)
        runner.block_output[graph_key] = runner.block_input[graph_key].clone()
        runner.block_graphs[graph_key] = Graph(graph_key)

    runner._create_graph = capture
    rotary = torch.zeros(8, 4)
    first = torch.ones(8, 8)
    first_layout = torch.tensor([0, 4, 8], dtype=torch.int32)
    second_layout = torch.tensor([0, 2, 8], dtype=torch.int32)
    with (
        patch.object(module, "set_graph_pool_id"),
        patch.object(module.ViTNpuGraphRunner, "_graph_memory_pool", object()),
    ):
        out = runner.run(first, first_layout, rotary, rotary)
        retained = out[:4]  # Same ownership pattern as embedding-cache splits.
        torch.testing.assert_close(first, torch.ones_like(first))
        torch.testing.assert_close(out, torch.full_like(out, 11))
        later = runner.run(torch.full_like(first, 2), first_layout, rotary, rotary)
        runner.run(torch.full_like(first, 3), second_layout, rotary, rotary)
    torch.testing.assert_close(retained, torch.full_like(retained, 11))
    torch.testing.assert_close(later, torch.full_like(later, 12))


def test_uncached_layout_fallback_preserves_deepstack_and_cached_graphs():
    """New layouts at capacity must compute every output without growing graphs."""
    module = _load_npu_graph_runner_module()

    class Block(_Block):
        def __call__(self, x, **kwargs):
            # Ascend attention requires an output workspace even when graph
            # capture is skipped, since the graph option remains enabled.
            assert kwargs["output_ws"].shape == (x.shape[0], 2, 4)
            torch.testing.assert_close(
                kwargs["cu_seqlens"], torch.tensor([2, 4], dtype=torch.int32)
            )
            return x + 1

    vit = SimpleNamespace(
        blocks=[Block()],
        merger=lambda x: x.squeeze(1) * 2,
        deepstack_visual_indexes=[0],
        deepstack_merger_list=[lambda x: x.squeeze(1) * 3],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    runner = module.ViTNpuGraphRunner(vit)
    # No real graphs needed: reaching the limit must use the actual eager
    # block/merger computation, including deepstack and output permutation.
    runner._MAX_CACHED_GRAPHS = 1
    existing = object()
    runner.block_graphs["existing"] = existing
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    rotary = torch.zeros(4, 4)
    indices = torch.tensor([3, 1, 2, 0])
    out = runner.run(
        x, torch.tensor([0, 2, 4]), rotary, rotary, output_indices=indices
    )
    expected = torch.cat([(x + 1) * 2, (x + 1) * 3], dim=1)[indices]
    torch.testing.assert_close(out, expected)
    assert runner.block_graphs == {"existing": existing}
    assert not runner.block_input


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
