import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.multimodal.internvl_vit_cuda_graph_runner import (
    InternViTCudaGraphRunner,
)
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Block:
    def forward(self, x):
        return x


def _runner(*, use_data_parallel: bool) -> ViTCudaGraphRunner:
    vit = SimpleNamespace(
        blocks=[_Block()],
        deepstack_visual_indexes=[],
        deepstack_merger_list=None,
        use_data_parallel=use_data_parallel,
    )
    return ViTCudaGraphRunner(vit)


def test_dp_vit_graph_capture_does_not_enter_tp_communication_capture():
    runner = _runner(use_data_parallel=True)
    with patch(
        "sglang.srt.multimodal.vit_cuda_graph_runner.get_tp_group",
        side_effect=AssertionError("DP capture must be rank-local"),
    ):
        with runner._capture_context():
            pass


def test_non_dp_vit_graph_capture_uses_tp_communication_capture():
    entered = []

    class Capture:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *args):
            return False

    group = SimpleNamespace(ca_comm=SimpleNamespace(capture=lambda: Capture()))
    runner = _runner(use_data_parallel=False)
    with patch(
        "sglang.srt.multimodal.vit_cuda_graph_runner.get_tp_group", return_value=group
    ):
        with runner._capture_context():
            pass
    assert entered == [True]


def test_vit_graph_runner_caches_resolved_backend_name():
    class Block:
        attn = SimpleNamespace(
            qkv_backend_name="fa3",
            qkv_backend=object(),
        )

        def forward(self, x, output_ws=None):
            return x

    vit = SimpleNamespace(blocks=[Block()])

    runner = ViTCudaGraphRunner(vit)

    assert runner._attn_backend == "fa3"


def test_vit_graph_key_includes_full_and_window_attention_boundaries():
    runner = _runner(use_data_parallel=True)
    x = torch.empty(8, 1, 4)

    first = runner._get_graph_key(
        x,
        torch.tensor([0, 4, 8]),
        torch.tensor([0, 2, 4, 8]),
    )
    second = runner._get_graph_key(
        x,
        torch.tensor([0, 2, 8]),
        torch.tensor([0, 4, 6, 8]),
    )

    assert first != second


def test_npu_graph_capture_and_replay_keep_sequence_boundaries():
    path = (
        Path(sys.modules[ViTCudaGraphRunner.__module__].__file__).parents[1]
        / "hardware_backend/npu/graph_runner/vit_npu_graph_runner.py"
    )
    spec = importlib.util.spec_from_file_location("npu_graph_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}):
        spec.loader.exec_module(module)

    block = _Block()
    block.attn = SimpleNamespace(num_attention_heads_per_partition=2, head_size=4)
    runner = module.ViTNpuGraphRunner(
        SimpleNamespace(blocks=[block], device=torch.device("cpu"), dtype=torch.float32)
    )
    runner.device_module = SimpleNamespace(graph_pool_handle=lambda: object())
    captured = []

    def capture(graph_key):
        captured.append(graph_key)
        runner.block_graphs[graph_key] = SimpleNamespace(replay=lambda: None)

    x = torch.ones(8, 8)
    cos, sin = torch.ones(8, 4), torch.zeros(8, 4)
    with (
        patch.object(module, "set_graph_pool_id"),
        patch.object(runner, "_create_graph", side_effect=capture),
    ):
        for boundaries in ([0, 4, 8], [0, 2, 8], [0, 4, 8]):
            result = runner.run(x, torch.tensor(boundaries), cos, sin)
            torch.testing.assert_close(result.squeeze(1), x)

    assert len(captured) == 2
    assert runner.block_ws[captured[0]].shape == (8, 2, 4)
    assert runner.cu_seq_lens[captured[0]].tolist() == [4, 8]
    assert runner.cu_seq_lens[captured[1]].tolist() == [2, 8]


def test_vit_graph_keeps_rotary_workspace_address_after_growth():
    runner = _runner(use_data_parallel=True)
    runner.vit.device = torch.device("cpu")
    runner.vit.dtype = torch.float32

    small = runner._get_sin_cos_ws("small", seq_len=4, head_dim=2)
    small_address = small[0].data_ptr()
    runner._get_sin_cos_ws("large", seq_len=16, head_dim=2)

    assert runner._get_sin_cos_ws("small", seq_len=4, head_dim=2)[0].data_ptr() == (
        small_address
    )
    assert len(runner._retired_sin_cos_ws) == 1


def test_internvl_graph_runner_caches_resolved_backend_name():
    attention = SimpleNamespace(
        qkv_backend_name="triton_attn",
        qkv_backend=object(),
    )
    layer = SimpleNamespace(attn=SimpleNamespace(attn=attention))
    encoder = SimpleNamespace(layers=[layer])

    runner = InternViTCudaGraphRunner(encoder)

    assert runner._attn_backend == "triton_attn"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
