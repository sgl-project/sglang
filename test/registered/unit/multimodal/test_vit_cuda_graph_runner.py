import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.srt.multimodal.internvl_vit_cuda_graph_runner import (
    InternViTCudaGraphRunner,
)
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.runtime_context import get_parallel


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
    # No tp_group is stated, so reading one would raise: the DP path must not
    # ask for the TP group at all.
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
    with get_parallel().override(tp_group=group):
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


def test_vit_graph_keeps_rotary_workspace_address_after_growth():
    runner = _runner(use_data_parallel=True)
    runner.vit.device = torch.device("cpu")
    runner.vit.dtype = torch.float32

    small = runner._get_sin_cos_ws("small", seq_len=4, head_dim=2, dtype=torch.float32)
    small_address = small[0].data_ptr()
    runner._get_sin_cos_ws("large", seq_len=16, head_dim=2, dtype=torch.float32)

    assert runner._get_sin_cos_ws("small", seq_len=4, head_dim=2, dtype=torch.float32)[
        0
    ].data_ptr() == (small_address)
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


def test_vit_graph_rotary_workspace_keeps_the_model_cos_sin_dtype():
    """The graph must rotate with the cos/sin values eager mode uses: fp32
    rotary inputs must not be rounded to the ViT dtype on their way in."""
    runner = _runner(use_data_parallel=True)
    runner.vit.device = torch.device("cpu")
    runner.vit.dtype = torch.bfloat16
    runner.block_input["key"] = torch.empty(4, 1, 2, dtype=torch.bfloat16)
    runner.block_output["key"] = torch.empty(4, 1, 2, dtype=torch.bfloat16)
    runner.block_graphs["key"] = SimpleNamespace(replay=lambda: None)
    # 1 + 2**-10 is not representable in bf16.
    cos = torch.full((4, 2), 1 + 2**-10, dtype=torch.float32)
    sin = torch.full((4, 2), 2**-10, dtype=torch.float32)

    runner.replay(
        graph_key="key",
        x_3d=torch.ones(4, 1, 2, dtype=torch.bfloat16),
        rotary_pos_emb_cos=cos,
        rotary_pos_emb_sin=sin,
    )

    used_cos, used_sin = runner._sin_cos_ws_by_graph["key"]
    assert torch.equal(used_cos, cos) and torch.equal(used_sin, sin)


class _RecordingBlock:
    attn = SimpleNamespace(
        qkv_backend_name="aiter_attn",
        num_attention_heads_per_partition=1,
        head_size=2,
    )

    def __init__(self):
        self.cu_seqlens_calls = []

    def forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, output_ws):
        self.cu_seqlens_calls.append(cu_seqlens)
        return x + 1

    __call__ = forward


def _capacity_runner(capacity):
    block = _RecordingBlock()
    vit = SimpleNamespace(
        blocks=[block],
        merger=lambda y: y * 2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_data_parallel=True,
    )
    with envs.SGLANG_VIT_CUDA_GRAPH_CACHE_CAPACITY.override(capacity):
        runner = ViTCudaGraphRunner(vit)
    runner.create_graph = lambda **kwargs: pytest.fail("captured past capacity")
    return runner, block


def _run(runner, cu_seqlens):
    return runner.run(
        x=torch.zeros(6, 2),
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
        cu_window_seqlens=None,
        position_embeddings=None,
        rotary_pos_emb_cos=torch.ones(6, 2),
        rotary_pos_emb_sin=torch.zeros(6, 2),
    )


def test_vit_graph_cache_capacity_runs_new_layouts_without_capturing():
    """Past the capacity a new layout must neither capture nor allocate per-key
    buffers, and must run the blocks with the arguments a graph would use."""
    runner, block = _capacity_runner(capacity=1)
    runner.block_graphs["captured"] = object()

    out = _run(runner, [0, 4, 6])

    assert torch.equal(out, torch.full((6, 1, 2), 2.0))
    assert list(runner.block_graphs) == ["captured"]
    assert not runner.block_input and not runner.block_ws
    assert not runner._sin_cos_ws_by_graph
    ((cu_seqlens, max_len),) = block.cu_seqlens_calls
    assert cu_seqlens.tolist() == [0, 4, 6] and max_len == 4


def test_vit_graph_cache_capacity_still_replays_captured_layouts():
    runner, block = _capacity_runner(capacity=1)
    key = runner._get_graph_key(torch.empty(6, 1, 2), torch.tensor([0, 4, 6]), None)
    runner.block_input[key] = torch.empty(6, 1, 2)
    runner.block_output[key] = torch.full((6, 1, 2), 7.0)
    runner.block_graphs[key] = SimpleNamespace(replay=lambda: None)

    out = _run(runner, [0, 4, 6])

    assert torch.equal(out, torch.full((6, 1, 2), 7.0))
    assert not block.cu_seqlens_calls


def test_vit_graph_cache_capacity_defaults_to_32():
    assert envs.SGLANG_VIT_CUDA_GRAPH_CACHE_CAPACITY.get() == 32
    runner, _ = _capacity_runner(capacity=32)
    captured = []
    runner.create_graph = lambda **kwargs: captured.append(kwargs)
    runner.replay = lambda **kwargs: None
    for i in range(31):
        runner.block_graphs[f"other{i}"] = object()

    _run(runner, [0, 4, 6])

    assert len(captured) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
