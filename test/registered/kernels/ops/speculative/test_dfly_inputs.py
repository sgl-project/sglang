import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.dfly import assert_dfly_plan_equal, dfly_plan_fixture

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("window,width", [(0, 1), (4, 5), (9, 128), (1024, 1)])
def test_dfly_plan_cuda_replay(physical, window, width):
    """Fixed addresses must observe current request/state tensors after an
    all-padding capture, bucket switches, permutations and repeated replay."""
    execution, source = dfly_plan_fixture(
        device="cuda",
        physical=physical,
        window=window,
        width=width,
        hidden_dtype=torch.bfloat16,
    )
    reference, _ = dfly_plan_fixture(
        device="cuda",
        physical=physical,
        window=window,
        width=width,
        hidden_dtype=torch.bfloat16,
    )
    reference._fuse_inputs = False
    staged = {name: value.clone() for name, value in source.items()}
    staged["live"].zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {}
    with torch.cuda.stream(stream):
        for bs in (2, 5):
            batch = {name: value[:bs] for name, value in staged.items()}
            execution.plan(**batch)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = execution.plan(**batch)
            graphs[bs] = graph, outputs
    torch.cuda.current_stream().wait_stream(stream)
    for _, outputs in graphs.values():
        assert not outputs.block.active.any()
        assert (outputs.block.write_slots == 0).all()
        assert (outputs.topk_indices[0] == -1).all()
    pointers = {
        name: value.data_ptr()
        for name, value in vars(execution).items()
        if isinstance(value, torch.Tensor)
    }
    for step, selected in enumerate(
        ([0, 1, 2, 3, 4], [4, 3], [1, 0, 4, 3, 2], [3, 3], [4, 0, 1, 2, 3])
    ):
        bs = len(selected)
        for name, value in staged.items():
            value[:bs].copy_(source[name][selected])
        staged["bonus_tokens"][:bs].add_(step)
        execution._state.hidden.add_(1)
        reference._state.hidden.copy_(execution._state.hidden)
        execution._state.topk.copy_(execution._state.topk.roll(1, dims=-1))
        reference._state.topk.copy_(execution._state.topk)
        execution._state.mrope_delta.add_(1)
        reference._state.mrope_delta.copy_(execution._state.mrope_delta)
        expected = reference.plan(
            **{name: value[:bs] for name, value in staged.items()}
        )
        graph, actual = graphs[bs]
        for _ in range(2):
            execution._draft_rows.fill_(12345)
            graph.replay()
            assert_dfly_plan_equal(actual, expected)
            assert actual.block.input_ids.data_ptr() == execution._plan_ids.data_ptr()
            assert (
                actual.topk_indices[0].data_ptr() == actual.topk_indices[-1].data_ptr()
            )
        assert pointers == {
            name: value.data_ptr()
            for name, value in vars(execution).items()
            if isinstance(value, torch.Tensor)
        }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
