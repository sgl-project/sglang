"""The output-only serving API must retain native validation and skip conversions."""

import sys

import pytest
import torch
from sgl_kernel.flash_mla import (
    flash_mla_packed_sparse_fwd,
    flash_mla_packed_sparse_output,
    flash_mla_sparse_fwd,
)

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.fixture(autouse=True)
def require_packed_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    properties = torch.cuda.get_device_properties(0)
    if (
        properties.major,
        properties.minor,
        properties.multi_processor_count,
    ) not in ((10, 0, 148), (10, 3, 148)):
        pytest.skip("Packed prefill is qualified on 148-SM B200 and B300 GPUs")
    if not hasattr(torch.ops.sgl_kernel, "packed_sparse_prefill_output"):
        pytest.fail("Blackwell test requires the FlashMLA packed prefill API")


def inputs():
    return [
        torch.randn(513, 16, 512, device="cuda", dtype=torch.bfloat16),
        torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16),
        torch.zeros(513, 128, device="cuda", dtype=torch.int32),
        torch.full((513,), 128, device="cuda", dtype=torch.int32),
        512**-0.5,
        torch.zeros(16, device="cuda"),
    ]


@pytest.mark.parametrize(
    "case",
    ["dtype", "stride", "empty_kv", "indices", "lengths", "sink", "device"],
)
def test_invalid_inputs(case):
    args = inputs()
    if case == "dtype":
        args[0] = args[0].float()
    elif case == "stride":
        args[0] = torch.empty(513, 16, 1024, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif case == "empty_kv":
        args[1] = args[1][:0]
    elif case == "indices":
        args[2] = args[2].long()
    elif case == "lengths":
        args[3] = args[3][:-1]
    elif case == "sink":
        args[5] = args[5].bfloat16()
    elif case == "device":
        args[1] = args[1].cpu()
    with pytest.raises(RuntimeError):
        flash_mla_packed_sparse_output(*args)


def test_no_statistics_conversion():
    args = inputs()
    flash_mla_packed_sparse_output(*args)
    torch.cuda.synchronize()
    counts = {}
    for name, fn in [
        ("full", flash_mla_packed_sparse_fwd),
        ("output", flash_mla_packed_sparse_output),
    ]:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            fn(*args)
            torch.cuda.synchronize()
        counts[name] = sum(
            e.count for e in profile.key_averages() if e.key == "aten::mul_"
        )
    assert counts == {"full": 2, "output": 0}


def test_graph_output_equality():
    args = inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flash_mla_packed_sparse_output(*args)
        flash_mla_packed_sparse_fwd(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = flash_mla_packed_sparse_output(*args)
        full = flash_mla_packed_sparse_fwd(*args)[0]
    for index in [31, 87]:
        args[2].fill_(index)
        graph.replay()
        torch.testing.assert_close(output, full, rtol=0, atol=0)


@pytest.mark.parametrize("width", [128, 640])
def test_collision_free_output_equality(width):
    args = inputs()
    # Distinct keys below the table size have distinct multiplicative hashes.
    args[2] = torch.arange(width, device="cuda", dtype=torch.int32).repeat(513, 1)
    args[3].fill_(width)
    out = flash_mla_packed_sparse_output(*args)
    expected = flash_mla_packed_sparse_fwd(*args)[0]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [128, 640])
@pytest.mark.parametrize("rows", [512, 513])
@pytest.mark.parametrize("with_sink", [False, True])
def test_matches_stock(width, rows, with_sink):
    torch.manual_seed(419)
    q = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(4096, (rows, width), device="cuda", dtype=torch.int32)
    lengths = torch.randint(1, width + 1, (rows,), device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda") if with_sink else None
    stock = flash_mla_sparse_fwd(
        q, kv, indices.unsqueeze(1), 512**-0.5, 512, sink, lengths
    )
    packed = flash_mla_packed_sparse_fwd(
        q[:, :16],
        kv,
        indices,
        lengths,
        512**-0.5,
        sink[:16] if sink is not None else None,
    )
    reference = stock[0][:, :16].float()
    relative_l2 = (packed[0].float() - reference).norm() / reference.norm()
    assert relative_l2 < 0.005
    # Both public SGLang APIs expose auxiliary statistics in log2 units.
    for actual, expected in zip(packed[1:], stock[1:]):
        torch.testing.assert_close(actual, expected[:, :16], rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
