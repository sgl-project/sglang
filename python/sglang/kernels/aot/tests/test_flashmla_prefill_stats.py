"""Sparse prefill's public statistics are base-2, including the fused H64 path."""

import math

import pytest
import torch
from sgl_kernel.flash_mla import flash_mla_sparse_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def supported_device():
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("FlashMLA sparse prefill requires SM90 or SM100-family")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def make_inputs(heads, dim, topk, mode):
    torch.manual_seed(42)
    q = torch.randn(3, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1024, 1, dim, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(1024, (3, 1, topk), device="cuda", dtype=torch.int32)
    sink = torch.linspace(-3, 3, heads, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([0, 65, topk], device="cuda", dtype=torch.int32)
    if mode == "masked":
        indices[:, :, ::7] = -1
        indices[:, :, 1::11] = 1024
    elif mode == "rising":
        q.fill_(1)
        kv.copy_(torch.linspace(-2, 2, 1024, device="cuda")[:, None, None])
        indices.copy_(torch.linspace(0, 1023, topk, device="cuda").int())
    elif mode == "negative":
        q.fill_(-1)
        kv.fill_(1)
    elif mode == "empty":
        indices.fill_(-1)
        sink = lengths = None
    elif mode == "sink-positive":
        sink.fill_(1000)
    elif mode == "sink-negative":
        sink.fill_(-1000)
    elif mode == "no-options":
        sink = lengths = None
    return q, kv, indices, sink, lengths


def call(inputs):
    q, kv, indices, sink, lengths = inputs
    return flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        q.shape[-1] ** -0.5,
        attn_sink=sink,
        topk_length=lengths,
    )


def check(inputs, result):
    q, kv, indices, sink, lengths = inputs
    selected_indices = indices[:, 0].long()
    valid = (selected_indices >= 0) & (selected_indices < kv.shape[0])
    if lengths is not None:
        valid &= torch.arange(indices.shape[-1], device=q.device) < lengths[:, None]
    selected = kv[:, 0][selected_indices.clamp(0, kv.shape[0] - 1)].float()
    logits = (q.float() @ selected.transpose(1, 2)) * q.shape[-1] ** -0.5
    logits.masked_fill_(~valid[:, None], -torch.inf)
    maximum = logits.amax(-1)
    lse = logits.logsumexp(-1)
    # Sink normalizes output but does not enter either public statistic.
    denominator = lse if sink is None else torch.logaddexp(lse, sink[None])
    probabilities = (logits - denominator[:, :, None]).exp().nan_to_num()
    expected_out = probabilities @ selected[:, :, :512]
    expected_lse = torch.where(valid.any(-1)[:, None], lse, torch.inf)

    out, got_max, got_lse = result
    assert out.dtype == torch.bfloat16
    assert got_max.dtype == got_lse.dtype == torch.float32
    torch.testing.assert_close(out.float(), expected_out, atol=0.012, rtol=0.025)
    torch.testing.assert_close(
        got_max, maximum * math.log2(math.e), atol=2e-4, rtol=2e-4
    )
    torch.testing.assert_close(
        got_lse, expected_lse * math.log2(math.e), atol=2e-4, rtol=2e-4
    )


@pytest.mark.parametrize("heads,dim", [(64, 512), (64, 576), (128, 512), (128, 576)])
@pytest.mark.parametrize("topk", [128, 512])
@pytest.mark.parametrize(
    "mode",
    [
        "masked",
        "rising",
        "negative",
        "empty",
        "sink-positive",
        "sink-negative",
        "no-options",
    ],
)
def test_prefill_stats_units(heads, dim, topk, mode):
    inputs = make_inputs(heads, dim, topk, mode)
    check(inputs, call(inputs))


@pytest.mark.parametrize("heads,dim", [(64, 512), (64, 576), (128, 512), (128, 576)])
def test_prefill_stats_graph_mutation(heads, dim):
    inputs = make_inputs(heads, dim, 128, "masked")
    for _ in range(3):
        call(inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call(inputs)
    for value in inputs:
        original = value.clone()
        if value.dtype == torch.int32:
            value.zero_()
        else:
            value.add_(0.125)
        graph.replay()
        torch.cuda.synchronize()
        check(inputs, output)
        value.copy_(original)


def test_h64_stats_conversion_is_fused():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Only SM100-family H64 fuses the statistics conversion")
    if (
        torch.profiler.ProfilerActivity.CUDA
        not in torch.profiler.supported_activities()
    ):
        pytest.skip("CUDA profiling unavailable")
    inputs = make_inputs(64, 512, 128, "no-options")
    for _ in range(5):
        call(inputs)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        call(inputs)
        torch.cuda.synchronize()
    kernels = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 1, [e.name for e in kernels]
    assert "sparse_attn_fwd_kernel" in kernels[0].name
