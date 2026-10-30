"""GLM-5.3-Flash projection scheduling under CUDA graph capture."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import unquant
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.models.glm5_next import Glm5NextLinearAttention
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="the projection overlap is tuned for SM90",
)


class GemmStreams(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.streams = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (
            torch.ops.aten.linear.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
        ):
            self.streams.add(torch.cuda.current_stream().cuda_stream)
        return func(*args, **(kwargs or {}))


@pytest.fixture
def runtime():
    with (
        get_context().override_server_args(
            device="cuda", enable_lora=False, lora_paths=None, bf16_gemm_backend="torch"
        ),
        envs.SGLANG_GLM5_KDA_PROJ_OVERLAP.override(True),
    ):
        yield


def make_attention(tp, *, alt_stream=True, **kwargs):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with (
            torch.device("cuda"),
            get_parallel().override(
                tp_size=tp,
                tp_rank=0,
                attn_tp_size=tp,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_cp_size=1,
                moe_tp_size=tp,
            ),
        ):
            attention = Glm5NextLinearAttention(
                layer_idx=0,
                hidden_size=4096,
                config=SimpleNamespace(
                    linear_attn_config={
                        "head_dim": 128,
                        "num_heads": 64,
                        "short_conv_kernel_size": 4,
                    }
                ),
                alt_stream=torch.cuda.Stream() if alt_stream else None,
                **kwargs,
            )
        with torch.no_grad():
            attention.fused_qkvbfg_a_proj.weight.normal_(std=4096**-0.5)
            attention.fused_fg_b_proj.weight.normal_(std=128**-0.5)
        return attention
    finally:
        torch.set_default_dtype(old_dtype)


def split_reference(attention, x):
    nq, nb, nf = attention.split_sizes
    w = attention.fused_qkvbfg_a_proj.weight
    qkv = F.linear(x, w[:nq])
    beta, fg = F.linear(x, w[nq:]).split([nb, nf], dim=-1)
    forget, gate = attention.fused_fg_b_proj(
        fg.view(-1, 2, attention.head_dim).transpose(0, 1)
    )
    return qkv, beta, forget, gate


def merged_reference(attention, x):
    qkv, beta, fg = F.linear(x, attention.fused_qkvbfg_a_proj.weight).split(
        attention.split_sizes, dim=-1
    )
    forget, gate = attention.fused_fg_b_proj(
        fg.view(-1, 2, attention.head_dim).transpose(0, 1)
    )
    return qkv, beta, forget, gate


def capture_projection(attention, x, forward_batch=None):
    graph = torch.cuda.CUDAGraph()
    with model_capture_mode():
        for _ in range(3):
            attention.forward_qkvbfg_fused(x, forward_batch)
        torch.cuda.synchronize()
        with GemmStreams() as observed, torch.cuda.graph(graph):
            actual = attention.forward_qkvbfg_fused(x, forward_batch)
    return graph, actual, observed.streams


@torch.inference_mode()
@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("m", [1, 6])
def test_projection_uses_two_streams_and_replays_new_inputs(runtime, tp, m):
    torch.manual_seed(42)
    attention = make_attention(tp)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    graph, actual, streams = capture_projection(attention, x)
    assert len(streams) == 2, "QKV and the gate chain must use separate streams"
    for _ in range(5):
        x.normal_()
        graph.replay()
        reference = split_reference(attention, x)
        for got, expected in zip(actual, reference):
            torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize("tp,m", [(4, 32), (8, 15)])
def test_last_overlap_shape(runtime, tp, m):
    attention = make_attention(tp)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    graph, actual, streams = capture_projection(attention, x)
    assert len(streams) == 2
    graph.replay()
    for got, expected in zip(actual, split_reference(attention, x)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize("tp,m", [(4, 33), (8, 16), (8, 128), (2, 1)])
def test_untuned_shapes_keep_the_merged_projection(runtime, tp, m):
    attention = make_attention(tp)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    graph, actual, streams = capture_projection(attention, x)
    assert len(streams) == 1
    graph.replay()
    for got, expected in zip(actual, merged_reference(attention, x)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize("alt_stream", [False, True])
def test_eager_keeps_the_merged_projection(runtime, alt_stream):
    attention = make_attention(4, alt_stream=alt_stream)
    x = torch.randn(6, 4096, device="cuda", dtype=torch.bfloat16)
    with GemmStreams() as observed:
        actual = attention.forward_qkvbfg_fused(x, None)
    assert len(observed.streams) == 1
    for got, expected in zip(actual, merged_reference(attention, x)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize(
    "settings",
    [
        {"enable_deterministic_inference": True},
        {"enable_torch_compile": True},
        {"enable_two_batch_overlap": True},
        {"enable_single_batch_overlap": True},
        {"debug_cuda_graph": True},
    ],
)
def test_incompatible_execution_modes_keep_the_merge(runtime, settings):
    with get_context().override_server_args(device="cuda", **settings):
        attention = make_attention(4)
        x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
        graph, actual, streams = capture_projection(attention, x)
        assert len(streams) == 1
        graph.replay()
        for got, expected in zip(actual, merged_reference(attention, x)):
            torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_environment_opt_out_preserves_the_merge(runtime):
    with envs.SGLANG_GLM5_KDA_PROJ_OVERLAP.override(False):
        attention = make_attention(4)
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    graph, actual, streams = capture_projection(attention, x)
    assert len(streams) == 1
    graph.replay()
    for got, expected in zip(actual, merged_reference(attention, x)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize(
    "backend", [unquant.Bf16GemmBackend.GEMV, unquant.Bf16GemmBackend.CUTEDSL]
)
def test_other_gemm_backends_keep_the_merge(runtime, monkeypatch, backend):
    attention = make_attention(4)
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", backend)
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    graph, actual, streams = capture_projection(attention, x)
    assert len(streams) == 1
    graph.replay()
    for got, expected in zip(actual, merged_reference(attention, x)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_non_bf16_inputs_keep_the_merge(runtime):
    attention = make_attention(4).float()
    x = torch.randn(1, 4096, device="cuda", dtype=torch.float32)
    graph, actual, streams = capture_projection(attention, x)
    assert len(streams) == 1
    graph.replay()
    for got, expected in zip(actual, merged_reference(attention, x)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize(
    "mode,overlap",
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.TARGET_VERIFY, True),
        (ForwardMode.EXTEND, False),
    ],
)
def test_only_decode_and_verify_overlap(runtime, mode, overlap):
    attention = make_attention(8)
    x = torch.randn(6, 4096, device="cuda", dtype=torch.bfloat16)
    graph, actual, streams = capture_projection(
        attention, x, SimpleNamespace(forward_mode=mode)
    )
    assert len(streams) == (2 if overlap else 1)
    graph.replay()
    reference = split_reference if overlap else merged_reference
    for got, expected in zip(actual, reference(attention, x)):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
