"""Qwen3 QK-norm/RoPE dispatch and graph replay; no checkpoint or server."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

INDUCTOR_PREFILL = CudaGraphConfig(prefill=PhaseConfig(tc_compiler="inductor"))


@pytest.fixture(autouse=True)
def runtime():
    defaults = dict(
        enable_fused_qk_norm_rope=True,
        bf16_gemm_backend="torch",
        cuda_graph_config=CudaGraphConfig(),
    )
    with (
        torch.inference_mode(),
        get_context().override_server_args(**defaults),
        get_parallel().override(attn_tp_size=1, attn_tp_group=None, tp_group=None),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False),
    ):
        torch.manual_seed(19)
        yield defaults


def make_layer(d=128, dtype=torch.bfloat16, **kwargs):
    layer = qwen3.Qwen3Attention(
        hidden_size=8 * d,
        num_heads=8,
        num_kv_heads=2,
        head_dim=d,
        rms_norm_eps=1e-6,
        max_position_embeddings=131072,
        **kwargs,
    )
    cache = layer.rotary_emb.cos_sin_cache.cuda()
    layer = layer.to("cuda", dtype)
    layer.rotary_emb.cos_sin_cache = cache
    for name, weight in layer.named_parameters():
        weight.normal_(std=0.05)
        if "norm" in name:
            weight.add_(1)
    return layer


def assert_prepared_equal(actual, expected):
    assert len(actual) == len(expected) == 3
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize(
    "d,mode,dtype,should_fuse",
    [
        (64, ForwardMode.DECODE, torch.bfloat16, True),
        (128, ForwardMode.DECODE, torch.bfloat16, True),
        (256, ForwardMode.DECODE, torch.bfloat16, True),
        (128, ForwardMode.EXTEND, torch.bfloat16, False),
        (128, ForwardMode.TARGET_VERIFY, torch.bfloat16, False),
        (128, ForwardMode.DECODE, torch.float16, False),
    ],
)
def test_dispatch(d, mode, dtype, should_fuse):
    layer = make_layer(d, dtype=dtype)
    x = torch.randn(4, 8 * d, device="cuda", dtype=dtype)
    pos = torch.arange(4, device="cuda") + 120000
    expected = layer.forward_prepare_native(pos, x)  # Missing batch uses fallback.
    # Run the real model forward; attention MMA is outside this test's scope.
    with (
        patch.object(layer.attn, "forward", side_effect=lambda q, *a, **kw: q) as attn,
        patch.object(
            qwen3, "table_qk_norm_rope_", wraps=qwen3.table_qk_norm_rope_
        ) as fused,
    ):
        out = layer(pos, x, SimpleNamespace(forward_mode=mode))
    assert fused.call_count == int(should_fuse)
    assert_prepared_equal(attn.call_args.args[:3], expected)
    assert torch.equal(out, layer.o_proj(expected[0])[0])


@pytest.mark.parametrize("d", [64, 128, 256])
def test_graph_replay(d):
    """Capture projection + fused preparation, then change input and position buffers."""
    layer = make_layer(d)
    x = torch.randn(4, 8 * d, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(4, device="cuda")
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    layer.forward_prepare_native(pos, x, batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer.forward_prepare_native(pos, x, batch)
    for offset in (0, 8192, 120000):
        x.normal_()
        pos.copy_(torch.arange(4, device="cuda") + offset)
        graph.replay()
        assert_prepared_equal(actual, layer.forward_prepare_native(pos, x))


@pytest.mark.parametrize(
    "settings,layer_args,deterministic_env",
    [
        ({"enable_fused_qk_norm_rope": False}, {}, False),
        ({"enable_deterministic_inference": True}, {}, False),
        ({"rl_on_policy_target": "fsdp"}, {}, False),
        ({"cuda_graph_config": INDUCTOR_PREFILL}, {}, False),
        ({}, {"d": 512}, False),
        ({}, {"rope_scaling": {"rope_type": "linear", "factor": 2.0}}, False),
        ({}, {}, True),
    ],
)
def test_exclusions(runtime, settings, layer_args, deterministic_env):
    with (
        get_context().override_server_args(**{**runtime, **settings}),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(deterministic_env),
    ):
        assert not make_layer(**layer_args).use_table_qk_norm_rope


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
