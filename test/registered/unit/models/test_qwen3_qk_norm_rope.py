"""Model dispatch and changing-input graph replay; kernel numerics are tested separately."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import factory
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def runtime():
    defaults = dict(
        enable_fused_qk_norm_rope=True,
        bf16_gemm_backend="torch",
        attention_backend="fa3",
        cuda_graph_config=CudaGraphConfig(),
    )
    with (
        torch.inference_mode(),
        get_context().override_server_args(**defaults),
        get_parallel().override(attn_tp_size=1, attn_tp_group=None, tp_group=None),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False),
        patch.dict(factory._ROPE_DICT, clear=True),
    ):
        torch.manual_seed(19)
        yield defaults


def make_layer(d=128, dtype=torch.bfloat16, enable_prefill_qk_norm_rope=True, **kwargs):
    layer = qwen3.Qwen3Attention(
        hidden_size=8 * d,
        num_heads=8,
        num_kv_heads=2,
        head_dim=d,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        enable_prefill_qk_norm_rope=enable_prefill_qk_norm_rope,
        **kwargs,
    )
    # Moving model parameters must not downcast the FP32 RoPE table.
    cache = layer.rotary_emb.cos_sin_cache.cuda()
    layer = layer.to("cuda", dtype)
    layer.rotary_emb.cos_sin_cache = cache
    for name, weight in layer.named_parameters():
        weight.normal_(std=0.05)
        if "norm" in name:
            weight.add_(1)
    return layer


def separate(layer, positions, x, batch):
    with (
        patch.object(layer, "use_table_qk_norm_rope", False),
        patch.object(layer, "use_prefill_qk_norm_rope", False),
    ):
        return layer.forward_prepare_native(positions, x, batch)


def assert_same_qkv(actual, expected):
    for got, ref in zip(actual, expected, strict=True):
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize(
    "mode,dtype,should_fuse",
    [
        (ForwardMode.DECODE, torch.bfloat16, True),
        (ForwardMode.EXTEND, torch.bfloat16, False),
        (ForwardMode.TARGET_VERIFY, torch.bfloat16, False),
        (ForwardMode.MIXED, torch.bfloat16, False),
        (ForwardMode.DRAFT_EXTEND_V2, torch.bfloat16, False),
        (ForwardMode.DECODE, torch.float16, False),
    ],
)
def test_model_dispatch(mode, dtype, should_fuse):
    layer = make_layer(dtype=dtype)
    assert layer.use_table_qk_norm_rope
    should_prefill = (
        mode == ForwardMode.EXTEND and torch.cuda.get_device_capability()[0] == 9
    )
    if should_prefill:
        assert layer.use_prefill_qk_norm_rope
    x = torch.randn(4, layer.hidden_size, device="cuda", dtype=dtype)
    positions = torch.arange(4, device="cuda") + 17
    batch = SimpleNamespace(forward_mode=mode)
    expected = separate(layer, positions, x, batch)
    # Only attention itself is stubbed: construction, projections, norm and RoPE are real.
    with (
        patch.object(layer.attn, "forward", side_effect=lambda q, *a, **kw: q) as attn,
        patch.object(
            qwen3, "table_qk_norm_rope_", wraps=qwen3.table_qk_norm_rope_
        ) as fused,
        patch.object(qwen3, "qk_norm_rope", wraps=qwen3.qk_norm_rope) as prefill,
    ):
        layer(positions, x, batch)
    assert fused.call_count == int(should_fuse)
    if should_prefill:
        prefill.assert_called_once()
    else:
        prefill.assert_not_called()
    if should_fuse:
        assert fused.call_args.kwargs["match_cuda"] is True
    assert_same_qkv(attn.call_args.args[:3], expected)
    if mode == ForwardMode.EXTEND:
        assert attn.call_args.args[0].is_contiguous()


@pytest.mark.parametrize(
    "settings,layer_args,deterministic_env",
    [
        ({"enable_fused_qk_norm_rope": False}, {}, False),
        ({"enable_deterministic_inference": True}, {}, False),
        ({"rl_on_policy_target": "fsdp"}, {}, False),
        (
            {
                "cuda_graph_config": CudaGraphConfig(
                    prefill=PhaseConfig(tc_compiler="inductor")
                )
            },
            {},
            False,
        ),
        ({}, {"d": 512}, False),
        ({}, {"rope_scaling": {"rope_type": "linear", "factor": 2.0}}, False),
        ({}, {}, True),
    ],
)
def test_constructor_exclusions(runtime, settings, layer_args, deterministic_env):
    with (
        get_context().override_server_args(**{**runtime, **settings}),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(deterministic_env),
    ):
        layer = make_layer(**layer_args)
        assert not layer.use_table_qk_norm_rope
        assert not layer.use_prefill_qk_norm_rope


@pytest.mark.parametrize("excluded", ["caller", "backend", "head_dim", "weights"])
def test_prefill_fallbacks(runtime, excluded):
    settings = (
        {**runtime, "attention_backend": "triton"} if excluded == "backend" else runtime
    )
    with get_context().override_server_args(**settings):
        layer = make_layer(
            d=64 if excluded == "head_dim" else 128,
            enable_prefill_qk_norm_rope=excluded != "caller",
        )
    if excluded == "weights":
        layer.q_norm.weight = torch.nn.Parameter(layer.q_norm.weight.float())
        layer.k_norm.weight = torch.nn.Parameter(layer.k_norm.weight.float())
    x = torch.randn(4, layer.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(4, device="cuda")
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    if excluded == "weights":
        # The separate CUDA norm also rejects mixed activation/weight dtypes.
        with pytest.raises(RuntimeError, match="Dtype value"):
            layer.forward_prepare_native(positions, x, batch)
        with pytest.raises(RuntimeError, match="Dtype value"):
            separate(layer, positions, x, batch)
        return
    actual = layer.forward_prepare_native(positions, x, batch)
    assert_same_qkv(actual, separate(layer, positions, x, batch))
    assert not actual[0].is_contiguous()


@pytest.mark.parametrize(
    "mode,dtype,batch_size",
    [
        (ForwardMode.DECODE, torch.bfloat16, 1),
        (ForwardMode.DECODE, torch.bfloat16, 4),
        (ForwardMode.EXTEND, torch.bfloat16, 17),
        (ForwardMode.EXTEND, torch.float16, 17),
    ],
)
def test_changing_input_graph_replay(mode, dtype, batch_size):
    layer = make_layer(dtype=dtype)
    x = torch.randn(batch_size, layer.hidden_size, device="cuda", dtype=dtype)
    positions = torch.arange(batch_size, device="cuda")
    batch = SimpleNamespace(forward_mode=mode)
    layer.forward_prepare_native(positions, x, batch)  # JIT warmup before capture.
    graph = torch.cuda.CUDAGraph()
    with (
        patch.object(
            qwen3, "table_qk_norm_rope_", wraps=qwen3.table_qk_norm_rope_
        ) as fused,
        torch.cuda.graph(graph),
    ):
        actual = layer.forward_prepare_native(positions, x, batch)
    assert fused.call_count == int(mode == ForwardMode.DECODE)
    for offset in (0, 17, 2048):
        x.normal_()
        positions.copy_(torch.arange(batch_size, device="cuda") + offset)
        graph.replay()
        assert_same_qkv(actual, separate(layer, positions, x, batch))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
