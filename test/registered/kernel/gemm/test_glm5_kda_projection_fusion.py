"""BF16 KDA projection loading and outputs follow the attention TP group."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)
PREFIX = "model.layers.0.self_attn"
PROJECTIONS = ("qkv_proj", "b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj")


def _fp8_config(quantized_projection=None):
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    ignored = [f"{PREFIX}.o_proj"]
    for name in PROJECTIONS:
        if name == quantized_projection:
            continue
        # Match the checkpoint's unfused Q/K/V names.
        for shard in ("q_proj", "k_proj", "v_proj") if name == "qkv_proj" else (name,):
            ignored.append(f"{PREFIX}.{shard}")
    return Fp8Config(ignored_layers=ignored)


@pytest.fixture
def make_layer(monkeypatch):
    import sglang.srt.layers.linear as linear
    import sglang.srt.models.glm5_next as glm
    from sglang.srt.configs.glm5_next import Glm5NextTextConfig

    config = Glm5NextTextConfig(
        hidden_size=256,
        dtype=torch.bfloat16,
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [0],
        },
    )
    parallel = SimpleNamespace()
    # Patch Linear defaults too, so missing explicit attention TP uses the wrong group.
    monkeypatch.setattr(glm, "get_parallel", lambda: parallel)
    monkeypatch.setattr(linear, "get_parallel", lambda: parallel)

    def create(tp=(1, 0, 1, 0), quant_config=None, enable_lora=False, lora_paths=None):
        (
            parallel.tp_size,
            parallel.tp_rank,
            parallel.attn_tp_size,
            parallel.attn_tp_rank,
        ) = tp
        with (
            torch.device("cuda"),
            patch.object(
                glm,
                "get_lora",
                return_value=SimpleNamespace(
                    enable_lora=enable_lora, lora_paths=lora_paths
                ),
            ),
        ):
            return glm.Glm5NextLinearAttention(
                0, 256, config, quant_config=quant_config, prefix=PREFIX
            )

    old_dtype = torch.get_default_dtype()
    reset_context()
    try:
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        torch.set_default_dtype(torch.bfloat16)
        yield create
    finally:
        torch.set_default_dtype(old_dtype)
        reset_context()


@pytest.mark.parametrize(
    "tp,mixed_fp8",
    [
        ((4, 3, 4, 3), False),
        ((8, 7, 4, 3), False),
        ((4, 3, 4, 3), True),
        ((8, 7, 1, 0), True),  # Full DP: global rank 7 owns all attention heads.
        ((8, 7, 4, 3), True),  # Partial DP: global rank 7 owns attention shard 3.
    ],
)
def test_bf16_projection_loading_and_outputs(make_layer, tp, mixed_fp8):
    quant_config = _fp8_config() if mixed_fp8 else None
    fused = make_layer(tp, quant_config)
    # LoRA eligibility selects the existing unfused reference without an adapter.
    unfused = make_layer(tp, quant_config, enable_lora=True)
    assert fused.do_fuse_qkvbfg
    assert not unfused.do_fuse_qkvbfg
    attn_size = tp[2]

    generator = torch.Generator(device="cuda").manual_seed(5381)
    mapping = [
        (8192, "qkv_proj", "q"),
        (8192, "qkv_proj", "k"),
        (8192, "qkv_proj", "v"),
        (64, "b_proj", None),
        (128, "f_a_proj", None),
        (128, "g_a_proj", None),
    ]
    for shard, (size, name, qkv_id) in enumerate(mapping):
        weight = (
            torch.randn(size, 256, device="cuda", generator=generator) * 0.025
        ).bfloat16()
        fused.fused_qkvbfg_a_proj.weight_loader(
            fused.fused_qkvbfg_a_proj.weight, weight, shard
        )
        module = getattr(unfused, name)
        if qkv_id is None:
            module.weight_loader(module.weight, weight)
        else:
            module.weight_loader(module.weight, weight, qkv_id)
    for shard, name in enumerate(("f_b_proj", "g_b_proj")):
        weight = (
            torch.randn(8192, 128, device="cuda", generator=generator) * 0.025
        ).bfloat16()
        fused.fused_fg_b_proj.weight_loader(fused.fused_fg_b_proj.weight, weight, shard)
        module = getattr(unfused, name)
        module.weight_loader(module.weight, weight)

    def check_outputs(actual, expected, tokens):
        widths = (3 * 8192 // attn_size, 64 // attn_size) + (8192 // attn_size,) * 2
        for got, ref, width in zip(actual, expected, widths):
            assert got.shape == ref.shape == (tokens, width)
            torch.testing.assert_close(got, ref, atol=0.004, rtol=0.01)

    with torch.inference_mode():
        for tokens in (1, 17):
            hidden = torch.randn(
                tokens, 256, device="cuda", generator=generator
            ).bfloat16()
            check_outputs(
                fused.forward_qkvbfg_fused(hidden, None),
                unfused.forward_qkvbfg(hidden, None),
                tokens,
            )
            # Capture one representative DP topology; eager covers every case.
            if not (mixed_fp8 and tp == (8, 7, 4, 3)):
                continue
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fused.forward_qkvbfg_fused(hidden, None)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fused.forward_qkvbfg_fused(hidden, None)
            hidden.copy_(
                torch.randn(tokens, 256, device="cuda", generator=generator).bfloat16()
            )
            graph.replay()
            check_outputs(captured, unfused.forward_qkvbfg(hidden, None), tokens)


@pytest.mark.parametrize("quantized_projection", PROJECTIONS)
def test_fp8_projection_disables_fusion(make_layer, quantized_projection):
    layer = make_layer(quant_config=_fp8_config(quantized_projection))
    assert not layer.do_fuse_qkvbfg
    assert all(hasattr(layer, name) for name in PROJECTIONS)


def test_lora_paths_disable_fusion(make_layer):
    layer = make_layer(quant_config=_fp8_config(), lora_paths=["adapter"])
    assert not layer.do_fuse_qkvbfg
    assert hasattr(layer, "qkv_proj")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
