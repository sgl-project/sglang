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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mixed_fp8", [False, True])
@pytest.mark.parametrize(
    "tp_size,tp_rank,attn_size,attn_rank",
    [
        (1, 0, 1, 0),
        (4, 0, 4, 0),
        (4, 3, 4, 3),
        (8, 7, 1, 0),  # Full DP: global rank 7 owns all attention heads.
        (8, 7, 4, 3),  # Partial DP: global rank 7 owns attention shard 3.
    ],
)
def test_bf16_projection_loading_and_outputs(
    monkeypatch, tp_size, tp_rank, attn_size, attn_rank, mixed_fp8
):
    import sglang.srt.layers.linear as linear
    import sglang.srt.models.glm5_next as glm
    from sglang.srt.configs.glm5_next import Glm5NextTextConfig
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    parallel = SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_tp_size=attn_size,
        attn_tp_rank=attn_rank,
    )
    # Keep the global group distinct in the Linear defaults too, so omitting
    # an explicit attention rank/size cannot accidentally use the right shard.
    monkeypatch.setattr(glm, "get_parallel", lambda: parallel)
    monkeypatch.setattr(linear, "get_parallel", lambda: parallel)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
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
        prefix = "model.layers.0.self_attn"
        projection_names = (
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "o_proj",
        )
        mixed_config = Fp8Config(
            ignored_layers=[f"{prefix}.{name}" for name in projection_names]
        )
        quant_config = mixed_config if mixed_fp8 else None
        with torch.device("cuda"):
            fused = glm.Glm5NextLinearAttention(
                0, 256, config, quant_config=quant_config, prefix=prefix
            )
            # Construct the existing unfused reference with identical precision.
            with patch.object(
                glm, "are_linear_prefixes_unquantized", return_value=False
            ):
                unfused = glm.Glm5NextLinearAttention(
                    0, 256, config, quant_config=quant_config, prefix=prefix
                )
        assert fused.do_fuse_qkvbfg
        assert not unfused.do_fuse_qkvbfg

        if mixed_fp8 and tp_size == 1:
            for lora in (
                SimpleNamespace(enable_lora=True, lora_paths=None),
                SimpleNamespace(enable_lora=False, lora_paths=["adapter"]),
            ):
                with (
                    torch.device("cuda"),
                    patch.object(glm, "get_lora", return_value=lora),
                ):
                    with_lora = glm.Glm5NextLinearAttention(
                        0, 256, config, quant_config=quant_config, prefix=prefix
                    )
                assert not with_lora.do_fuse_qkvbfg
                assert hasattr(with_lora, "qkv_proj")
                del with_lora

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
            fused.fused_fg_b_proj.weight_loader(
                fused.fused_fg_b_proj.weight, weight, shard
            )
            module = getattr(unfused, name)
            module.weight_loader(module.weight, weight)

        def check_outputs(actual, expected, tokens):
            widths = (3 * 8192 // attn_size, 64 // attn_size) + (8192 // attn_size,) * 2
            for got, ref, width in zip(actual, expected, widths):
                assert got.shape == ref.shape == (tokens, width)
                torch.testing.assert_close(got, ref, atol=0.004, rtol=0.01)

        with torch.inference_mode():
            for tokens in (1, 2, 17):
                hidden = torch.randn(
                    tokens, 256, device="cuda", generator=generator
                ).bfloat16()
                check_outputs(
                    fused.forward_qkvbfg_fused(hidden, None),
                    unfused.forward_qkvbfg(hidden, None),
                    tokens,
                )
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
                    torch.randn(
                        tokens, 256, device="cuda", generator=generator
                    ).bfloat16()
                )
                graph.replay()
                check_outputs(captured, unfused.forward_qkvbfg(hidden, None), tokens)
    finally:
        torch.set_default_dtype(old_dtype)
        reset_context()


@pytest.mark.parametrize(
    "quantized_projection",
    [None, "qkv_proj", "b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"],
)
def test_fp8_projection_eligibility(quantized_projection):
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.utils import are_linear_prefixes_unquantized

    prefix = "model.layers.0.self_attn"
    names = ("qkv_proj", "b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj")
    ignored = []
    for name in names:
        if name == quantized_projection:
            continue
        # Exercise the checkpoint's unfused Q/K/V names too.
        for shard in ("q_proj", "k_proj", "v_proj") if name == "qkv_proj" else (name,):
            ignored.append(f"{prefix}.{shard}")
    config = Fp8Config(ignored_layers=ignored)
    assert are_linear_prefixes_unquantized(
        config, [f"{prefix}.{name}" for name in names]
    ) == (quantized_projection is None)


def test_fusion_does_not_probe_other_quant_configs():
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.utils import are_linear_prefixes_unquantized

    class ClassSensitiveConfig(Fp8Config):
        def get_quant_method(self, layer, prefix):
            raise AssertionError("A base-class probe cannot resolve this config")

    assert not are_linear_prefixes_unquantized(ClassSensitiveConfig(), ["qkv_proj"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
