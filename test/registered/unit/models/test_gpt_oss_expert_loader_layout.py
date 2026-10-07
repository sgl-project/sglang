"""GPT-OSS packed expert checkpoints follow the destination expert layout."""

import re
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.quark.schemes.quark_w4a8_mxfp4_moe import (
    QuarkW4A8MXFp4MoE,
)
from sglang.srt.layers.quantization.quark.weights import (
    _load_gptoss_quark_expert_weights,
)
from sglang.srt.models.gpt_oss import GptOssForCausalLM
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

EXPERT_PATTERN = re.compile(
    r"^(.*\.mlp\.experts)\.(\d+)\.(gate_up_proj|down_proj)\."
    r"(weight|weight_scale|input_scale|bias)$"
)


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return get_parallel().override(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def quark_config():
    return QuarkConfig(
        quant_config={
            "packed_modules_mapping": {},
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "global_quant_config": {
                "weight": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": False,
                    "scale_format": "e8m0",
                },
                "input_tensors": {
                    "dtype": "fp8_e4m3",
                    "qscheme": "per_tensor",
                    "is_dynamic": False,
                },
                "output_tensors": None,
                "bias": None,
            },
        },
        is_prequantized=True,
    )


def build_model(kind, intermediate=384, original=None, device="cuda"):
    config = SimpleNamespace(
        hidden_size=128,
        intermediate_size=intermediate,
        num_local_experts=4,
        num_hidden_layers=1,
    )
    if original is not None:
        config.original_intermediate_size = original
    model = GptOssForCausalLM.__new__(GptOssForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = config
    model.quant_config = (
        Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
        if kind == "mxfp4"
        else quark_config()
        if kind == "quark"
        else None
    )
    # Quark's W4A8 runner is AMD-only. Keep native parameter construction and
    # the checkpoint loader while avoiding runner creation in this CUDA test.
    runner_scope = (
        patch.object(QuarkW4A8MXFp4MoE, "create_moe_runner", return_value=None)
        if kind == "quark"
        else nullcontext()
    )
    with torch.device(device), runner_scope:
        experts = FusedMoE(
            num_experts=4,
            top_k=2,
            hidden_size=128,
            intermediate_size=intermediate,
            layer_id=0,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=10.0,
            quant_config=model.quant_config,
            params_dtype=torch.bfloat16,
            with_bias=True,
            use_weight_loader_fused=kind == "unquantized",
            reduce_results=False,
            prefix="model.layers.0.mlp.experts",
        )
    if kind == "mxfp4":
        assert isinstance(experts.quant_method, Mxfp4MoEMethod)
    elif kind == "quark":
        assert isinstance(experts.scheme, QuarkW4A8MXFp4MoE)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = experts
    return model, experts


def values(shape, dtype, offset=0, device="cuda"):
    count = 1
    for size in shape:
        count *= size
    indices = torch.arange(count, device=device) + offset
    data = (
        (indices % 16 + (indices // 16 % 16) * 16)
        if dtype == torch.uint8
        else (indices % 31 - 15) / 64
    )
    return data.to(dtype).reshape(shape)


def load_experts(model, kind, *, changed=False, offset=0):
    module = model.model.layers[0].mlp.experts
    config = model.config
    intermediate = getattr(
        config, "original_intermediate_size", config.intermediate_size
    )
    hidden = config.hidden_size
    device = module.w13_weight.device
    source = {
        "w13_weight": values(
            (4, 2 * intermediate, hidden // 2), torch.uint8, offset, device
        ),
        "w2_weight": values(
            (4, hidden, intermediate // 2), torch.uint8, offset + 3, device
        ),
        "w13_weight_scale": values(
            (4, 2 * intermediate, hidden // 32), torch.uint8, offset + 5, device
        ),
        "w2_weight_scale": values(
            (4, hidden, intermediate // 32), torch.uint8, offset + 7, device
        ),
        "w13_weight_bias": values(
            (4, 2 * intermediate), module.w13_weight_bias.dtype, offset + 11, device
        ),
        "w2_weight_bias": values(
            (4, hidden), module.w2_weight_bias.dtype, offset + 13, device
        ),
    }
    if kind == "quark":
        source["w13_input_scale"] = values((4,), torch.float32, offset + 17, device)
        source["w2_input_scale"] = values((4,), torch.float32, offset + 19, device)
    prefix = "model.layers.0.mlp.experts"
    suffixes = (
        ("w13_weight", "gate_up_proj", "blocks"),
        ("w2_weight", "down_proj", "blocks"),
        ("w13_weight_scale", "gate_up_proj", "scales"),
        ("w2_weight_scale", "down_proj", "scales"),
        ("w13_weight_bias", "gate_up_proj", "bias"),
        ("w2_weight_bias", "down_proj", "bias"),
    )
    checkpoint = []
    if kind == "mxfp4":
        for param_name, projection, suffix in suffixes:
            data = source[param_name]
            if suffix == "blocks":
                data = data.unflatten(-1, (-1, 16))
            checkpoint.append((f"{prefix}.{projection}_{suffix}", data))
        checkpoint.append((f"{prefix}.unused_tensor", torch.ones(1, device=device)))
    else:
        for expert in range(4):
            for param_name, projection, suffix in suffixes:
                suffix = {"blocks": "weight", "scales": "weight_scale"}.get(
                    suffix, suffix
                )
                checkpoint.append(
                    (
                        f"{prefix}.{expert}.{projection}.{suffix}",
                        source[param_name][expert],
                    )
                )
            for param_name, projection in (
                ("w13_input_scale", "gate_up_proj"),
                ("w2_input_scale", "down_proj"),
            ):
                checkpoint.append(
                    (
                        f"{prefix}.{expert}.{projection}.input_scale",
                        source[param_name][expert],
                    )
                )
        checkpoint.append(("not-an-expert-weight", torch.ones(1, device=device)))
    before = {name: param.detach().clone() for name, param in module.named_parameters()}
    with loading_scope(changed):
        loaded = (
            model._load_mxfp4_experts_weights(iter(checkpoint))
            if kind == "mxfp4"
            else _load_gptoss_quark_expert_weights(
                model, iter(checkpoint), EXPERT_PATTERN
            )
        )
    width = (
        (config.intermediate_size // 32 + module.moe_tp_size - 1) // module.moe_tp_size
    ) * 32
    start = module.moe_tp_rank * width
    stop = min(start + width, intermediate)
    expert_start = module.moe_ep_rank * module.num_local_experts
    expert_stop = expert_start + module.num_local_experts
    expected = {}
    for name, original_data in source.items():
        data = original_data[expert_start:expert_stop]
        target = before[name]
        if name.startswith("w13") and "input_scale" not in name:
            if kind == "mxfp4":
                data = data[:, 2 * start : 2 * stop]
                if data.dim() == 3:
                    target[:, : data.shape[1], : data.shape[2]] = data
                else:
                    target[:, : data.shape[1]] = data
            else:
                gate = data[:, 0::2, ...][:, start:stop]
                up = data[:, 1::2, ...][:, start:stop]
                half = target.shape[1] // 2
                if data.dim() == 3:
                    target[:, : gate.shape[1], : gate.shape[2]] = gate
                    target[:, half : half + up.shape[1], : up.shape[2]] = up
                else:
                    target[:, : gate.shape[1]] = gate
                    target[:, half : half + up.shape[1]] = up
        elif name == "w2_weight":
            data = data[..., start // 2 : stop // 2]
            target[:, : data.shape[1], : data.shape[2]] = data
        elif name == "w2_weight_scale":
            data = data[..., start // 32 : stop // 32]
            target[:, : data.shape[1], : data.shape[2]] = data
        elif name == "w2_weight_bias":
            target[:, : data.shape[1]] = (
                data if module.moe_tp_rank == 0 else torch.zeros_like(data)
            )
        else:
            target.copy_(data)
        torch.testing.assert_close(getattr(module, name), target, rtol=0, atol=0)
        expected[name] = target
    assert loaded == {f"{prefix}.{name}" for name in source}
    return expected


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestGptOssExpertLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def check_loads(self, changed):
        with torch.inference_mode():
            for ep_size in (1, 2, 4):
                for rank in range(4):
                    reset_context()
                    publish(
                        ServerArgs(
                            model_path="dummy",
                            device="cuda",
                            tp_size=4,
                            ep_size=ep_size,
                            moe_runner_backend="triton",
                        ),
                        role="test",
                        ranks=SpawnRanks(world_rank=rank),
                    )
                    for kind in ("mxfp4", "quark"):
                        for intermediate, original in (
                            ((384, None), (288, None), (384, 288))
                            if kind == "mxfp4"
                            else ((384, None), (512, None))
                        ):
                            model, _ = build_model(kind, intermediate, original)
                            for offset in (0, 11):
                                load_experts(
                                    model, kind, changed=changed, offset=offset
                                )

    def test_normal_down_projection_bias_uses_the_expert_rank(self):
        with torch.inference_mode():
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy",
                        device="cuda",
                        tp_size=4,
                        moe_runner_backend="triton",
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                model, module = build_model("unquantized")
                for changed in (False, True):
                    bias = values((4, 128), module.w2_weight_bias.dtype)
                    expected = bias.clone() if rank == 0 else torch.zeros_like(bias)
                    with loading_scope(changed):
                        model._load_normal_weights(
                            [("model.layers.0.mlp.experts.down_proj_bias", bias)],
                            is_nextn=False,
                            weight_name_mapping=None,
                        )
                    torch.testing.assert_close(
                        module.w2_weight_bias, expected, rtol=0, atol=0
                    )

    def test_native_loaders_after_scope_exit(self):
        self.check_loads(True)


if __name__ == "__main__":
    unittest.main()
