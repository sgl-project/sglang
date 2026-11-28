"""Single-NPU weight checks using one TP shard of a local LongCat2 FP8 checkpoint.

Does not load the model, start collectives, or require the LongCat LSA operator.
Run in the serving container with PYTHONPATH=python. See --help for arguments.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from safetensors import safe_open


def load_shard(model_path, layer, attention, tp_size, tp_rank, mtp=False):
    config = json.loads((model_path / "config.json").read_text())
    quant = config["quantization_config"]
    if quant["quant_method"] != "fp8" or quant["weight_block_size"] != [128, 128]:
        raise ValueError("Expected the native 128x128 block-FP8 checkpoint")
    if not 0 <= tp_rank < tp_size or config["num_attention_heads"] % tp_size:
        raise ValueError("Invalid TP size/rank for the checkpoint head count")
    head_dim = config["qk_nope_head_dim"] + config["v_head_dim"]
    rows = config["num_attention_heads"] // tp_size * head_dim
    start, end = tp_rank * rows, (tp_rank + 1) * rows
    if start % 128 or end % 128:
        raise ValueError("This check requires block-aligned TP shards")
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = (
        f"model.mtp.layers.{layer}.self_attn.kv_b_proj"
        if mtp
        else f"model.layers.{layer}.self_attn.{attention}.kv_b_proj"
    )
    tensors = []
    for suffix, first, last in (
        ("weight", start, end),
        ("weight_scale_inv", start // 128, end // 128),
    ):
        name = f"{prefix}.{suffix}"
        with safe_open(
            model_path / weight_map[name], framework="pt", device="cpu"
        ) as f:
            tensors.append(f.get_slice(name)[first:last, :].contiguous())
    return config, tensors[0], tensors[1]


def compare(name, output, reference, tolerance=0.01):
    output, reference = output.float().cpu(), reference.float().cpu()
    if output.shape != reference.shape or not torch.isfinite(output).all():
        raise AssertionError(f"{name}: invalid shape or non-finite output")
    relative_l2 = (
        (output - reference).norm() / reference.norm().clamp_min(1e-12)
    ).item()
    result = {"check": name, "relative_l2": relative_l2, "tolerance": tolerance}
    print(json.dumps(result), flush=True)
    if relative_l2 > tolerance:
        raise AssertionError(f"{name}: relative L2 exceeds {tolerance}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp-size", type=int, default=32)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--attention", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--mtp", action="store_true", help="Check the MTP decoder weights"
    )
    parser.add_argument("--check-empty-rank", action="store_true")
    args = parser.parse_args()
    # Select native G-B before the quantization methods inspect the environment.
    os.environ["SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8"] = "0"
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend, is_npu_arch35

    torch.npu.set_device(args.device)
    init_npu_backend()
    if not is_npu_arch35():
        raise RuntimeError("Run this check on arch35 / 950DT")

    from sglang.kernels.ops.gemm import (
        npu_w8a8_block_fp8_grouped_matmul,
        npu_w8a8_block_fp8_linear,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM
    from sglang.srt.models.longcat_flash_nextn import LongcatFlashForCausalLMNextN

    model_class = LongcatFlashForCausalLMNextN if args.mtp else LongcatFlashForCausalLM

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(args.device),
                "model_source": inspect.getfile(model_class),
                "mtp": args.mtp,
                "model_path": str(args.model_path),
                "tp_size": args.tp_size,
                "tp_rank": args.tp_rank,
            }
        ),
        flush=True,
    )
    config, weight, scale = load_shard(
        args.model_path,
        args.layer,
        args.attention,
        args.tp_size,
        args.tp_rank,
        args.mtp,
    )
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        raise TypeError(f"Expected E4M3 + FP32, got {weight.dtype}, {scale.dtype}")
    quant_config = Fp8Config.from_config(config["quantization_config"])
    if not Fp8LinearMethod(quant_config).use_npu_block_fp8:
        raise AssertionError("FP8 Linear did not select the native NPU block path")
    device_weight, device_scale = weight.to(args.device), scale.to(args.device)
    projection = SimpleNamespace(weight=device_weight, weight_scale_inv=device_scale)
    attentions = [
        SimpleNamespace(
            kv_b_proj=projection,
            qk_nope_head_dim=config["qk_nope_head_dim"],
            v_head_dim=config["v_head_dim"],
            w_kc=None,
            w_vc=None,
            w_scale=None,
        )
        for _ in range(1 if args.mtp else 2)
    ]
    owner = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=1, mla_scale_q_lora=False, mla_scale_kv_lora=False
        ),
        quant_config=quant_config,
        model=(
            SimpleNamespace(decoder=SimpleNamespace(self_attn=attentions[0]))
            if args.mtp
            else SimpleNamespace(layers=[SimpleNamespace(self_attn=attentions)])
        ),
    )
    model_class.post_load_weights(owner)
    rows, cols = weight.shape
    row_ids, col_ids = torch.arange(rows) // 128, torch.arange(cols) // 128
    dequant = weight.float() * scale[row_ids[:, None], col_ids[None, :]]
    head_dim = config["qk_nope_head_dim"] + config["v_head_dim"]
    expected = dequant.to(torch.bfloat16).reshape(-1, head_dim, cols)
    k_dim = config["qk_nope_head_dim"]
    attention = attentions[0]
    torch.testing.assert_close(
        attention.w_kc.cpu(), expected[:, :k_dim], rtol=0, atol=0
    )
    torch.testing.assert_close(
        attention.w_vc.cpu(), expected[:, k_dim:].transpose(1, 2), rtol=0, atol=0
    )
    assert attention.w_scale is None
    assert attention.w_vc.transpose(1, 2).is_contiguous()
    assert attention.w_kc.transpose(1, 2).is_contiguous()
    torch.testing.assert_close(
        device_weight.view(torch.uint8).cpu(), weight.view(torch.uint8)
    )
    torch.testing.assert_close(device_scale.cpu(), scale, rtol=0, atol=0)
    print(
        json.dumps({"check": "MLA weights", "exact": True, "shape": list(weight.shape)})
    )

    generator = torch.Generator().manual_seed(17)
    heads = expected.shape[0]
    for name, matrix in (("w_kc", attention.w_kc), ("w_vc", attention.w_vc)):
        x = torch.randn(17, heads, matrix.shape[1], generator=generator).to(
            torch.bfloat16
        )
        output = torch_npu.npu_transpose_batchmatmul(
            x.to(args.device),
            matrix,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        reference = torch.bmm(
            x.transpose(0, 1).float(), matrix.float().cpu()
        ).transpose(0, 1)
        compare(f"MLA BMM {name}", output, reference)

    x = torch.randn(17, cols, generator=generator).to(torch.bfloat16).to(args.device)
    qx, xs = torch.ops.npu.npu_dynamic_block_quant(
        x, dst_type=torch.float8_e4m3fn, row_block_size=1, col_block_size=128
    )
    x_dequant = qx.float().cpu() * xs.cpu().repeat_interleave(128, dim=-1)[:, :cols]
    output = npu_w8a8_block_fp8_linear(x, device_weight, device_scale)
    compare("block-FP8 Linear", output, x_dequant @ dequant.T)

    # Distinct scales per expert catch scale/group axis mixups; middle group is empty.
    expert_weights = weight.view(torch.uint8).repeat(3, 1, 1).view(torch.float8_e4m3fn)
    factors = torch.tensor([1.0, 1.3, 0.7])
    expert_scales = scale[None] * factors[:, None, None]
    expert_weights, expert_scales = (
        expert_weights.to(args.device),
        expert_scales.to(args.device),
    )
    groups = torch.tensor([7, 7, 17], dtype=torch.int64, device=args.device)
    output = npu_w8a8_block_fp8_grouped_matmul(
        x, expert_weights, expert_scales, groups, 0, torch.bfloat16
    )
    reference = torch.cat(
        (x_dequant[:7] @ dequant.T, x_dequant[7:] @ (dequant * 0.7).T)
    )
    compare("block-FP8 GMM with empty group", output, reference)
    if args.check_empty_rank:
        output = npu_w8a8_block_fp8_grouped_matmul(
            x[:0],
            expert_weights,
            expert_scales,
            torch.zeros_like(groups),
            0,
            torch.bfloat16,
        )
        assert output.shape == (0, rows)
        print(json.dumps({"check": "empty rank GMM", "shape": list(output.shape)}))
    torch.npu.synchronize()
    print("PASS: selected weight checks only; full-model/TP/EP/LSA remain separate.")


if __name__ == "__main__":
    main()
