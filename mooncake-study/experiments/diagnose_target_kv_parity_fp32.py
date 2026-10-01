"""FP32 diagnostic for the KV draft; this does not certify the BF16 serving path."""

import argparse
import json
from unittest.mock import patch

import torch
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import dflash
from sglang.test import dspark_target_kv_parity as parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument(
        "--reference-attention",
        choices=("eager", "sdpa", "flex_attention"),
        default="eager",
    )
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original_runner = parity.ServingKVParityRunner
    original_check = parity.check_fixed_input_parity
    original_qk = dflash.apply_qk_norm

    def fp32_runner(*args, embed, head, **kwargs):
        serving = original_runner(
            *args, embed=embed.float(), head=head.float(), dtype=torch.float32, **kwargs
        )
        # The production fused RMSNorm only accepts narrow activation dtypes.
        for module in serving.model.modules():
            if isinstance(module, (RMSNorm, SiluAndMul)):
                module.forward = module.forward_native
        for layer in serving.model.layers:
            rotary = layer.self_attn.rotary_emb
            rotary.forward = rotary.forward_native
        return serving

    def fp32_check(reference, *args, **kwargs):
        return original_check(reference.float(), *args, **kwargs)

    def native_qk(*args, **kwargs):
        return original_qk(*args, **kwargs, allow_inplace=False)

    with (
        parity.single_gpu_parity_context(),
        patch.object(parity, "ServingKVParityRunner", fp32_runner),
        patch.object(parity, "check_fixed_input_parity", fp32_check),
        patch.object(dflash, "apply_qk_norm", native_qk),
        patch(
            "transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward",
            side_effect=AssertionError("target decoder is forbidden"),
        ),
    ):
        report = parity._validate_captured_checkpoint(
            args.checkpoint,
            args.target_path,
            attention_backend=args.reference_attention,
        )
        print(
            json.dumps(
                {
                    **report,
                    "diagnostic_only": True,
                    "dtype": "float32",
                    "native_auxiliary_kernels": True,
                    "serving_attention": "triton",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
