"""Measure reference-only KV sensitivity to kernel and forward shape changes."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for dtype in (torch.bfloat16, torch.float32):
        # Loading under the requested dtype preserves FP32 RoPE frequencies;
        # model.to(dtype=...) would also narrow those non-parameter buffers.
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model_path, dtype=dtype, attn_implementation="eager"
            )
            .cuda()
            .eval()
        )
        for backend in ("eager", "sdpa"):
            model.set_attn_implementation(backend)
            first = None
            for length in (128, 160, 161, 163, 164):
                tokens = ([100, 200, 300, 400] * 40 + [100] * 4)[:length]
                output = model(torch.tensor([tokens], device="cuda"), use_cache=True)
                values = output.past_key_values.layers[14].values[0, :, :128].float()
                if first is None:
                    first = values.clone()
                difference = (values - first).abs()
                print(
                    json.dumps(
                        {
                            "dtype": str(dtype),
                            "backend": backend,
                            "sequence_length": length,
                            "v14_position20_head0_dim118": values[0, 20, 118].item(),
                            "shared_prefix_max_abs_vs_length128": difference.max().item(),
                            "shared_prefix_relative_rms_vs_length128": (
                                difference.square().mean().sqrt()
                                / first.square().mean().sqrt()
                            ).item(),
                        }
                    ),
                    flush=True,
                )
        del model, output, first, values
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
