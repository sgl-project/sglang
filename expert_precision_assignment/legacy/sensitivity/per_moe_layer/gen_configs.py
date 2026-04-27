"""Generate one heter-precision config per MoE layer for sensitivity sweep.

For layer L under test:
  - int4_only_experts: all 128 experts of layer L (no BF16 copy)
  - bf16_only_experts: all 128 experts of every OTHER layer (no INT4 loaded)
This means only layer L gets INT4 weights loaded, saving ~16 GB of VRAM.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write heter_layer{L}.json files into.")
    ap.add_argument("--num_layers", type=int, default=48,
                    help="Total decoder layers.")
    ap.add_argument("--moe_layer_step", type=int, default=1,
                    help="1 if every layer is MoE (Qwen3-30B-A3B default).")
    ap.add_argument("--num_experts", type=int, default=128)
    ap.add_argument("--int4_checkpoint", type=str, default=(
        "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/"
        "snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"))
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--threshold", type=int, default=128,
                    help="expert_batch policy threshold.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    moe_layers = [i for i in range(args.num_layers)
                  if i % args.moe_layer_step == 0]
    all_experts = list(range(args.num_experts))

    for L in moe_layers:
        # Layer L: all experts are INT4-only (quantized, no BF16 copy)
        int4_only_path = os.path.join(args.out_dir, f"int4_only_layer{L}.json")
        with open(int4_only_path, "w") as f:
            json.dump({str(L): all_experts}, f)

        # All other MoE layers: all experts are BF16-only (no INT4 loaded)
        bf16_only = {str(other): all_experts
                     for other in moe_layers if other != L}
        bf16_only_path = os.path.join(args.out_dir, f"bf16_only_layer{L}.json")
        with open(bf16_only_path, "w") as f:
            json.dump(bf16_only, f)

        cfg = {
            "groups": [
                {"name": "cold", "num_bits": 4,
                 "group_size": args.group_size,
                 "checkpoint": args.int4_checkpoint},
                {"name": "hot", "num_bits": 16},
            ],
            "policy": "expert_batch",
            "policy_params": {"threshold": args.threshold},
            "int4_only_experts_file": int4_only_path,
            "bf16_only_experts_file": bf16_only_path,
        }
        cfg_path = os.path.join(args.out_dir, f"heter_layer{L}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"Wrote {len(moe_layers)} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
