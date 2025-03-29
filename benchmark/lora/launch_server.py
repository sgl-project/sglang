import argparse
import os

NUM_LORAS = 4
LORA_PATH = {
    "base": "meta-llama/Llama-2-7b-hf",
    "lora": "winddude/wizardLM-LlaMA-LoRA-7B",
}


def launch_server(args):
    base_path = LORA_PATH["base"]
    lora_path = LORA_PATH["lora"]

    if args.base_only:
        cmd = f"python3 -m sglang.launch_server --model {base_path} "
    else:
        cmd = f"python3 -m sglang.launch_server --model {base_path} --lora-paths "
        for i in range(NUM_LORAS):
            lora_name = f"lora{i}"
            cmd += f"{lora_name}={lora_path} "
    cmd += f"--disable-radix "
    cmd += f"--max-loras-per-batch {args.max_loras_per_batch} "
    cmd += f"--max-running-requests {args.max_running_requests} "
    cmd += f"--lora-backend {args.lora_backend} "
    cmd += f"--tp-size {args.tp_size} "
    if args.disable_custom_all_reduce:
        cmd += "--disable-custom-all-reduce"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-only",
        action="store_true",
    )
    parser.add_argument(
        "--max-loras-per-batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora-backend",
        type=str,
        default="triton",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for distributed inference",
    )
    # disable_custom_all_reduce
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable custom all reduce when device does not support p2p communication",
    )
    args = parser.parse_args()

    launch_server(args)
