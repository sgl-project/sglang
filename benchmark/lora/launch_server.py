import argparse
import os

NUM_LORAS = 128
LORA_PATH = {
    "base": "mistralai/Mistral-7B-Instruct-v0.3",
    "lora": "/home/ying/test_lora",
}


def launch_server(args):
    base_path = LORA_PATH["base"]
    lora_path = LORA_PATH["lora"]
    max_loras_per_batch = 4

    if args.base_only:
        cmd = f"python -m sglang.launch_server --model {base_path} "
    else:
        cmd = f"python -m sglang.launch_server --model {base_path} --lora-paths "
        for i in range(NUM_LORAS):
            lora_name = f"lora{i}"
            cmd += f"{lora_name}={lora_path} "
    cmd += f"--disable-radix --disable-cuda-graph "
    cmd += f"--max-loras-per-batch {args.max_loras_per_batch} "
    cmd += f"--max-running-requests {args.max_running_requests}"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-loras",
        type=int,
        default=128,
    )
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
    args = parser.parse_args()

    launch_server(args)
