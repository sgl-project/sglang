import argparse
import os

DEFAULT_BASE_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
DEFAULT_LORA_PATH = "winddude/wizardLM-LlaMA-LoRA-7B"
DEFAULT_NUM_LORAS = 4


def launch_server(args):
    base_path = args.base_model_path
    lora_path = args.lora_path

    if args.base_only:
        cmd = f"python3 -m sglang.launch_server --model-path {base_path} "
    else:
        if args.num_loras <= 0:
            raise ValueError(
                "--num-loras must be greater than 0 unless --base-only is set"
            )

        cmd = f"python3 -m sglang.launch_server --model-path {base_path} --lora-paths "
        for i in range(args.num_loras):
            lora_name = f"lora{i}"
            cmd += f"{lora_name}={lora_path} "
    cmd += f"--disable-radix "
    cmd += f"--max-loras-per-batch {args.max_loras_per_batch} "
    cmd += f"--max-running-requests {args.max_running_requests} "
    cmd += f"--lora-backend {args.lora_backend} "
    cmd += f"--tp-size {args.tp_size} "
    if args.disable_custom_all_reduce:
        cmd += "--disable-custom-all-reduce"
    if args.enable_mscclpp:
        cmd += "--enable-mscclpp"
    if args.enable_torch_symm_mem:
        cmd += "--enable-torch-symm-mem"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Base model path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=DEFAULT_LORA_PATH,
        help=(
            "LoRA adapter path or Hugging Face model ID used for all registered "
            "LoRA adapters."
        ),
    )
    parser.add_argument(
        "--num-loras",
        type=int,
        default=DEFAULT_NUM_LORAS,
        help=(
            "Number of LoRA adapters to register. For example, 4 registers "
            "lora0, lora1, lora2, and lora3."
        ),
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
    parser.add_argument(
        "--lora-backend",
        type=str,
        default="csgmv",
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
    parser.add_argument(
        "--enable-mscclpp",
        action="store_true",
        help="Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
    )
    parser.add_argument(
        "--enable-torch-symm-mem",
        action="store_true",
        help="Enable using torch symm mem for all-reduce kernel and fall back to NCCL.",
    )
    args = parser.parse_args()

    launch_server(args)
