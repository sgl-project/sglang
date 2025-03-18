""" """

import argparse
import json
import pathlib
import safetensors
import shutil
import subprocess
import sys
import tqdm
import torch

from transformers import AutoTokenizer
from typing import Any


def check_quark():
    try:
        import quark
    except ImportError:
        print("Quark library is not installed. Please follow the instructions at 'README.md'.")
        raise


def load_safetensors(path: str) -> dict:
    return safetensors.torch.load_file(path)


def _run_command(executable: str, cwd: str, args: list[Any]):
    output_lines = []
    try:
        command = [executable, *(str(arg) for arg in args)]
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        # Read output line by line
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)  # Print to console in real-time
            sys.stdout.flush()  # Ensure immediate printing
            output_lines.append(line.strip())  # Store output

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Command '{' '.join(command)}' failed with exit code {process.returncode}\n")
        return "\n".join(output_lines)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command '{command}' failed with exit code {e.returncode}\n" f"Error Output: {e.stderr.strip()}"
        )


def detect_shards(input_path: pathlib.Path) -> int:
    # Auto-detect total_bin and total_shards
    files = sorted(input_path.glob("pytorch_model-*.bin"))

    bin_shard_pairs = set()
    for file in files:
        parts = file.stem.split("-")
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            bin_idx = int(parts[1])
            shard_idx = int(parts[2])
            bin_shard_pairs.add((bin_idx, shard_idx))

    # Get max indices to determine counts
    if not bin_shard_pairs:
        raise RuntimeError("No valid bin-shard files found in the source directory.")

    total_bin = max(b for b, _ in bin_shard_pairs) + 1
    total_shards = max(s for _, s in bin_shard_pairs) + 1

    return total_shards


def preprocess_sharded_model(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    merge_sharded_model(input_path, output_path)
    copy_config(input_path, output_path)
    update_config_act(output_path)
    update_config_custom_model(output_path)
    save_tokenizer_to_path(output_path)


def merge_sharded_model(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    print("Merging sharded model files...")
    weight_map = {}
    total_size = 0

    if any(output_path.iterdir()):
        # not empty directory, assume it's a pre-existing merged model
        print("Skipping. Directory is not empty, assuming it's a pre-existing merged model.")
        return

    # Auto-detect total_bin and total_shards
    files = sorted(input_path.glob("pytorch_model-*.bin"))

    bin_shard_pairs = set()
    for file in files:
        parts = file.stem.split("-")
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            bin_idx = int(parts[1])
            shard_idx = int(parts[2])
            bin_shard_pairs.add((bin_idx, shard_idx))

    # Get max indices to determine counts
    if not bin_shard_pairs:
        raise RuntimeError("No valid bin-shard files found in the source directory.")

    total_bin = max(b for b, _ in bin_shard_pairs) + 1
    total_shards = max(s for _, s in bin_shard_pairs) + 1

    print(f"Detected {total_bin} bins and {total_shards} shards.")

    for i in tqdm.tqdm(range(total_bin), desc="bins"):
        merged_state_dict = {}

        for j in tqdm.tqdm(range(total_shards), desc="shards", leave=False):
            filename = input_path / f"pytorch_model-{i:05d}-{j:03d}.bin"

            if not filename.exists():
                raise FileNotFoundError(f"Missing expected file: {filename}")

            state_dict = torch.load(filename, weights_only=True, map_location=torch.device("cpu"))

            for k, v in state_dict.items():
                weight_map[k] = f"pytorch_model-{i:05d}.bin"

                if j == 0:
                    merged_state_dict[k] = v
                    total_size += torch.numel(v) * v.element_size()
                else:
                    if "w1" in k or "w3" in k:
                        merged_state_dict[k] = torch.cat((merged_state_dict[k], v), dim=0)
                    elif "w2" in k:
                        merged_state_dict[k] = torch.cat((merged_state_dict[k], v), dim=1)
                    elif k not in merged_state_dict or not torch.all(torch.isclose(merged_state_dict[k], v)):
                        raise ValueError(
                            f"Unexpected sharded key: {k} of shape: {v.shape}. Cannot decide which dimension to merge."
                        )

                    total_size += torch.numel(v) * v.element_size()

        torch.save(merged_state_dict, output_path / f"pytorch_model-{i:05d}.bin")

    index_dict = {"metadata": {"total_size": total_size}, "weight_map": weight_map}

    with open(output_path / "pytorch_model.bin.index.json", "w") as f:
        json.dump(index_dict, f)

    print("Processing completed successfully!")


def copy_config(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    print("Copying configuration file...")
    shutil.copy(input_path / "config.json", output_path)


def update_config_act(output_path: pathlib.Path) -> None:
    print("Updating configuration file to use 'gelu' for activation...")
    config_dict = json.load(open(output_path / "config.json", "r"))
    config_dict["hidden_act"] = "gelu"
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4, separators=(",", ": "))


def update_config_custom_model(output_path: pathlib.Path) -> None:
    print("Updating configuration file to make model loadable in huggingface/transformers...")
    files = [
        "configuration_grok1.py",
        "modeling_grok1.py",
        "modeling_grok1_outputs.py",
    ]
    for file in files:
        shutil.copy(pathlib.Path(__file__).parent / "grok1" / file, output_path)
    config_dict = json.load(open(output_path / "config.json", "r"))
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_grok1.Grok1Config",
        "AutoModel": "modeling_grok1.Grok1Model",
        "AutoModelForCausalLM": "modeling_grok1.Grok1ModelForCausalLM",
    }
    config_dict["model_type"] = "grok-1"
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4, separators=(",", ": "))


def save_tokenizer_to_path(output_path: pathlib.Path, tokenizer="Xenova/grok-1-tokenizer") -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True, use_fast=True)
    tokenizer.save_pretrained(output_path)


def update_config_quantization(
    input_path: pathlib.Path, quantized_path: pathlib.Path, output_path: pathlib.Path
) -> None:
    config_dict = json.load(open(input_path / "config.json", "r"))
    qconfig_dist = json.load(open(quantized_path / "config.json", "r"))
    config_dict["quantization_config"] = qconfig_dist["quantization_config"]
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4, separators=(",", ": "))


def quantize(input_path: pathlib.Path, output_path: pathlib.Path, quark_examples_dir: pathlib.Path) -> None:
    args = [
        "quantize_quark.py",
        "--model_dir",
        input_path,
        "--output_dir",
        output_path,
        "--quant_scheme",
        "w_fp8_a_fp8",
        "--kv_cache_dtype",
        "fp8",
        "--num_calib_data",
        128,
        "--model_export",
        "hf_format",
        "--multi_gpu",
        "--custom_mode",
        "fp8",
        "--skip_evaluation",
    ]
    _run_command("python", quark_examples_dir / "torch/language_modeling/llm_ptq", args)


def shard_weight(tensor: torch.Tensor, shard_dim: int, shard_id: int, num_of_shards: int) -> torch.Tensor:
    return torch.chunk(tensor, num_of_shards, dim=shard_dim)[shard_id].clone()


def shard_checkpoint(
    input_path: pathlib.Path, quantized_path: pathlib.Path, output_path: pathlib.Path, total_shards: int
) -> None:
    # Auto-detect total_bin
    q_total_bins = len(list(quantized_path.glob("*.safetensors")))

    # TODO: Inspect if this is needed.
    state_dict = torch.load(input_path / "pytorch_model-00000-000.bin", weights_only=True)
    lm_head_weight = state_dict["lm_head.weight"]

    sharded_keys_and_dims = [
        ("w1.weight", 0),
        ("w1.weight_scale1", 0),
        ("w2.weight", 1),
        ("w3.weight", 0),
        ("w3.weight_scale1", 0),
    ]

    for bin in tqdm.tqdm(range(q_total_bins)):
        state_dict = load_safetensors(quantized_path / f"model-{bin+1:05d}-of-00033.safetensors")
        if bin == 0:
            # transformers skip saving lm_head.weight due to being tied with model.embed_tokens.weight
            # add it back here.
            state_dict["lm_head.weight"] = lm_head_weight

        for shard in range(total_shards):
            shard_state_dict = state_dict.copy()

            for k, v in shard_state_dict.items():
                for key_to_shard, dim in sharded_keys_and_dims:
                    if k.endswith(key_to_shard):
                        shard_state_dict[k] = shard_weight(v, dim, shard, total_shards)

            # Casting scales to float32
            for k, v in shard_state_dict.items():
                if k.find("scale") != -1:
                    shard_state_dict[k] = v.to(torch.float32).clone()

            filename = f"pytorch_model-{bin:05d}-{shard:03d}.bin"
            torch.save(shard_state_dict, output_path / filename)


def postprocess_quantized_model(
    input_path: pathlib.Path, quantized_model_path: pathlib.Path, output_path: pathlib.Path, total_shards: int
) -> None:
    # Step 0: Split the quantized model checkpoint file into shards
    shard_checkpoint(input_path, quantized_model_path, output_path, total_shards)
    # Step 1: Copy config.json file and update quantization configurations
    update_config_quantization(input_path, quantized_model_path, output_path)


def main(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    intermediate_path = pathlib.Path(args.intermediate)
    quark_examples_path = pathlib.Path(args.quark_examples_dir)

    # Step 0: Check if Quark library is installed
    check_quark()

    # Quark is built on top of PyTorch and huggingface/transformers for LLMs.
    # Loading sharded model is not supported directly.
    # Currently, merge sharded model before running quantization.
    unsharded_model_path = intermediate_path / "unsharded"
    unsharded_quantized_model_path = intermediate_path / "unsharded_quantized"

    output_path.mkdir(parents=True, exist_ok=True)
    intermediate_path.mkdir(parents=True, exist_ok=True)
    unsharded_model_path.mkdir(parents=True, exist_ok=True)
    unsharded_quantized_model_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge the sharded model checkpoint file.
    preprocess_sharded_model(input_path, unsharded_model_path)
    # Step 2: Quantize the merged model checkpoint file.
    quantize(unsharded_model_path, unsharded_quantized_model_path, quark_examples_path)
    # Step 3: Postprocess the quantized model checkpoint file into sglang-ready format.
    total_shards = detect_shards(input_path)
    postprocess_quantized_model(input_path, unsharded_quantized_model_path, output_path, total_shards)

    print(f"Quantized model saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize sharded model with Quark.")
    parser.add_argument("--input", type=str, required=True, help="Path to the sharded model checkpoint file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output quantized model checkpoint file")
    parser.add_argument(
        "--intermediate",
        type=str,
        required=True,
        help="Path to the temporary directory to store intermediate unsharded model",
    )
    parser.add_argument(
        "--quark-examples-dir",
        "--quark_examples_dir",
        type=str,
        required=True,
        help="Path to the Quark examples directory",
    )

    args = parser.parse_args()
    main(args)
