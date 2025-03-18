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

from multiprocessing import Pool, current_process
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


def detect_bins_and_shards(input_path: pathlib.Path) -> tuple[int, int]:
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

    return total_bin, total_shards


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


def update_config_quantization_w4(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    config_dict = json.load(open(input_path / "config.json", "r"))
    config_dict["quantization_config"] = {
        "activation_scheme": "dynamic",
        "kv_cache_scheme": "dynamic",
        "quant_method": "fp8",
        "int4_experts": {"bits": 4, "sym": True, "group": "column"},
    }
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


def quantize_w4a8kv8(
    input_path: pathlib.Path,
    intermediate_path: pathlib.Path,
    output_path: pathlib.Path,
    quark_examples_path: pathlib.Path,
) -> None:
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
    _, total_shards = detect_bins_and_shards(input_path)
    postprocess_quantized_model(input_path, unsharded_quantized_model_path, output_path, total_shards)


def quantize_fp8_scale_tensorwise(w):
    FP8_MAX = 448.0
    scale = w.abs().amax().float() / FP8_MAX
    scaled = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return scaled, scale


def quantize_int4_scale_columnwise(w):
    S4_MAX = 7
    w_flat = w.reshape(-1, w.shape[-1]).float()
    scale = w_flat.abs().amax(axis=-1) / S4_MAX
    scaled = torch.round(w_flat / scale[:, None]).to(torch.int8).clamp(-S4_MAX, S4_MAX)
    return scaled.reshape(w.shape), scale.reshape(w.shape[:-1])


def pack(to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
    if to_pack.ndim > 2:
        raise ValueError("Pack: Only supports tensors with dimensions not greater than 2.")

    if reorder:
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    else:
        order_map = [0, 1, 2, 3, 4, 5, 6, 7]
    pack_num = 8
    if to_pack.ndim == 2:
        packed = torch.zeros(to_pack.shape[0], to_pack.shape[1] // pack_num, dtype=torch.int32, device=to_pack.device)
        new_c = to_pack.shape[1] // pack_num
        for c in range(new_c):
            for i in range(pack_num):
                # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                packed_col = to_pack[:, c * pack_num + order_map[i]].to(torch.int32)
                packed_col = packed_col & 0x0F
                packed[:, c] = torch.bitwise_or(packed[:, c], torch.bitwise_left_shift(packed_col, i * 4))
    elif to_pack.ndim == 0:
        packed = to_pack.to(torch.int32)
    else:
        packed = torch.zeros(to_pack.shape[0] // pack_num, dtype=torch.int32, device=to_pack.device)
        new_c = to_pack.shape[0] // pack_num
        for c in range(new_c):
            for i in range(pack_num):
                # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                packed_col = to_pack[c * pack_num + order_map[i]]
                packed_col = packed_col & 0x0F
                packed[c] = torch.bitwise_or(packed[c], torch.bitwise_left_shift(packed_col, i * 4))

    return packed


def quantize_bin(template_input_path: str, template_output_path: str, total_shards: int) -> None:
    state_dicts = []
    new_state_dicts = []

    # Get the current process name
    process_name = current_process().name
    # Extract the process ID (0-indexed)
    process_id = int(process_name.split("-")[-1]) - 1

    for shard in tqdm.tqdm(range(total_shards), desc="Loading state dicts"):
        state_dicts.append(torch.load(template_input_path.format(shard), map_location=f"cuda:{process_id}"))
        new_state_dicts.append({})

    # All shards have same keys. This assumption must stand.

    for key in tqdm.tqdm(state_dicts[0].keys(), desc="Processing keys"):
        if "w1.weight" in key or "w2.weight" in key or "w3.weight" in key:
            # handle merged bf16 -> fp8 -> int4, and split
            if "w2.weight" in key:
                cat_dim = 1
            else:
                cat_dim = 0
            w = torch.cat([state_dicts[i][key] for i in range(total_shards)], dim=cat_dim)
            fp8_w, fp8_scale = quantize_fp8_scale_tensorwise(w)
            int4_w, int4_scale = quantize_int4_scale_columnwise(w)
            int4_scale /= fp8_scale
            packed_int4_w = pack(int4_w)
            for shard in range(total_shards):

                fp8_scale_key = key.replace("weight", "weight_scale")
                new_state_dicts[shard][fp8_scale_key] = fp8_scale
                int4_scale_key = key.replace("weight", "weight_scale1")
                if "w2.weight" in key:
                    new_state_dicts[shard][int4_scale_key] = int4_scale
                    w_column_shard_size = packed_int4_w.shape[-1] // total_shards
                    new_state_dicts[shard][key] = packed_int4_w[
                        :, shard * w_column_shard_size : (shard + 1) * w_column_shard_size
                    ].clone()
                else:
                    w_row_shard_size = packed_int4_w.shape[0] // total_shards
                    new_state_dicts[shard][int4_scale_key] = int4_scale[
                        shard * w_row_shard_size : (shard + 1) * w_row_shard_size
                    ].clone()
                    new_state_dicts[shard][key] = packed_int4_w[
                        shard * w_row_shard_size : (shard + 1) * w_row_shard_size, :
                    ].clone()
        elif "proj.weight" in key:
            for shard in range(total_shards):
                fp8_w, fp8_scale = quantize_fp8_scale_tensorwise(state_dicts[shard][key])
                fp8_scale_key = key.replace("weight", "weight_scale")
                new_state_dicts[shard][key] = fp8_w.clone()
                new_state_dicts[shard][fp8_scale_key] = fp8_scale
        else:
            for shard in range(total_shards):
                new_state_dicts[shard][key] = state_dicts[shard][key]

    for shard in tqdm.tqdm(range(total_shards), desc="Saving state dicts"):
        torch.save(new_state_dicts[shard], template_output_path.format(shard))


def quantize_weight_only(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    total_bins, total_shards = detect_bins_and_shards(input_path)
    bin_formats = ["pytorch_model-{:05d}-{}.bin".format(i, "{:03d}") for i in range(total_bins)]
    paths = [(str(input_path) + "/" + p, str(output_path) + "/" + p, total_shards) for p in bin_formats]

    output_path.mkdir(parents=True, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    with Pool(processes=num_gpus) as pool:
        pool.starmap(quantize_bin, paths)

    update_config_quantization_w4(input_path, output_path)


def main(args):
    weight_only = args.weight_only
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if weight_only:
        quantize_weight_only(input_path, output_path)
    else:
        if args.intermediate is None:
            raise ValueError("--intermediate must be provided when --weight_only is not set.")
        if args.quark_examples_dir is None:
            raise ValueError("--quark-examples-dir must be provided when --weight_only is not set.")
        intermediate_path = pathlib.Path(args.intermediate)
        quark_examples_path = pathlib.Path(args.quark_examples_dir)
        quantize_w4a8kv8(input_path, intermediate_path, output_path, quark_examples_path)

    print(f"Quantized model saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize sharded model with Quark.")
    parser.add_argument("--input", type=str, required=True, help="Path to the sharded model checkpoint file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output quantized model checkpoint file")
    parser.add_argument(
        "--intermediate",
        type=str,
        help="Path to the temporary directory to store intermediate unsharded model",
    )
    parser.add_argument(
        "--quark-examples-dir",
        "--quark_examples_dir",
        type=str,
        help="Path to the Quark examples directory",
    )
    parser.add_argument(
        "--weight-only",
        "--weight_only",
        action="store_true",
        help="Perform weight-only quantization. Otherwise perform weight-activation-kv-cache quantization.",
    )

    args = parser.parse_args()
    main(args)
