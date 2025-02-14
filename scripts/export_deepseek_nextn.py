import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


def get_nexn_layer_id(config):
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError("'num_hidden_layers' not found in model config.")
    return config.num_hidden_layers


def update_and_save_config(config, output_dir):
    new_config = config.to_dict()
    new_config.update(
        {
            "architectures": ["DeepseekV3ForCausalLMNextN"],
        }
    )
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_non_safetensors_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        src_file_path = os.path.join(input_dir, filename)
        if os.path.isfile(src_file_path) and not filename.endswith(".safetensors"):
            dst_file_path = os.path.join(output_dir, filename)
            shutil.copy2(src_file_path, dst_file_path)
    print(f"All non-safetensors files have been copied to {output_dir}")


def export_nextn_layer_parameters(input_dir, output_dir, nexn_layer_id):
    prefix = f"model.layers.{nexn_layer_id}"
    output_path = os.path.join(output_dir, "./nextn_layer_parameters.safetensors")
    params = {}
    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors"):
            continue

        file_path = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        try:
            with safe_open(file_path, framework="pt") as f:
                matching_keys = [k for k in f.keys() if k.startswith(prefix)]

                if not matching_keys:
                    print(f"  No parameters starting with '{prefix}' found")
                    continue

                for key in matching_keys:
                    params[key] = f.get_tensor(key)

                save_file(params, output_path)
                print(f"  Saved {len(matching_keys)} parameters to {output_path}")

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")

    if params:
        save_file(params, output_path)
        print(f"Saved {len(params)} parameters to {output_path}")
    else:
        print("No matching parameters found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export NextN layer paramerters for DeepSeek-V3/R1"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input HF model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output nextn model directory.",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.input_dir, trust_remote_code=True)
    nextn_layer_id = get_nexn_layer_id(config)
    os.makedirs(args.output_dir, exist_ok=True)
    copy_non_safetensors_files(args.input_dir, args.output_dir)
    update_and_save_config(config, args.output_dir)
    export_nextn_layer_parameters(args.input_dir, args.output_dir, nextn_layer_id)
