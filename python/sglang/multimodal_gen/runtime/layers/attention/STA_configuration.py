# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from sglang.multimodal_gen.utils import dict_to_3d_list


def configure_sta(
    mode: str = "STA_searching",
    layer_num: int = 40,
    time_step_num: int = 50,
    head_num: int = 40,
    **kwargs,
) -> list[list[list[Any]]]:
    """
    Configure Sliding Tile Attention (STA) parameters based on the specified mode.

    Parameters:
    ----------
    mode : str
        The STA mode to use. Options are:
        - 'STA_searching': Generate a set of mask candidates for initial search
        - 'STA_tuning': Select best mask strategy based on previously saved results
        - 'STA_inference': Load and use a previously tuned mask strategy
    layer_num: int, number of layers
    time_step_num: int, number of timesteps
    head_num: int, number of heads

    **kwargs : dict
        Mode-specific parameters:

        For 'STA_searching':
        - mask_candidates: list of str, optional, mask candidates to use
        - mask_selected: list of int, optional, indices of selected masks

        For 'STA_tuning':
        - mask_search_files_path: str, required, path to mask search results
        - mask_candidates: list of str, optional, mask candidates to use
        - mask_selected: list of int, optional, indices of selected masks
        - skip_time_steps: int, optional, number of time steps to use full attention (default 12)
        - save_dir: str, optional, directory to save mask strategy (default "mask_candidates")

        For 'STA_inference':
        - load_path: str, optional, path to load mask strategy (default "mask_candidates/mask_strategy.json")
    """
    valid_modes = ["STA_searching", "STA_tuning", "STA_inference", "STA_tuning_cfg"]
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")

    if mode == "STA_searching":
        # Get parameters with defaults
        mask_candidates: list[str] | None = kwargs.get("mask_candidates")
        if mask_candidates is None:
            raise ValueError("mask_candidates is required for STA_searching mode")
        mask_selected: list[int] = kwargs.get(
            "mask_selected", list(range(len(mask_candidates)))
        )

        # Parse selected masks
        selected_masks: list[list[int]] = []
        for index in mask_selected:
            mask = mask_candidates[index]
            masks_list = [int(x) for x in mask.split(",")]
            selected_masks.append(masks_list)

        # Create 3D mask structure with fixed dimensions (t=50, l=60)
        masks_3d: list[list[list[list[int]]]] = []
        for i in range(time_step_num):  # Fixed t dimension = 50
            row = []
            for j in range(layer_num):  # Fixed l dimension = 60
                row.append(selected_masks)  # Add all masks at each position
            masks_3d.append(row)

        return masks_3d

    elif mode == "STA_tuning":
        # Get required parameters
        mask_search_files_path: str | None = kwargs.get("mask_search_files_path")
        if not mask_search_files_path:
            raise ValueError("mask_search_files_path is required for STA_tuning mode")

        # Get optional parameters with defaults
        mask_candidates_tuning: list[str] | None = kwargs.get("mask_candidates")
        if mask_candidates_tuning is None:
            raise ValueError("mask_candidates is required for STA_tuning mode")
        mask_selected_tuning: list[int] = kwargs.get(
            "mask_selected", list(range(len(mask_candidates_tuning)))
        )
        skip_time_steps_tuning: int | None = kwargs.get("skip_time_steps")
        save_dir_tuning: str | None = kwargs.get("save_dir", "mask_candidates")

        # Parse selected masks
        selected_masks_tuning: list[list[int]] = []
        for index in mask_selected_tuning:
            mask = mask_candidates_tuning[index]
            masks_list = [int(x) for x in mask.split(",")]
            selected_masks_tuning.append(masks_list)

        # Read JSON results
        results = read_specific_json_files(mask_search_files_path)
        averaged_results = average_head_losses(results, selected_masks_tuning)

        # Add full attention mask for specific cases
        full_attention_mask_tuning: list[int] | None = kwargs.get("full_attention_mask")
        if full_attention_mask_tuning is not None:
            selected_masks_tuning.append(full_attention_mask_tuning)

        # Select best mask strategy
        timesteps_tuning: int = kwargs.get("timesteps", time_step_num)
        if skip_time_steps_tuning is None:
            skip_time_steps_tuning = 12
        mask_strategy, sparsity, strategy_counts = select_best_mask_strategy(
            averaged_results,
            selected_masks_tuning,
            skip_time_steps_tuning,
            timesteps_tuning,
            head_num,
        )

        # Save mask strategy
        if save_dir_tuning is not None:
            os.makedirs(save_dir_tuning, exist_ok=True)
            file_path = os.path.join(
                save_dir_tuning, f"mask_strategy_s{skip_time_steps_tuning}.json"
            )
            with open(file_path, "w") as f:
                json.dump(mask_strategy, f, indent=4)
            print(f"Successfully saved mask_strategy to {file_path}")

        # Print sparsity and strategy counts for information
        print(f"Overall sparsity: {sparsity:.4f}")
        print("\nStrategy usage counts:")
        total_heads = time_step_num * layer_num * head_num  # Fixed dimensions
        for strategy, count in strategy_counts.items():
            print(f"Strategy {strategy}: {count} heads ({count/total_heads*100:.2f}%)")

        # Convert dictionary to 3D list with fixed dimensions
        mask_strategy_3d = dict_to_3d_list(
            mask_strategy, t_max=time_step_num, l_max=layer_num, h_max=head_num
        )

        return mask_strategy_3d
    elif mode == "STA_tuning_cfg":
        # Get required parameters for both positive and negative paths
        mask_search_files_path_pos: str | None = kwargs.get(
            "mask_search_files_path_pos"
        )
        mask_search_files_path_neg: str | None = kwargs.get(
            "mask_search_files_path_neg"
        )
        save_dir_cfg: str | None = kwargs.get("save_dir")

        if (
            not mask_search_files_path_pos
            or not mask_search_files_path_neg
            or not save_dir_cfg
        ):
            raise ValueError(
                "mask_search_files_path_pos, mask_search_files_path_neg, and save_dir are required for STA_tuning_cfg mode"
            )

        # Get optional parameters with defaults
        mask_candidates_cfg: list[str] | None = kwargs.get("mask_candidates")
        if mask_candidates_cfg is None:
            raise ValueError("mask_candidates is required for STA_tuning_cfg mode")
        mask_selected_cfg: list[int] = kwargs.get(
            "mask_selected", list(range(len(mask_candidates_cfg)))
        )
        skip_time_steps_cfg: int | None = kwargs.get("skip_time_steps")

        # Parse selected masks
        selected_masks_cfg: list[list[int]] = []
        for index in mask_selected_cfg:
            mask = mask_candidates_cfg[index]
            masks_list = [int(x) for x in mask.split(",")]
            selected_masks_cfg.append(masks_list)

        # Read JSON results for both positive and negative paths
        pos_results = read_specific_json_files(mask_search_files_path_pos)
        neg_results = read_specific_json_files(mask_search_files_path_neg)
        # Combine positive and negative results into one list
        combined_results = pos_results + neg_results

        # Average the combined results
        averaged_results = average_head_losses(combined_results, selected_masks_cfg)

        # Add full attention mask for specific cases
        full_attention_mask_cfg: list[int] | None = kwargs.get("full_attention_mask")
        if full_attention_mask_cfg is not None:
            selected_masks_cfg.append(full_attention_mask_cfg)

        timesteps_cfg: int = kwargs.get("timesteps", time_step_num)
        if skip_time_steps_cfg is None:
            skip_time_steps_cfg = 12
        # Select best mask strategy using combined results
        mask_strategy, sparsity, strategy_counts = select_best_mask_strategy(
            averaged_results,
            selected_masks_cfg,
            skip_time_steps_cfg,
            timesteps_cfg,
            head_num,
        )

        # Save mask strategy
        os.makedirs(save_dir_cfg, exist_ok=True)
        file_path = os.path.join(
            save_dir_cfg, f"mask_strategy_s{skip_time_steps_cfg}.json"
        )
        with open(file_path, "w") as f:
            json.dump(mask_strategy, f, indent=4)
        print(f"Successfully saved mask_strategy to {file_path}")

        # Print sparsity and strategy counts for information
        print(f"Overall sparsity: {sparsity:.4f}")
        print("\nStrategy usage counts:")
        total_heads = time_step_num * layer_num * head_num  # Fixed dimensions
        for strategy, count in strategy_counts.items():
            print(f"Strategy {strategy}: {count} heads ({count/total_heads*100:.2f}%)")

        # Convert dictionary to 3D list with fixed dimensions
        mask_strategy_3d = dict_to_3d_list(
            mask_strategy, t_max=time_step_num, l_max=layer_num, h_max=head_num
        )

        return mask_strategy_3d

    else:  # STA_inference
        # Get parameters with defaults
        load_path: str | None = kwargs.get(
            "load_path", "mask_candidates/mask_strategy.json"
        )
        if load_path is None:
            raise ValueError("load_path is required for STA_inference mode")

        # Load previously saved mask strategy
        with open(load_path) as f:
            mask_strategy = json.load(f)

        # Convert dictionary to 3D list with fixed dimensions
        mask_strategy_3d = dict_to_3d_list(
            mask_strategy, t_max=time_step_num, l_max=layer_num, h_max=head_num
        )

        return mask_strategy_3d


# Helper functions


def read_specific_json_files(folder_path: str) -> list[dict[str, Any]]:
    """Read and parse JSON files containing mask search results."""
    json_contents: list[dict[str, Any]] = []

    # List files only in the current directory (no walk)
    files = os.listdir(folder_path)
    # Filter files
    matching_files = [f for f in files if "mask" in f and f.endswith(".json")]
    print(f"Found {len(matching_files)} matching files: {matching_files}")

    for file_name in matching_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path) as file:
            data = json.load(file)
            json_contents.append(data)

    return json_contents


def average_head_losses(
    results: list[dict[str, Any]], selected_masks: list[list[int]]
) -> dict[str, dict[str, np.ndarray]]:
    """Average losses across all prompts for each mask strategy."""
    # Initialize a dictionary to store the averaged results
    averaged_losses: dict[str, dict[str, np.ndarray]] = {}
    loss_type = "L2_loss"
    # Get all loss types (e.g., 'L2_loss')
    averaged_losses[loss_type] = {}

    for mask in selected_masks:
        mask_str = str(mask)
        data_shape = np.array(results[0][loss_type][mask_str]).shape
        accumulated_data = np.zeros(data_shape)

        # Sum across all prompts
        for prompt_result in results:
            accumulated_data += np.array(prompt_result[loss_type][mask_str])

        # Average by dividing by number of prompts
        averaged_data = accumulated_data / len(results)
        averaged_losses[loss_type][mask_str] = averaged_data

    return averaged_losses


def select_best_mask_strategy(
    averaged_results: dict[str, dict[str, np.ndarray]],
    selected_masks: list[list[int]],
    skip_time_steps: int = 12,
    timesteps: int = 50,
    head_num: int = 40,
) -> tuple[dict[str, list[int]], float, dict[str, int]]:
    """Select the best mask strategy for each head based on loss minimization."""
    best_mask_strategy: dict[str, list[int]] = {}
    loss_type = "L2_loss"
    # Get the shape of time steps and layers
    layers = len(averaged_results[loss_type][str(selected_masks[0])][0])

    # Counter for sparsity calculation
    total_tokens = 0  # total number of masked tokens
    total_length = 0  # total sequence length

    strategy_counts: dict[str, int] = {str(strategy): 0 for strategy in selected_masks}
    full_attn_strategy = selected_masks[-1]  # Last strategy is full attention
    print(f"Strategy {full_attn_strategy}, skip first {skip_time_steps} steps ")

    for t in range(timesteps):
        for layer_idx in range(layers):
            for h in range(head_num):
                if t < skip_time_steps:  # First steps use full attention
                    strategy = full_attn_strategy
                else:
                    # Get losses for this head across all strategies
                    head_losses = []
                    for strategy in selected_masks[:-1]:  # Exclude full attention
                        head_losses.append(
                            averaged_results[loss_type][str(strategy)][t][layer_idx][h]
                        )

                    # Find which strategy gives minimum loss
                    best_strategy_idx = np.argmin(head_losses)
                    strategy = selected_masks[best_strategy_idx]

                best_mask_strategy[f"{t}_{layer_idx}_{h}"] = strategy

                # Calculate sparsity
                nums = strategy  # strategy is already a list of numbers
                total_tokens += (
                    nums[0] * nums[1] * nums[2]
                )  # masked tokens for chosen strategy
                total_length += (
                    full_attn_strategy[0]
                    * full_attn_strategy[1]
                    * full_attn_strategy[2]
                )

                # Count strategy usage
                strategy_counts[str(strategy)] += 1

    overall_sparsity = 1 - total_tokens / total_length

    return best_mask_strategy, overall_sparsity, strategy_counts


def save_mask_search_results(
    mask_search_final_result: list[dict[str, list[float]]],
    prompt: str,
    mask_strategies: list[str],
    output_dir: str = "output/mask_search_result/",
) -> str | None:
    if not mask_search_final_result:
        print("No mask search results to save")
        return None

    # Create result dictionary with defaultdict for nested lists
    mask_search_dict: dict[str, dict[str, list[list[float]]]] = {
        "L2_loss": defaultdict(list),
        "L1_loss": defaultdict(list),
    }

    mask_selected = list(range(len(mask_strategies)))
    selected_masks: list[list[int]] = []
    for index in mask_selected:
        mask = mask_strategies[index]
        masks_list = [int(x) for x in mask.split(",")]
        selected_masks.append(masks_list)

    # Process each mask strategy
    for i, mask_strategy in enumerate(selected_masks):
        mask_strategy_str = str(mask_strategy)
        # Process L2 loss
        step_results: list[list[float]] = []
        for step_data in mask_search_final_result:
            if isinstance(step_data, dict) and "L2_loss" in step_data:
                layer_losses = [float(loss) for loss in step_data["L2_loss"]]
                step_results.append(layer_losses)
        mask_search_dict["L2_loss"][mask_strategy_str] = step_results

        step_results = []
        for step_data in mask_search_final_result:
            if isinstance(step_data, dict) and "L1_loss" in step_data:
                layer_losses = [float(loss) for loss in step_data["L1_loss"]]
                step_results.append(layer_losses)
        mask_search_dict["L1_loss"][mask_strategy_str] = step_results

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename based on the first 20 characters of the prompt
    filename = prompt[:50].replace(" ", "_")
    filepath = os.path.join(output_dir, f"mask_search_{filename}.json")

    # Save the results to a JSON file
    with open(filepath, "w") as f:
        json.dump(mask_search_dict, f, indent=4)

    print(f"Successfully saved mask research results to {filepath}")

    return filepath
