# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import argparse
import os
from typing import Any


def update_config_from_args(
    config: Any, args_dict: dict[str, Any], prefix: str = "", pop_args: bool = False
) -> bool:
    """
    Update configuration object from arguments dictionary.

    Args:
        config: The configuration object to update
        args_dict: Dictionary containing arguments
        prefix: Prefix for the configuration parameters in the args_dict.
               If None, assumes direct attribute mapping without prefix.
    """
    # Handle top-level attributes (no prefix)
    args_not_to_remove = [
        "model_path",
    ]
    args_to_remove = []
    if prefix.strip() == "":
        for key, value in args_dict.items():
            if hasattr(config, key) and value is not None:
                if key == "text_encoder_precisions" and isinstance(value, list):
                    setattr(config, key, tuple(value))
                else:
                    setattr(config, key, value)
                if pop_args:
                    args_to_remove.append(key)
    else:
        # Handle nested attributes with prefix
        prefix_with_dot = f"{prefix}."
        for key, value in args_dict.items():
            if key.startswith(prefix_with_dot) and value is not None:
                attr_name = key[len(prefix_with_dot) :]
                if hasattr(config, attr_name):
                    setattr(config, attr_name, value)
                if pop_args:
                    args_to_remove.append(key)

    if pop_args:
        for key in args_to_remove:
            if key not in args_not_to_remove:
                args_dict.pop(key)

    return len(args_to_remove) > 0


def clean_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    """
    Clean the arguments by removing the ones that not explicitly provided by the user.
    """
    provided_args = {}
    for k, v in vars(args).items():
        if v is not None and hasattr(args, "_provided") and k in args._provided:
            provided_args[k] = v

    return provided_args


def _list_onnx_files(dir_path: str) -> list[str]:
    try:
        entries = os.listdir(dir_path)
    except OSError:
        return []
    return [
        os.path.join(dir_path, name)
        for name in entries
        if name.endswith(".onnx") and os.path.isfile(os.path.join(dir_path, name))
    ]


def _list_end2end_onnx_files(dir_path: str) -> list[str]:
    try:
        entries = os.listdir(dir_path)
    except OSError:
        return []
    return [
        os.path.join(dir_path, name)
        for name in entries
        if name == "end2end.onnx" and os.path.isfile(os.path.join(dir_path, name))
    ]


def _list_onnx_files_recursive(base_dir: str) -> list[str]:
    matches = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name.endswith(".onnx"):
                matches.append(os.path.join(root, name))
    return matches


def _list_end2end_onnx_files_recursive(base_dir: str) -> list[str]:
    matches = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name == "end2end.onnx":
                matches.append(os.path.join(root, name))
    return matches


def resolve_wan_preprocess_model_paths(
    preprocess_model_path: str,
) -> tuple[str | None, str | None]:
    if not preprocess_model_path:
        return None, None

    base_dir = preprocess_model_path
    if os.path.isdir(os.path.join(base_dir, "process_checkpoint")):
        base_dir = os.path.join(base_dir, "process_checkpoint")

    det_candidates = []
    for subdir in ("det", "detector", "detection"):
        det_candidates.extend(_list_onnx_files(os.path.join(base_dir, subdir)))
    pose_candidates = []
    for subdir in ("pose2d", "pose", "pose_2d"):
        pose_dir = os.path.join(base_dir, subdir)
        pose_candidates.extend(_list_end2end_onnx_files(pose_dir))
        if not pose_candidates and os.path.isdir(pose_dir):
            pose_candidates.extend(_list_end2end_onnx_files_recursive(pose_dir))

    if not det_candidates or not pose_candidates:
        all_candidates = _list_onnx_files_recursive(base_dir)
        all_pose_candidates = _list_end2end_onnx_files_recursive(base_dir)
        if not det_candidates:
            det_candidates = all_candidates
        if not pose_candidates:
            pose_candidates = all_pose_candidates

    det_path = det_candidates[0] if det_candidates else None
    pose_path = pose_candidates[0] if pose_candidates else None
    return det_path, pose_path
