# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utility functions for VisionFly examples."""

import math
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.profiler
from diffusers.utils import export_to_video
from PIL import Image
from torch.profiler import ProfilerActivity

from vfly.utils import create_default_vfly_configs
from vfly.utils.auto_tuner import AutoTuner
from vfly.utils.logger import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int, deterministic: bool):
    """Set random seed for reproducible results."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Setup distributed training environment."""
    local_rank = 0
    world_size = 1
    if dist.is_initialized():
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
    return local_rank, world_size


def configure_cpu_offload(pipe, args, local_rank: int, model_wise=None, block_wise=None):
    """Configure CPU offload for the pipeline."""
    if args.enable_sequential_cpu_offload:
        if not args.disable_torch_compile:
            logger.error("Sequential CPU offload and torch compile are not supported together")
            raise ValueError("Sequential CPU offload and torch compile are not supported together")
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logger.info("enabled diffusers' sequential CPU offload")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logger.info("enabled diffusers' model CPU offload")
    elif args.enable_vfly_cpu_offload:
        if args.linear_type == "auto":
            raise NotImplementedError("vfly cpu offload does not support auto linear type")
        assert args.vfly_block_cpu_offload_stride > 0, "vfly_block_cpu_offload_stride must be greater than 0"

        pipe.enable_vfly_cpu_offload(
            model_wise=model_wise,
            block_wise=block_wise,
            offloading_stride=args.vfly_block_cpu_offload_stride,
        )
        logger.info(f"enabled vfly's cpu offload, offloading stride: {args.vfly_block_cpu_offload_stride}")
    else:
        pipe.to(f"cuda:{local_rank}")


def create_vfly_config(args) -> Dict[str, Any]:
    """Create VFly configuration from arguments."""
    configs = create_default_vfly_configs()
    new_configs = {
        "pipeline": {
            "enable_torch_compile": not args.disable_torch_compile,
            "torch_compile_models": args.torch_compile_models.split(","),
            "torch_compile_mode": getattr(args, "torch_compile_mode", "default"),
            "fuse_qkv": not args.disable_qkv_fusion,
        },
        "attn": {
            "type": args.attn_type,
            "choices": args.attn_choices.split(","),
            "cosine_similarity_threshold": args.attn_cosine_similarity_threshold,
            "mse_threshold": args.attn_mse_threshold,
            "sparsity": getattr(args, "sparsity", 0.25),
            "num_timesteps_high_precision": getattr(args, "num_timesteps_high_precision", 0.0),
            "num_layers_high_precision": getattr(args, "num_layers_high_precision", 0.0),
            "high_precision_attn_type": getattr(args, "high_precision_attn_type", "default"),
            "num_q_centroids": args.num_q_centroids,
            "num_k_centroids": args.num_k_centroids,
            "top_p_kmeans": args.top_p_kmeans,
            "min_kc_ratio": args.min_kc_ratio,
            "kmeans_iter_init": args.kmeans_iter_init,
            "kmeans_iter_step": args.kmeans_iter_step,
        },
        "linear": {
            "type": args.linear_type,
            "recipe": args.linear_recipe,
            "choices": args.linear_choices.split(","),
            "cosine_similarity_threshold": args.linear_cosine_similarity_threshold,
            "mse_threshold": args.linear_mse_threshold,
        },
        "parallel": {
            "disable_parallel_vae": args.disable_parallel_vae,
            "parallel_vae_split_dim": args.parallel_vae_split_dim,
            "dit_dp_size": args.dp,
            "dit_tp_size": args.tp,
            "dit_ulysses_size": args.ulysses,
            "dit_ring_size": args.ring,
            "dit_cp_size": args.cp,
            "dit_cfg_size": args.cfg,
            "dit_fsdp_size": args.fsdp,
            "refiner_dit_dp_size": args.refiner_dp,
            "refiner_dit_tp_size": args.refiner_tp,
            "refiner_dit_ulysses_size": args.refiner_ulysses,
            "refiner_dit_ring_size": args.refiner_ring,
            "refiner_dit_cp_size": args.refiner_cp,
            "refiner_dit_cfg_size": args.refiner_cfg,
            "refiner_dit_fsdp_size": args.refiner_fsdp,
            "t5_fsdp_size": args.t5_fsdp,
        },
        "teacache": {
            "enable_teacache": args.enable_teacache,
            "use_ret_steps": args.use_ret_steps,
            "teacache_thresh": args.teacache_thresh,
        },
    }

    configs.update(new_configs)

    return configs


def generate_output_path(args, output_dir: str = "output", save_type: str = "png") -> str:
    """Generate output path based on arguments."""
    if args.output_path is not None:
        return args.output_path

    model_name = args.model_path.split("/")[-1]

    # Common parameters
    params = [
        f"attn-{args.attn_type}",
        f"linear-{args.linear_type}",
        f"h{args.height}",
        f"w{args.width}",
    ]

    if hasattr(args, "num_frames"):
        params.append(f"frames{args.num_frames}")
    if hasattr(args, "fps"):
        params.append(f"fps{args.fps}")

    params.extend(
        [
            f"dp{args.dp}",
            f"tp{args.tp}",
            f"ulysses{args.ulysses}",
            f"ring{args.ring}",
            f"cp{args.cp}",
            f"cfg{args.cfg}",
            f"fsdp{args.fsdp}",
            f"t5_fsdp{args.t5_fsdp}",
            f"teacache{args.enable_teacache}",
            f"compile{not args.disable_torch_compile}",
            f"cg{args.enable_cuda_graph}",
            f"offload{args.enable_vfly_cpu_offload}",
            f"fuse_qkv{not args.disable_qkv_fusion}",
            f"parallelvae{not args.disable_parallel_vae}",
            f"seed{args.random_seed}",
        ]
    )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    return os.path.join(output_dir, f"{model_name}-{'-'.join(params)}.{save_type}")


def generate_autotuner_dir(args, parent_dir: str = "autotuner") -> str:
    if parent_dir is not None:
        os.makedirs(parent_dir, exist_ok=True)

    model_name = args.model_path.split("/")[-1]
    params = [
        f"attn-{args.attn_type}",
        f"linear-{args.linear_type}",
        f"h{args.height}",
        f"w{args.width}",
    ]

    if hasattr(args, "num_frames"):
        params.append(f"frames{args.num_frames}")
    if hasattr(args, "fps"):
        params.append(f"fps{args.fps}")

    params.extend(
        [
            f"ulysses{args.ulysses}",
            f"ring{args.ring}",
            f"teacache{args.enable_teacache}",
            f"fuse_qkv{not args.disable_qkv_fusion}",
            f"seed{args.random_seed}",
        ]
    )

    if parent_dir is None:
        autotuner_dir = f"{model_name}-{'-'.join(params)}"
    else:
        autotuner_dir = os.path.join(parent_dir, f"{model_name}-{'-'.join(params)}")

    os.makedirs(autotuner_dir, exist_ok=True)

    return autotuner_dir


def recompute_shape_for_vae(height, width, vae_scale_factor, patch_size, vae_parallel_dim="height"):
    """Recompute the shape(height, width) for vae
    1. All the dimensions(h, w) must be divisible by the vae_scale_factor and patch_size
    2. The dimension for parallel vae must be divisible by the world_size
    """
    world_size = int(os.getenv("WORLD_SIZE", 1))
    # 1. All the dimensions(h, w) must be divisible by the vae_scale_factor and patch_size
    mod_value1 = vae_scale_factor * patch_size
    # 2. The dimension for parallel vae must be divisible by the world_size * vae_scale_factor * world_size,
    # i.e., (dim / vae_scale_factor) % world_size == 0
    mod_value2 = mod_value1
    if world_size > 1:
        parallel_stride = world_size * vae_scale_factor
        mod_value2 = (parallel_stride * mod_value1) // math.gcd(parallel_stride, mod_value1)

    if vae_parallel_dim == "height":
        updated_height = (height // mod_value2) * mod_value2
        updated_width = (width // mod_value1) * mod_value1
    elif vae_parallel_dim == "width":
        updated_width = (width // mod_value2) * mod_value2
        updated_height = (height // mod_value1) * mod_value1
    else:
        raise ValueError(f"Invalid vae parallel dim: {vae_parallel_dim}")

    if updated_height != height or updated_width != width:
        logger.warning(f"Recomputed output shape(hxw) for vae: {height}x{width} -> {updated_height}x{updated_width}")
    return updated_height, updated_width


def calculate_dimensions(image: Optional[Image.Image], args, pipe) -> tuple[int, int]:
    """Calculate optimal dimensions for generation."""
    if not hasattr(args, "height") or not hasattr(args, "width"):
        return None, None

    if image is None:
        return args.height, args.width

    max_area = args.height * args.width
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    local_rank, world_size = setup_distributed()

    if not args.disable_parallel_vae:
        stride = pipe.vae_scale_factor_spatial * world_size
        lcm_stride = abs(stride * mod_value) // math.gcd(stride, mod_value)
        if args.parallel_vae_split_dim == "height":
            height = (round(np.sqrt(max_area * aspect_ratio)) // lcm_stride) * lcm_stride
            width = (round(np.sqrt(max_area / aspect_ratio)) // mod_value) * mod_value
        elif args.parallel_vae_split_dim == "width":
            width = (round(np.sqrt(max_area / aspect_ratio)) // lcm_stride) * lcm_stride
            height = (round(np.sqrt(max_area * aspect_ratio)) // mod_value) * mod_value
        else:
            raise ValueError(f"Invalid parallel vae split dim: {args.parallel_vae_split_dim}")
    else:
        height = (round(np.sqrt(max_area * aspect_ratio)) // mod_value) * mod_value
        width = (round(np.sqrt(max_area / aspect_ratio)) // mod_value) * mod_value

    return height, width


def autotuning(func, autotuner_dir, *args, **kwargs):
    with AutoTuner(mode="tuning", result_dir=autotuner_dir):
        _ = func(*args, **kwargs)


def benchmark_inference(
    func,
    warmup: bool = True,
    profile: bool = False,
    random_seed: int = None,
    enable_autotuner: bool = False,
    autotuner_dir: str = None,
    deterministic: bool = True,
    *args,
    **kwargs,
):
    """Benchmark inference function with CUDA timing."""
    if enable_autotuner:
        assert autotuner_dir is not None, "autotuner_dir must be provided when enable_autotuner is True"

    if warmup:
        logger.info("Warmup pipeline")
        if enable_autotuner:
            with AutoTuner(mode="inference", result_dir=autotuner_dir):
                _ = func(*args, **kwargs, warmup=True)
        else:
            _ = func(*args, **kwargs, warmup=True)

    if profile:
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # record_shapes=True,
            # profile_memory=True,
            # with_stack=True
        ) as profiler:
            _ = func(*args, **kwargs, warmup=True)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                profiler.export_chrome_trace("./trace.json")
        dist.barrier()

    if random_seed is not None:
        seed_everything(random_seed, deterministic=deterministic)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    if enable_autotuner:
        with AutoTuner(mode="inference", result_dir=autotuner_dir):
            result = func(*args, **kwargs)
    else:
        result = func(*args, **kwargs)
    torch.cuda.synchronize()
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

    logger.info(f"Inference completed in {elapsed_time:.3f} seconds")

    return result, elapsed_time


def save_output(output, output_path: str, output_type: str = "auto", fps: int = 16):
    """Save output (image or video) with distributed training support."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            _save_output_single(output, output_path, output_type, fps)
        dist.barrier()
    else:
        _save_output_single(output, output_path, output_type, fps)


def _save_output_single(output, output_path: str, output_type: str, fps: int):
    """Save output for single process."""
    logger.info(f"Saving output to {output_path}")

    if output_type == "auto":
        output_type = "video" if output_path.endswith((".mp4", ".avi", ".mov")) else "image"

    if output_type == "video":
        export_to_video(output, output_path, fps=fps)
    else:  # image
        output.save(output_path)


def log_args_and_timing(args, elapsed_time: float):
    """Log all arguments and timing information."""
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"Time taken: {elapsed_time:.3f} s")


def validate_parallel_config(args):
    """Validate parallel configuration arguments."""
    assert args.tp == 1, "Tensor parallel not supported yet"
    if args.attn_type not in ["sage-attn", "flash-attn3", "flash-attn3-fp8"]:
        assert args.ring == 1, f"Ring Attention parallel not supported yet for {args.attn_type}"
