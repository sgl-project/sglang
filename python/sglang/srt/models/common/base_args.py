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

"""Common argument parser for VisionFly examples."""

import argparse
import os


class BaseArgumentParser:
    """Base argument parser with common arguments for all VisionFly examples."""

    def __init__(self, description: str = "VisionFly Pipeline Inference"):
        self.parser = argparse.ArgumentParser(description=description)
        self._add_base_args()

    def _add_base_args(self):
        """Add common arguments shared by all examples."""
        # Model and basic generation args
        self.parser.add_argument("--model_path", type=str, help="Model path or HuggingFace model id")
        self.parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
        self.parser.add_argument("--output_path", type=str, default=None, help="Output path")

        # Performance and optimization args
        self.parser.add_argument(
            "--attn_type",
            type=str,
            default="sage-attn",
            choices=[
                "default",
                "auto",
                "fivx",
                "sage-attn",
                "sparse-videogen",
                "sparse-videogen2",
                "trtllm-attn",
                "flash-attn3",
                "flash-attn3-fp8",
                "flash-attn4",
                "te",
                "te-fp8",
            ],
            help="Attention type",
        )
        self.parser.add_argument(
            "--linear_type",
            type=str,
            default="default",
            choices=[
                "default",
                "auto",
                "trtllm-fp8-blockwise",
                "trtllm-fp8-per-tensor",
                "te-fp8-blockwise",
                "te-fp8-per-tensor",
                "te-MXFP8-blockwise-32",
                "trtllm-nvfp4",
                "torch-ao-fp8",
                "svd-nvfp4",
                "flashinfer-nvfp4-trtllm",
                "flashinfer-nvfp4-cudnn",
                "flashinfer-nvfp4-cutlass",
            ],
            help="Linear type",
        )
        self.parser.add_argument(
            "--linear_recipe",
            type=str,
            default="dynamic",
            choices=[
                "dynamic",
                "static",
            ],
            help="Linear recipe for autotuning",
        )
        self.parser.add_argument(
            "--autotuner_result_dir",
            type=str,
            default=None,
            help="The directory to save the autotuner results, by default, we will save the result under the `./autotuner` directory",
        )
        self.parser.add_argument(
            "--skip_autotuning",
            action="store_true",
            help="Skip autotuning if the result already exists",
        )
        self.parser.add_argument(
            "--attn_choices",
            type=str,
            default="default,sage-attn",
            help="Attention choices for autotuning",
        )
        self.parser.add_argument(
            "--attn_cosine_similarity_threshold",
            type=float,
            default=0.999,
            help="Cosine similarity threshold for autotuning",
        )
        self.parser.add_argument(
            "--attn_mse_threshold",
            type=float,
            default=0.002,
            help="MSE threshold for autotuning",
        )
        self.parser.add_argument(
            "--linear_choices",
            type=str,
            default="default,trtllm-fp8-per-tensor,trtllm-fp8-blockwise,trtllm-nvfp4,svd-nvfp4",
            help="Linear choices for autotuning",
        )
        self.parser.add_argument(
            "--linear_cosine_similarity_threshold",
            type=float,
            default=0.999,
            help="Cosine similarity threshold for autotuning",
        )
        self.parser.add_argument(
            "--linear_mse_threshold",
            type=float,
            default=0.01,
            help="MSE threshold for autotuning",
        )
        # Parallel processing args
        self.parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
        self.parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
        self.parser.add_argument("--ulysses", type=int, default=1, help="Ulysses parallel size")
        self.parser.add_argument("--ring", type=int, default=1, help="Ring parallel size")
        self.parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
        self.parser.add_argument("--cfg", type=int, default=1, help="CFG parallel size")
        self.parser.add_argument("--fsdp", type=int, default=1, help="FSDP size")
        self.parser.add_argument("--refiner_dp", type=int, default=1, help="Data parallel size")
        self.parser.add_argument("--refiner_tp", type=int, default=1, help="Tensor parallel size")
        self.parser.add_argument("--refiner_ulysses", type=int, default=1, help="Ulysses parallel size")
        self.parser.add_argument("--refiner_ring", type=int, default=1, help="Ring parallel size")
        self.parser.add_argument("--refiner_cp", type=int, default=1, help="Context parallel size")
        self.parser.add_argument("--refiner_cfg", type=int, default=1, help="CFG parallel size")
        self.parser.add_argument("--refiner_fsdp", type=int, default=1, help="FSDP size")
        self.parser.add_argument("--t5_fsdp", type=int, default=1, help="T5 FSDP size")

        # Torch compile args
        self.parser.add_argument("--disable_torch_compile", action="store_true", help="Use torch compile")
        self.parser.add_argument(
            "--torch_compile_models",
            type=str,
            default="transformer",
            help="Models to compile with torch compile (comma-separated list)",
        )
        self.parser.add_argument(
            "--torch_compile_mode",
            type=str,
            default="default",
            choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
            help="Torch compile mode",
        )

        # CudaGraph
        self.parser.add_argument("--enable_cuda_graph", action="store_true", help="Enable cuda graph")

        # CPU offload args
        self.parser.add_argument(
            "--enable_sequential_cpu_offload", action="store_true", help="Enable sequential CPU offload"
        )
        self.parser.add_argument("--enable_model_cpu_offload", action="store_true", help="Enable model CPU offload")
        self.parser.add_argument("--enable_vfly_cpu_offload", action="store_true", help="Enable vfly cpu offload")
        self.parser.add_argument(
            "--vfly_block_cpu_offload_stride",
            type=int,
            default=1,
            help="The stride of block cpu offload. Larger stride means less offloading blocks and thus cost more GPU memory.",
        )

        # VAE parallel args
        self.parser.add_argument("--disable_parallel_vae", action="store_true", help="Disable parallel VAE")
        self.parser.add_argument(
            "--parallel_vae_split_dim",
            type=str,
            default="width",
            choices=["height", "width"],
            help="Split dimension for parallel vae",
        )

        # TeaCache args
        self.parser.add_argument("--enable_teacache", action="store_true", help="Enable teacache")
        self.parser.add_argument("--teacache_thresh", type=float, default=0.2, help="Threshold for teacache")
        self.parser.add_argument("--use_ret_steps", action="store_true", help="Use ret steps for teacache")

        # QKV fusion args
        self.parser.add_argument(
            "--disable_qkv_fusion", action="store_true", help="Don't fuse qkv, by default we will fuse qkv"
        )

        # Generation args
        self.parser.add_argument("--prompt", type=str, default="", help="Text prompt for generation")
        self.parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for generation")
        self.parser.add_argument("--height", type=int, default=512, help="Image/Video height")
        self.parser.add_argument("--width", type=int, default=512, help="Image/Video width")
        self.parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
        self.parser.add_argument("--shift", type=int, default=5, help="Inference shift")
        self.parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
        self.parser.add_argument("--num_warmup_steps", type=int, default=1, help="Number of warmup steps")

        # Sparse attention args
        self.parser.add_argument(
            "--num_timesteps_high_precision",
            type=float,
            default=0.0,
            help="When timestep < num_inference_steps * num_timesteps_high_precision, use high precision attention operators",
        )
        self.parser.add_argument(
            "--num_layers_high_precision",
            type=float,
            default=0.0,
            help="When layer idx < num_layers * num_layers_high_precision, use high precision attention operators",
        )
        self.parser.add_argument(
            "--high_precision_attn_type",
            type=str,
            default="default",
            choices=["default", "sage-attn"],
            help="High precision attention type to fallback",
        )
        self.parser.add_argument("--sparsity", type=float, default=0.25, help="Sparsity for sparse attention")
        self.parser.add_argument(
            "--max_sequence_length", type=int, default=512, help="Max sequence length of text encoder"
        )

        # svd args
        self.parser.add_argument("--svd_fp4_checkpoint_path", type=str, default="", help="SVD FP4 checkpoint path")

        # multiple prompts args
        self.parser.add_argument(
            "--multiple_prompts", type=bool, default=False, help="Enable multiple prompts in json format as input"
        )

        # whether to enable profile
        self.parser.add_argument("--profile", action="store_true", help="Enable profile")

        # quantize args
        self.parser.add_argument("--export_vfly_dit", action="store_true", help="Enable export quantized DIT")
        self.parser.add_argument("--load_vfly_dit", action="store_true", help="Load quantized DIT")
        self.parser.add_argument("--vfly_ckpt_path", type=str, default="vfly_dit", help="Quantized DIT path")

        # svg2 args
        self.parser.add_argument(
            "--num_q_centroids", "--qc", type=int, default=100, help="Number of query centroids for KMEANS_BLOCK."
        )
        self.parser.add_argument(
            "--num_k_centroids", "--kc", type=int, default=500, help="Number of key centroids for KMEANS_BLOCK."
        )
        self.parser.add_argument(
            "--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in KMEANS_BLOCK."
        )
        self.parser.add_argument(
            "--min_kc_ratio",
            type=float,
            default=0.1,
            help="At least this proportion of key blocks to keep per query block in KMEANS_BLOCK.",
        )
        self.parser.add_argument(
            "--kmeans_iter_init",
            type=int,
            default=50,
            help="Number of KMeans iterations for initialization in KMEANS_BLOCK.",
        )
        self.parser.add_argument(
            "--kmeans_iter_step",
            type=int,
            default=2,
            help="Number of KMeans iterations for other diffusion steps in KMEANS_BLOCK.",
        )

    def add_video_args(self, default_num_frames: int = 33, default_fps: int = 16):
        """Add video generation arguments."""
        self.parser.add_argument(
            "--num_frames", type=int, default=default_num_frames, help="Number of frames to generate"
        )
        self.parser.add_argument("--fps", type=int, default=default_fps, help="FPS for exported video")

    def add_image_input_args(self, default_image: str = ""):
        """Add image input arguments for image-to-video."""
        self.parser.add_argument("--image", type=str, default=default_image, help="Input image URL or local path")

    def set_defaults(self, **kwargs):
        """Set default values for arguments."""
        self.parser.set_defaults(**kwargs)

    def parse_args(self, args=[]):  # wili, add parameter `args`
        """Parse and return arguments."""
        args = self.parser.parse_args(args)
        world_size = int(os.getenv("WORLD_SIZE", 1))
        rank = int(os.getenv("RANK", 0))
        if world_size == 1 or (world_size > 1 and rank == 0):
            print("-" * 50)
            print("Parsed arguments:")
            for arg in vars(args):
                print(f"  {arg:<30} : {getattr(args, arg)}")
            print("-" * 50)
        return args
